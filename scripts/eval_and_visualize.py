import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch
from pycocotools import mask as mask_util

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

from maskrcnn_ctc.datasets.register import register_msc


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def dice_score(pred, gt, eps=1e-8):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter) / (pred.sum() + gt.sum() + eps)


def jaccard_score(pred, gt, eps=1e-8):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + eps)


def seg_to_mask(segmentation, height, width):
    """
    Supports COCO RLE dicts and polygon lists.
    Returns uint8 mask of shape [H, W].
    """
    if isinstance(segmentation, dict):
        # COCO RLE
        m = mask_util.decode(segmentation)
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m > 0).astype(np.uint8)

    if isinstance(segmentation, list):
        # Polygon list
        rles = mask_util.frPyObjects(segmentation, height, width)
        rle = mask_util.merge(rles)
        m = mask_util.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m > 0).astype(np.uint8)

    raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")


def annos_to_gt_label_mask(annotations, height, width):
    """
    Builds a labeled GT mask from annotation segmentations.
    Background = 0, each object gets unique integer label.
    """
    label_mask = np.zeros((height, width), dtype=np.uint16)
    label_id = 1
    for anno in annotations:
        seg = anno.get("segmentation", None)
        if seg is None:
            continue
        m = seg_to_mask(seg, height, width).astype(bool)
        if m.sum() == 0:
            continue
        # overwrite is okay for GT objects that do not overlap
        label_mask[m] = label_id
        label_id += 1
    return label_mask


# def instances_to_label_mask(pred_masks, scores=None):
#     """
#     Convert N predicted instance masks into one labeled image.
#     Highest-score masks are assigned first.
#     """
#     if len(pred_masks) == 0:
#         return None

#     h, w = pred_masks[0].shape
#     label_mask = np.zeros((h, w), dtype=np.uint16)
#     occupied = np.zeros((h, w), dtype=bool)

#     order = np.arange(len(pred_masks))
#     if scores is not None and len(scores) == len(pred_masks):
#         order = np.argsort(scores)[::-1]

#     label_id = 1
#     for idx in order:
#         m = pred_masks[idx].astype(bool)
#         # avoid double-labeling overlapping pixels
#         m = np.logical_and(m, ~occupied)
#         if m.sum() == 0:
#             continue
#         label_mask[m] = label_id
#         occupied[m] = True
#         label_id += 1

#     return label_mask

def instances_to_label_mask(pred_masks, scores=None, min_area=0, topk=0):
    if len(pred_masks) == 0:
        return None

    h, w = pred_masks[0].shape
    label_mask = np.zeros((h, w), dtype=np.uint16)
    occupied = np.zeros((h, w), dtype=bool)

    order = np.arange(len(pred_masks))
    if scores is not None and len(scores) == len(pred_masks):
        order = np.argsort(scores)[::-1]

    if topk > 0:
        order = order[:topk]

    label_id = 1
    for idx in order:
        m = pred_masks[idx].astype(bool)
        if m.sum() < min_area:
            continue
        m = np.logical_and(m, ~occupied)
        if m.sum() == 0:
            continue
        label_mask[m] = label_id
        occupied[m] = True
        label_id += 1

    return label_mask

def load_image_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 2:
        img8 = normalize_to_uint8(img)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 1:
        img8 = normalize_to_uint8(img[:, :, 0])
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        return img

    raise ValueError(f"Unsupported image shape: {img.shape}")


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    out = 255.0 * (img - mn) / (mx - mn)
    return out.astype(np.uint8)


def frame_id_from_path(path):
    """
    Extract digits from filename.
    t000.tif -> 000
    image_12.png -> 12
    """
    stem = Path(path).stem
    nums = re.findall(r"\d+", stem)
    if nums:
        return int(nums[-1])
    return None


def save_comparison_panel(orig_bgr, gt_label, pred_label, overlay_bgr, save_path):
    gt_vis = label_to_color(gt_label)
    pred_vis = label_to_color(pred_label)

    panel = cv2.hconcat([
        orig_bgr,
        gt_vis,
        pred_vis,
        overlay_bgr
    ])
    cv2.imwrite(str(save_path), panel)


def label_to_color(label_mask):
    """
    Colorize a labeled mask for visualization.
    """
    h, w = label_mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_mask)
    ids = ids[ids != 0]

    rng = np.random.default_rng(12345)
    lut = {0: np.array([0, 0, 0], dtype=np.uint8)}
    for i in ids:
        lut[int(i)] = rng.integers(50, 255, size=3, dtype=np.uint8)

    for i in ids:
        color[label_mask == i] = lut[int(i)]

    return color


def run_external_metric(cmd_template, pred_dir, gt_dir, metric_name):
    if not cmd_template:
        return None

    cmd = cmd_template.format(pred_dir=str(pred_dir), gt_dir=str(gt_dir))
    print(f"\n[INFO] Running {metric_name} command:")
    print(cmd)

    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"\n[{metric_name} stdout]\n{proc.stdout}")
    if proc.stderr.strip():
        print(f"\n[{metric_name} stderr]\n{proc.stderr}")

    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def build_cfg(config_path, weights_path, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--weights", required=True, help="Path to model_final.pth or checkpoint")
    parser.add_argument("--dataset", default="msc01_val_GT", help="Dataset name to evaluate")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Prediction score threshold")
    parser.add_argument("--topk", type=int, default=0, help="keep only top-k predicted masks by score; 0 means keep all")
    parser.add_argument("--min-area", type=int, default=0, help="minimum mask area")
    parser.add_argument("--seg-cmd", default="", help="Optional external SEG command template with {pred_dir} and {gt_dir}")
    parser.add_argument("--det-cmd", default="", help="Optional external DET command template with {pred_dir} and {gt_dir}")
    args = parser.parse_args()

    register_msc()

    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlays"
    binary_dir = out_dir / "pred_binary_masks"
    inst_dir = out_dir / "pred_instance_masks"
    gt_dir = out_dir / "gt_instance_masks"
    panel_dir = out_dir / "comparison_panels"
    ctc_dir = out_dir / "ctc_masks"

    for d in [overlay_dir, binary_dir, inst_dir, gt_dir, panel_dir, ctc_dir]:
        ensure_dir(d)

    cfg = build_cfg(args.config, args.weights, args.score_thresh)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(args.dataset)
    metadata = MetadataCatalog.get(args.dataset)

    rows = []

    for idx, record in enumerate(dataset_dicts):
        file_name = record["file_name"]
        height = record["height"]
        width = record["width"]
        image_id = record["image_id"]

        img_bgr = load_image_bgr(file_name)
        outputs = predictor(img_bgr)
        instances = outputs["instances"].to("cpu")

        scores = instances.scores.numpy() if instances.has("scores") else np.array([])
        pred_masks = instances.pred_masks.numpy().astype(np.uint8) if instances.has("pred_masks") else np.zeros((0, height, width), dtype=np.uint8)

        pred_label = instances_to_label_mask(pred_masks, scores=scores, min_area=args.min_area, topk=args.topk)
        if pred_label is None:
            pred_label = np.zeros((height, width), dtype=np.uint16)

        pred_binary = (pred_label > 0).astype(np.uint8)

        gt_label = annos_to_gt_label_mask(record.get("annotations", []), height, width)
        gt_binary = (gt_label > 0).astype(np.uint8)

        dice = dice_score(pred_binary, gt_binary)
        jacc = jaccard_score(pred_binary, gt_binary)

        num_pred = int(pred_label.max())
        num_gt = int(gt_label.max())

        # To confirm if there are predicted mask
        print(f"[DEBUG] raw={len(pred_masks)} topk={args.topk} min_area={args.min_area}")

        # To confirm predicated labels are built
        print(f"[DEBUG] kept={int(pred_label.max()) if pred_label is not None else 0}")

        vis = Visualizer(
            img_bgr[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW
        )
        vis_out = vis.draw_instance_predictions(instances)
        overlay_rgb = vis_out.get_image()
        overlay_bgr = overlay_rgb[:, :, ::-1]

        frame_id = frame_id_from_path(file_name)
        if frame_id is None:
            frame_stub = f"{idx:03d}"
        else:
            frame_stub = f"{frame_id:03d}"

        overlay_path = overlay_dir / f"overlay_{frame_stub}.png"
        binary_path = binary_dir / f"mask_{frame_stub}.png"
        inst_path = inst_dir / f"mask_{frame_stub}.tif"
        gt_path = gt_dir / f"mask_{frame_stub}.tif"
        panel_path = panel_dir / f"panel_{frame_stub}.png"
        ctc_path = ctc_dir / f"mask{frame_stub}.tif"

        cv2.imwrite(str(overlay_path), overlay_bgr)
        cv2.imwrite(str(binary_path), (pred_binary * 255).astype(np.uint8))
        tiff.imwrite(str(inst_path), pred_label.astype(np.uint16))
        tiff.imwrite(str(gt_path), gt_label.astype(np.uint16))
        tiff.imwrite(str(ctc_path), pred_label.astype(np.uint16))
        save_comparison_panel(img_bgr, gt_label, pred_label, overlay_bgr, panel_path)

        rows.append({
            "image_id": image_id,
            "file_name": file_name,
            "frame_stub": frame_stub,
            "num_pred_instances": num_pred,
            "num_gt_instances": num_gt,
            "dice": float(dice),
            "jaccard": float(jacc),
            "mean_score": float(scores.mean()) if len(scores) else 0.0,
            "max_score": float(scores.max()) if len(scores) else 0.0,
        })

        print(
            f"[{idx+1}/{len(dataset_dicts)}] {Path(file_name).name} | "
            f"pred={num_pred} gt={num_gt} dice={dice:.4f} jacc={jacc:.4f}"
        )

    csv_path = out_dir / "per_frame_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "image_id", "file_name", "frame_stub", "num_pred_instances",
            "num_gt_instances", "dice", "jaccard", "mean_score", "max_score"
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    if rows:
        summary = {
            "dataset": args.dataset,
            "weights": str(args.weights),
            "num_images": len(rows),
            "mean_dice": float(np.mean([r["dice"] for r in rows])),
            "mean_jaccard": float(np.mean([r["jaccard"] for r in rows])),
            "mean_num_pred_instances": float(np.mean([r["num_pred_instances"] for r in rows])),
            "mean_num_gt_instances": float(np.mean([r["num_gt_instances"] for r in rows])),
            "score_thresh": args.score_thresh,
            "output_dir": str(out_dir),
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    seg_result = run_external_metric(args.seg_cmd, ctc_dir, gt_dir, "SEG") if args.seg_cmd else None
    det_result = run_external_metric(args.det_cmd, ctc_dir, gt_dir, "DET") if args.det_cmd else None

    if seg_result or det_result:
        ext_path = out_dir / "external_metric_runs.json"
        with open(ext_path, "w") as f:
            json.dump({"SEG": seg_result, "DET": det_result}, f, indent=2)


if __name__ == "__main__":
    main()