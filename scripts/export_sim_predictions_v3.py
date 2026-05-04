from __future__ import annotations

import argparse
import json
from pathlib import Path
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max  


import cv2
import numpy as np
import tifffile as tiff
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from maskrcnn_ctc.datasets.register import register_sim


def build_cfg(config_file: str, weights: str, score_thresh: float):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda"
    return cfg


def instances_to_labelmask(instances, shape):
    h, w = shape
    label_mask = np.zeros((h, w), dtype=np.uint16)

    if not instances.has("pred_masks"):
        return label_mask

    masks = instances.pred_masks.cpu().numpy()
    scores = instances.scores.cpu().numpy() if instances.has("scores") else np.ones(len(masks))

    # Sort high-confidence first
    order = np.argsort(-scores)

    current_id = 1
    for idx in order:
        m = masks[idx] > 0

        # Only write into empty pixels so earlier instances are preserved
        write_region = m & (label_mask == 0)

        if write_region.sum() == 0:
            continue

        label_mask[write_region] = current_id
        current_id += 1

    return label_mask

def refine_instance_mask(label_mask, min_size=10, min_distance=None):
    binary = label_mask > 0

    if binary.sum() == 0:
        return label_mask.astype(np.uint16)

    distance = ndi.distance_transform_edt(binary)

    # Adaptive min distance based on estimated object size
    avg_radius = max(3, int(np.sqrt(binary.sum() / max(1, label(binary).max())) / 2))
    if min_distance is None:
        min_distance = min(avg_radius, 8)

    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=1.5,
        labels=binary
    )

    markers = np.zeros_like(distance, dtype=np.int32)

    if len(coords) == 0:
        markers = label(binary)
    else:
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

    refined = watershed(
        -distance,
        markers,
        mask=binary
    )

    out = np.zeros_like(refined, dtype=np.uint16)
    current_id = 1

    for obj_id in np.unique(refined):
        if obj_id == 0:
            continue

        comp = refined == obj_id

        if comp.sum() < min_size:
            continue

        out[comp] = current_id
        current_id += 1

    return out.astype(np.uint16)
def dice_jaccard_binary(pred, gt):
    pred_bin = pred > 0
    gt_bin = gt > 0

    inter = np.logical_and(pred_bin, gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    dice = (2.0 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
    jacc = inter / union if union > 0 else 1.0
    return float(dice), float(jacc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    register_sim()
    cfg = build_cfg(args.config, args.weights, args.score_thresh)
    predictor = DefaultPredictor(cfg)

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.tif"))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    rows = []

    for i, img_path in enumerate(image_paths):
        img = tiff.imread(str(img_path))
        img = np.squeeze(img)
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

        outputs = predictor(img_rgb)
        instances = outputs["instances"].to("cpu")

        pred_mask = instances_to_labelmask(instances, img_u8.shape)
        pred_mask = refine_instance_mask(pred_mask, min_size=10, min_distance=5)

        stem = img_path.stem
        suffix = stem[1:]
        gt_path = gt_dir / f"man_seg{suffix}.tif"
        gt_mask = tiff.imread(str(gt_path)) if gt_path.exists() else np.zeros_like(pred_mask)

        dice, jacc = dice_jaccard_binary(pred_mask, gt_mask)

        out_mask = out_dir / f"mask{suffix}.tif"
        tiff.imwrite(str(out_mask), pred_mask.astype(np.uint16))

        rows.append({
            "image": img_path.name,
            "gt": gt_path.name if gt_path.exists() else "",
            "pred_mask": out_mask.name,
            "num_pred_instances": int(pred_mask.max()),
            "num_gt_instances": int(len(np.unique(gt_mask)) - (1 if 0 in np.unique(gt_mask) else 0)),
            "dice": dice,
            "jaccard": jacc,
        })

        print(f"[{i+1}/{len(image_paths)}] saved {out_mask.name}  dice={dice:.4f}  jacc={jacc:.4f}")

    mean_dice = float(np.mean([r["dice"] for r in rows])) if rows else 0.0
    mean_jacc = float(np.mean([r["jaccard"] for r in rows])) if rows else 0.0

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "num_images": len(rows),
            "mean_dice": mean_dice,
            "mean_jaccard": mean_jacc,
            "per_image": rows,
        }, f, indent=2)

    print("\nDONE")
    print(f"mean_dice    = {mean_dice:.6f}")
    print(f"mean_jaccard = {mean_jacc:.6f}")
    print(f"saved masks to {out_dir}")


if __name__ == "__main__":
    main()
