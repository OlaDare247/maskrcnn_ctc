from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    order = np.argsort(-scores)
    current_id = 1
    for idx in order:
        m = masks[idx] > 0
        label_mask[m] = current_id
        current_id += 1

    return label_mask


def dice_jaccard_binary(pred, gt):
    pred_bin = pred > 0
    gt_bin = gt > 0

    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    dice = (2.0 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
    jacc = inter / union if union > 0 else 1.0
    return float(dice), float(jacc)


def evaluate_threshold(
    predictor,
    image_dir: Path,
    gt_dir: Path,
    out_dir: Path,
    max_images: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.tif"))
    if max_images > 0:
        image_paths = image_paths[:max_images]

    rows = []

    for i, img_path in enumerate(image_paths):
        img = tiff.imread(str(img_path))
        img = np.squeeze(img)
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

        outputs = predictor(img_rgb)
        instances = outputs["instances"].to("cpu")
        pred_mask = instances_to_labelmask(instances, img_u8.shape)

        suffix = img_path.stem[1:]
        gt_path = gt_dir / f"man_seg{suffix}.tif"
        gt_mask = tiff.imread(str(gt_path)) if gt_path.exists() else np.zeros_like(pred_mask)

        dice, jacc = dice_jaccard_binary(pred_mask, gt_mask)

        out_mask = out_dir / f"mask{suffix}.tif"
        tiff.imwrite(str(out_mask), pred_mask.astype(np.uint16))

        rows.append({
            "image": img_path.name,
            "pred_mask": out_mask.name,
            "dice": dice,
            "jaccard": jacc,
            "num_pred_instances": int(pred_mask.max()),
        })

    mean_dice = float(np.mean([r["dice"] for r in rows])) if rows else 0.0
    mean_jacc = float(np.mean([r["jaccard"] for r in rows])) if rows else 0.0

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "num_images": len(rows),
            "mean_dice": mean_dice,
            "mean_jaccard": mean_jacc,
            "per_image": rows,
        }, f, indent=2)

    return mean_dice, mean_jacc, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    register_sim()

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for thr in args.thresholds:
        print(f"\n=== Evaluating threshold {thr:.3f} ===")
        cfg = build_cfg(args.config, args.weights, thr)
        predictor = DefaultPredictor(cfg)

        tag = f"thr_{str(thr).replace('.', 'p')}"
        out_dir = out_root / tag

        mean_dice, mean_jacc, rows = evaluate_threshold(
            predictor=predictor,
            image_dir=image_dir,
            gt_dir=gt_dir,
            out_dir=out_dir,
            max_images=args.max_images,
        )

        print(f"threshold={thr:.3f}  mean_dice={mean_dice:.6f}  mean_jaccard={mean_jacc:.6f}")

        summary.append({
            "threshold": thr,
            "mean_dice": mean_dice,
            "mean_jaccard": mean_jacc,
            "num_images": len(rows),
            "folder": str(out_dir),
        })

    summary = sorted(summary, key=lambda x: (-x["mean_dice"], -x["mean_jaccard"]))

    with open(out_root / "threshold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST RESULTS ===")
    for row in summary:
        print(
            f"thr={row['threshold']:.3f}  "
            f"dice={row['mean_dice']:.6f}  "
            f"jacc={row['mean_jaccard']:.6f}  "
            f"folder={row['folder']}"
        )


if __name__ == "__main__":
    main()
PY