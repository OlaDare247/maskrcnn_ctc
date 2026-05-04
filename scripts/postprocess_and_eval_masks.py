from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage.measure import label
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk


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


def relabel_connected(mask_bin: np.ndarray) -> np.ndarray:
    lab = label(mask_bin > 0)
    return lab.astype(np.uint16)


def postprocess_mask(mask: np.ndarray, min_size: int, hole_area: int, closing_radius: int) -> np.ndarray:
    mask_bin = mask > 0

    if min_size > 0:
        mask_bin = remove_small_objects(mask_bin, min_size=min_size)

    if hole_area > 0:
        mask_bin = remove_small_holes(mask_bin, area_threshold=hole_area)

    if closing_radius > 0:
        mask_bin = binary_closing(mask_bin, footprint=disk(closing_radius))

    return relabel_connected(mask_bin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--hole-area", type=int, default=20)
    parser.add_argument("--closing-radius", type=int, default=1)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("mask*.tif"))
    rows = []

    for pred_path in pred_files:
        suffix = pred_path.stem.replace("mask", "")
        gt_path = gt_dir / f"man_seg{suffix}.tif"

        pred_mask = tiff.imread(str(pred_path))
        gt_mask = tiff.imread(str(gt_path)) if gt_path.exists() else np.zeros_like(pred_mask)

        pp_mask = postprocess_mask(
            pred_mask,
            min_size=args.min_size,
            hole_area=args.hole_area,
            closing_radius=args.closing_radius,
        )

        dice, jacc = dice_jaccard_binary(pp_mask, gt_mask)

        out_mask = out_dir / pred_path.name
        tiff.imwrite(str(out_mask), pp_mask.astype(np.uint16))

        rows.append({
            "pred_mask": pred_path.name,
            "dice": dice,
            "jaccard": jacc,
            "num_pred_instances": int(pp_mask.max()),
        })

        print(f"{pred_path.name}: dice={dice:.4f}  jacc={jacc:.4f}")

    mean_dice = float(np.mean([r["dice"] for r in rows])) if rows else 0.0
    mean_jacc = float(np.mean([r["jaccard"] for r in rows])) if rows else 0.0

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "mean_dice": mean_dice,
            "mean_jaccard": mean_jacc,
            "params": {
                "min_size": args.min_size,
                "hole_area": args.hole_area,
                "closing_radius": args.closing_radius,
            },
            "per_image": rows,
        }, f, indent=2)

    print("\nPOSTPROCESS DONE")
    print(f"mean_dice    = {mean_dice:.6f}")
    print(f"mean_jaccard = {mean_jacc:.6f}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
PY