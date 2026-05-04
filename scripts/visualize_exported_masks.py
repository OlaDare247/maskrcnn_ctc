from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
from skimage.measure import regionprops


def overlay(img, mask, color):
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    layer = base.copy()
    layer[mask > 0] = color
    return cv2.addWeighted(base, 0.7, layer, 0.3, 0)


def draw_boxes_from_mask(img_u8, mask, color=(0, 255, 255)):
    out = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    for region in regionprops(mask.astype(np.int32)):
        y1, x1, y2, x2 = region.bbox
        area = region.area

        if area < 5:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            out,
            str(region.label),
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(image_dir.glob("*.tif"))[: args.max_images]

    for img_path in imgs:
        suffix = img_path.stem[1:]
        gt_path = gt_dir / f"man_seg{suffix}.tif"
        pred_path = pred_dir / f"mask{suffix}.tif"

        if not pred_path.exists():
            print(f"Missing prediction: {pred_path}")
            continue

        img = np.squeeze(tiff.imread(str(img_path)))
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        gt = tiff.imread(str(gt_path)) if gt_path.exists() else np.zeros_like(img_u8, dtype=np.uint16)
        pred = tiff.imread(str(pred_path))

        base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        gt_overlay = overlay(img_u8, gt, (255, 0, 0))
        pred_overlay = overlay(img_u8, pred, (0, 255, 0))
        pred_boxes = draw_boxes_from_mask(img_u8, pred, (0, 255, 255))

        panel = np.hstack([base, gt_overlay, pred_overlay, pred_boxes])

        out_path = out_dir / f"{img_path.stem}_compare_boxes.png"
        cv2.imwrite(str(out_path), panel)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()