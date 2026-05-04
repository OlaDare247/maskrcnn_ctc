from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from maskrcnn_ctc.datasets.register import register_sim


def build_cfg(config_file, weights, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda"
    return cfg


def load_gt_mask(gt_dir, img_path):
    suffix = img_path.stem[1:]
    gt_path = gt_dir / f"man_seg{suffix}.tif"
    if not gt_path.exists():
        return None
    return tiff.imread(str(gt_path))


def instances_to_mask(instances, shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint16)

    if not instances.has("pred_masks"):
        return mask

    masks = instances.pred_masks.cpu().numpy()
    for i, m in enumerate(masks):
        mask[m > 0] = i + 1

    return mask


def overlay(img, mask, color=(0, 255, 0)):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = img.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    register_sim()

    cfg = build_cfg(args.config, args.weights, args.score_thresh)
    predictor = DefaultPredictor(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(Path(args.image_dir).glob("*.tif"))[: args.max_images]

    for img_path in images:
    	img = tiff.imread(str(img_path))
    	img = np.squeeze(img)

    	if img.ndim != 2:
           raise ValueError(f"Expected 2D grayscale image, got shape {img.shape} for {img_path}")

    	img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    	img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

    	outputs = predictor(img_rgb)
    	pred_mask = instances_to_mask(outputs["instances"].to("cpu"), img_u8.shape)

    	gt_mask = load_gt_mask(Path(args.gt_dir), img_path)
    	if gt_mask is None:
            gt_mask = np.zeros_like(pred_mask)

    	pred_overlay = overlay(img_u8, pred_mask, (0, 255, 0))
    	gt_overlay = overlay(img_u8, gt_mask, (255, 0, 0))

    	base_vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    	combined = np.hstack([base_vis, gt_overlay, pred_overlay])

    	out_path = out_dir / f"{img_path.stem}.png"
    	cv2.imwrite(str(out_path), combined)

    	print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()