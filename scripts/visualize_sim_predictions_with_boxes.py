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



def build_cfg(config_file: str, weights: str, score_thresh: float):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda"
    return cfg


def load_gt_mask(gt_dir: Path, img_path: Path):
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
    scores = instances.scores.cpu().numpy() if instances.has("scores") else np.ones(len(masks))

    order = np.argsort(-scores)
    current_id = 1
    for idx in order:
        m = masks[idx] > 0
        mask[m] = current_id
        current_id += 1

    return mask


def overlay_mask(img_u8, mask, color):
    base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    layer = base.copy()
    layer[mask > 0] = color
    return cv2.addWeighted(base, 0.7, layer, 0.3, 0)


def draw_boxes(img_u8, instances, color=(0, 255, 255)):
    out = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    if not instances.has("pred_boxes"):
        return out

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy() if instances.has("scores") else np.ones(len(boxes))

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            out,
            f"{scores[i]:.2f}",
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    register_sim()
    cfg = build_cfg(args.config, args.weights, args.score_thresh)
    predictor = DefaultPredictor(cfg)

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.tif"))[: args.max_images]

    for img_path in image_paths:
        img = tiff.imread(str(img_path))
        img = np.squeeze(img)

        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got {img.shape} for {img_path}")

        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

        outputs = predictor(img_rgb)
        instances = outputs["instances"].to("cpu")

        pred_mask = instances_to_mask(instances, img_u8.shape)
        gt_mask = load_gt_mask(gt_dir, img_path)
        if gt_mask is None:
            gt_mask = np.zeros_like(pred_mask)

        panel_img = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        panel_gt = overlay_mask(img_u8, gt_mask, (255, 0, 0))      # blue
        panel_pred = overlay_mask(img_u8, pred_mask, (0, 255, 0))  # green
        panel_boxes = draw_boxes(img_u8, instances, (0, 255, 255)) # yellow

        combined = np.hstack([panel_img, panel_gt, panel_pred, panel_boxes])

        out_path = out_dir / f"{img_path.stem}_boxes.png"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
PY