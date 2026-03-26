import argparse
import numpy as np
from pathlib import Path
import tifffile as tiff

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from maskrcnn_ctc.datasets.ctc_io import save_label_tiff
from maskrcnn_ctc.datasets.paths import CTC_ROOT, PROJECT_ROOT

def load_gray(path: Path):
    im = tiff.imread(str(path))
    if im.ndim == 3:
        im = im[..., 0]
    return im

def to_3ch(im):
    return np.repeat(im[:, :, None], 3, axis=2)

def instances_to_label(instances, H, W):
    """
    Convert Detectron2 instances -> single label image (1..N).
    Resolve overlaps by score ordering (highest wins).
    """
    if len(instances) == 0:
        return np.zeros((H, W), dtype=np.uint16)

    masks = instances.pred_masks.cpu().numpy().astype(bool)
    scores = instances.scores.cpu().numpy()
    order = np.argsort(-scores)

    label = np.zeros((H, W), dtype=np.uint16)
    cur_id = 1
    for idx in order:
        m = masks[idx]
        # write only where empty
        write = m & (label == 0)
        if write.any():
            label[write] = cur_id
            cur_id += 1
    return label

def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)

    zoo_cfg = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.MODEL.WEIGHTS = args.weights if args.weights else model_zoo.get_checkpoint_url(zoo_cfg)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh
    cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)

    seq_dir = CTC_ROOT / args.seq
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(seq_dir.glob("t*.tif"))
    print(f"Found {len(imgs)} frames")

    for i, p in enumerate(imgs):
        im = load_gray(p)
        H, W = im.shape
        im3 = to_3ch(im)

        outputs = predictor(im3)
        inst = outputs["instances"].to("cpu")
        label = instances_to_label(inst, H, W)

        # CTC expects mask like mask000.tif, etc.
        out_path = out_dir / f"mask{i:03d}.tif"
        save_label_tiff(out_path, label)

    print("Done. Saved to:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None, help="Path to model_final.pth (optional)")
    ap.add_argument("--seq", default="01", choices=["01", "02"])
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "01_Msk_MaskRCNN"))
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()
    main(args)
