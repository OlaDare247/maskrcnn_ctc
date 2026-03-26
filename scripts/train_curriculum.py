import os
import argparse

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

from maskrcnn_ctc.datasets.register import register_msc
from maskrcnn_ctc.engine.trainer_temporal import TemporalCTCTrainer

def make_cfg(config_file, output_dir=None, train_name=None, test_name=None, temporal_lambda=0.0):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    zoo_cfg = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_cfg)

    if output_dir:
        cfg.OUTPUT_DIR = output_dir

    if train_name:
        cfg.DATASETS.TRAIN = (train_name,)
    if test_name:
        cfg.DATASETS.TEST = (test_name,)

    # Add temporal lambda into cfg
    cfg.SOLVER.TEMPORAL_LAMBDA = float(temporal_lambda)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def train_one(cfg, resume=False):
    trainer = TemporalCTCTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

def main(args):
    setup_logger()
    train_ds, val_ds = register_msc()

    # Stage 1: train on ST
    cfg1 = make_cfg(
        args.config,
        output_dir=args.out_stage1,
        train_name="msc01_train_ST",
        test_name="msc01_val_GT",
        temporal_lambda=0.0
    )
    train_one(cfg1, resume=False)

    # Stage 2: fine-tune on GT (sparse)
    cfg2 = make_cfg(
        args.config,
        output_dir=args.out_stage2,
        train_name="msc01_val_GT",   # yes: train on sparse GT frames
        test_name="msc01_val_GT",
        temporal_lambda=0.0
    )
    # initialize from stage1 final
    cfg2.MODEL.WEIGHTS = os.path.join(cfg1.OUTPUT_DIR, "model_final.pth")
    train_one(cfg2, resume=False)

    # Stage 3: temporal regularization fine-tune
    cfg3 = make_cfg(
        args.config,
        output_dir=args.out_stage3,
        train_name="msc01_train_ST",
        test_name="msc01_val_GT",
        temporal_lambda=args.temporal_lambda
    )
    cfg3.MODEL.WEIGHTS = os.path.join(cfg2.OUTPUT_DIR, "model_final.pth")
    train_one(cfg3, resume=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-stage1", default="/mnt/c/Users/ooluwadare/maskrcnn_ctc/outputs/stage1_ST")
    ap.add_argument("--out-stage2", default="/mnt/c/Users/ooluwadare/maskrcnn_ctc/outputs/stage2_GT")
    ap.add_argument("--out-stage3", default="/mnt/c/Users/ooluwadare/maskrcnn_ctc/outputs/stage3_TEMP")
    ap.add_argument("--temporal-lambda", type=float, default=0.2)
    args = ap.parse_args()
    main(args)
