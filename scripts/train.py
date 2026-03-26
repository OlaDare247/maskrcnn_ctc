import os
import argparse

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

from maskrcnn_ctc.datasets.register import register_msc
from maskrcnn_ctc.engine.trainer import CTCTrainer

def build_cfg(config_file: str, output_dir: str = None):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # Use model zoo backbone init
    zoo_cfg = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_cfg)

    if output_dir is not None:
        cfg.OUTPUT_DIR = output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main(args):
    setup_logger()
    register_msc()

    cfg = build_cfg(args.config, args.output_dir)
    trainer = CTCTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
