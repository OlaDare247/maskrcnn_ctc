from __future__ import annotations

import argparse

from maskrcnn_ctc.config_utils import build_cfg
from maskrcnn_ctc.datasets.register import register_msc, register_sim, register_hela
from maskrcnn_ctc.engine.trainer_temporal import TemporalCTCTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["msc", "sim", "hela"],
        help="Which registered dataset pair to use",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.dataset == "msc":
        train_name, val_name = register_msc()
        output_dir = "/home/users/ooluwadare/maskrcnn_ctc/outputs/msc_maskrcnn"
    elif args.dataset == "sim":
        train_name, val_name = register_sim()
        output_dir = "/home/users/ooluwadare/maskrcnn_ctc/outputs/sim_maskrcnn"
    else:
        train_name, val_name = register_hela()
        output_dir = "/home/users/ooluwadare/maskrcnn_ctc/outputs/hela_maskrcnn"

    cfg = build_cfg(
        config_file=args.config,
        output_dir=output_dir,
        train_names=[train_name],
        test_names=[val_name],
    )

    trainer = TemporalCTCTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()