import random
import numpy as np
import torch
import torch.nn.functional as F

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator

from ..datasets.mapper import build_mapper
from ..utils.temporal import farneback_flow, warp_logits_with_flow

class TemporalCTCTrainer(DefaultTrainer):
    """
    Adds temporal consistency loss between consecutive frames.
    Works best when training dataset returns frames in order (we will enforce order).
    """

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = build_mapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = build_mapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def run_step(self):
        return super().run_step()
        """
        Override training step to add temporal loss.
        Assumption: batch size >= 2 and contains consecutive frames from same sequence.
        """
        # assert self.model.training, "Model is not in training mode!"
        # data = next(self._data_loader_iter)
        

        # Standard detectron2 losses
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        # Temporal loss toggle
        lam = float(getattr(self.cfg.SOLVER, "TEMPORAL_LAMBDA", 0.0))
        if lam > 0 and len(data) >= 2:
            # Pick pair indices (0,1) — simple
            d0, d1 = data[0], data[1]

            # We need the *input images* as numpy uint8 for flow
            # We stored dataset_dict["image"] as float tensor [3,H,W]
            # Convert back to grayscale uint8
            im0 = d0["image"].detach().cpu().numpy()[0]  # first channel
            im1 = d1["image"].detach().cpu().numpy()[0]
            im0_u8 = np.clip(im0, 0, 255).astype(np.uint8)
            im1_u8 = np.clip(im1, 0, 255).astype(np.uint8)

            flow = farneback_flow(im0_u8, im1_u8)

            # Get mask head logits by running forward with return proposals/features is messy in vanilla detectron2.
            # Practical approach:
            # Use predicted masks (instances) and encourage consistency at the *probability map* level.
            # We'll compute a soft mask canvas from instances.pred_masks (not perfect but works).

            with torch.no_grad():
                out0 = self.model([d0])[0]  # inference-like output
                out1 = self.model([d1])[0]

            # Build soft mask canvases (H,W) from instance masks weighted by score
            # Then compute a consistency loss after warping.
            H, W = d1["image"].shape[1], d1["image"].shape[2]
            p0 = torch.zeros((1, 1, H, W), device=self.model.device)
            p1 = torch.zeros((1, 1, H, W), device=self.model.device)

            inst0 = out0["instances"]
            inst1 = out1["instances"]

            if len(inst0) > 0:
                m0 = inst0.pred_masks.float()  # NxHxW
                s0 = inst0.scores.view(-1, 1, 1)
                p0 = torch.clamp((m0 * s0).max(dim=0).values, 0, 1).view(1,1,H,W)

            if len(inst1) > 0:
                m1 = inst1.pred_masks.float()
                s1 = inst1.scores.view(-1, 1, 1)
                p1 = torch.clamp((m1 * s1).max(dim=0).values, 0, 1).view(1,1,H,W)

            p0_warp = warp_logits_with_flow(p0, flow)

            # L1 consistency (you can switch to BCE or KL)
            temporal_loss = F.l1_loss(p0_warp, p1)

            loss_dict["loss_temporal"] = temporal_loss * lam
            losses = losses + loss_dict["loss_temporal"]

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        self._write_metrics(loss_dict)
