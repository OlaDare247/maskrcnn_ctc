import os
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator

from ..datasets.mapper import build_mapper

class CTCTrainer(DefaultTrainer):
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
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
