"""Train YOLOv8 on a custom dataset."""
import argparse

from ultralytics import YOLO

from scripts.utils import validate_train_config


def main(cfg_path):
    cfg = validate_train_config(cfg_path)
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yolov8.yaml")
    args = p.parse_args()
    main(args.config)
