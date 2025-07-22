"""Train YOLOv8 on custom dataset.
Usage: python scripts/train.py --config configs/yolov8.yaml
"""
import argparse, yaml
from ultralytics import YOLO


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yolov8.yaml")
    main(p.parse_args().config)
