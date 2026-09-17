"""Train YOLOv8 on a custom dataset."""
import argparse


from scripts.utils import validate_train_config


def main(cfg_path):
    cfg = validate_train_config(cfg_path)
    from scripts.validate_dataset import validate_dataset
    validate_dataset(cfg["data"])
    from ultralytics import YOLO
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yolov8.yaml")
    args = p.parse_args()
    main(args.config)
