"""Evaluate a trained YOLOv8 model on the validation or test split."""
import argparse

from ultralytics import YOLO

from scripts.utils import validate_yolo_data_config


def main(weights, split, data, imgsz):
    validate_yolo_data_config(data)
    model = YOLO(weights)
    metrics = model.val(data=data, split=split, imgsz=imgsz)
    print(f"mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:{metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--imgsz", type=int, default=416)
    a = p.parse_args()
    main(a.weights, a.split, a.data, a.imgsz)
