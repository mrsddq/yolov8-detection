"""Evaluate a trained YOLOv8 model on the val/test split.
Usage: python scripts/evaluate.py --weights runs/detect/train/weights/best.pt
"""
import argparse
from ultralytics import YOLO


def main(weights, split):
    model = YOLO(weights)
    metrics = model.val(data="configs/data.yaml", split=split, imgsz=416)
    print(f"mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:{metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val")
    a = p.parse_args()
    main(a.weights, a.split)
