"""Run inference on an image or directory.
Usage: python scripts/infer.py --weights best.pt --source assets/sample.jpg
"""
import argparse
from ultralytics import YOLO


def main(weights, source, conf):
    model = YOLO(weights)
    results = model.predict(source=source, conf=conf, save=True, project="runs/detect", name="infer")
    for r in results:
        print(r.summary())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()
    main(a.weights, a.source, a.conf)
