"""Run YOLOv8 inference on an image, video, webcam, or directory."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main(weights, source, conf, project, name, track=False):
    if not str(source).isdigit() and not Path(source).exists():
        raise FileNotFoundError(f"Inference source not found: {source}")

    model = YOLO(weights)
    if track:
        results = model.track(source=source, conf=conf, tracker="bytetrack.yaml", persist=True, save=True, project=project, name=name)
    else:
        results = model.predict(source=source, conf=conf, save=True, project=project, name=name)
    print(f"Saved predictions to {project}/{name}")
    for result in results:
        print(result.summary())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="infer")
    p.add_argument("--track", action="store_true")
    a = p.parse_args()
    main(a.weights, a.source, a.conf, a.project, a.name, a.track)
