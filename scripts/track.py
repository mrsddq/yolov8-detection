from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main(weights: str, source: str, project: str, name: str) -> None:
    if not str(source).isdigit() and not Path(source).exists():
        raise FileNotFoundError(f"Tracking source not found: {source}")
    model = YOLO(weights)
    model.track(source=source, tracker="bytetrack.yaml", persist=True, save=True, project=project, name=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 + ByteTrack tracking.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project", default="runs/track")
    parser.add_argument("--name", default="bytetrack")
    args = parser.parse_args()
    main(args.weights, args.source, args.project, args.name)
