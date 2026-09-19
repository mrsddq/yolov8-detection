"""Fail-fast YOLO image/label validation before any model or weight download."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image
from scripts.utils import validate_yolo_data_config


def validate_label(path, classes):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: nonnumeric label") from exc
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError(f"{path}:{number}: expected five finite values")
        cls, x, y, width, height = values
        if cls != int(cls) or not 0 <= cls < classes:
            raise ValueError(f"{path}:{number}: invalid class ID")
        if width <= 0 or height <= 0 or min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
            raise ValueError(f"{path}:{number}: box is outside normalized image bounds")
        rows.append(values)
    return rows


def validate_dataset(config_path, splits=("train", "val")):
    config_path = Path(config_path).resolve()
    config = validate_yolo_data_config(config_path)
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = config_path.parent.parent / root
    seen, result = {}, {}
    for split in splits:
        if split not in config:
            raise ValueError(f"Missing split {split}")
        images = Path(config[split])
        images = images if images.is_absolute() else root / images
        paths = sorted(p for p in images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not paths:
            raise ValueError(f"No images in {split}: {images}")
        boxes, backgrounds = 0, 0
        label_paths = set()
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Unreadable image: {path}") from exc
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError(f"Image leakage between {seen[digest][0]} and {split}: {path}")
            seen[digest] = (split, str(path))
            relative = path.relative_to(root / "images")
            label = (root / "labels" / relative).with_suffix(".txt")
            if label in label_paths:
                raise ValueError(f"Multiple image files map to one label: {label}")
            label_paths.add(label)
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}; use an empty file for verified backgrounds")
            rows = validate_label(label, len(config["names"]))
            boxes += len(rows)
            backgrounds += not rows
        result[split] = {"images": len(paths), "boxes": boxes, "backgrounds": backgrounds}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    a = p.parse_args()
    print(json.dumps(validate_dataset(a.data, a.splits), indent=2))
