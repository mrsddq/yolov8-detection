from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from PIL import Image


VISDRONE_TO_YOLO = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
}


def convert_split(source_root: Path, output_root: Path, split: str) -> None:
    images_dir = source_root / f"VisDrone2019-DET-{split}" / "images"
    ann_dir = source_root / f"VisDrone2019-DET-{split}" / "annotations"
    if not images_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing VisDrone images or annotations for {split}")
    if not list(images_dir.glob("*.jpg")):
        raise ValueError(f"No JPG images found for {split}")
    out_images = output_root / "images" / split
    out_labels = output_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images_dir.glob("*.jpg")):
        with Image.open(image_path) as image:
            width, height = image.size
        label_lines = []
        annotation_path = ann_dir / f"{image_path.stem}.txt"
        if annotation_path.exists():
            for line in annotation_path.read_text(encoding="utf-8").splitlines():
                left, top, box_w, box_h, score, cls, *_ = [float(part) for part in line.split(",")]
                cls = int(cls)
                if int(score) == 0 or cls not in VISDRONE_TO_YOLO:
                    continue
                if not all(math.isfinite(x) for x in (left, top, box_w, box_h)) or box_w <= 0 or box_h <= 0:
                    raise ValueError(f"Invalid box in {annotation_path}")
                right, bottom = min(width, left + box_w), min(height, top + box_h)
                left, top = max(0.0, left), max(0.0, top)
                box_w, box_h = right - left, bottom - top
                if box_w <= 0 or box_h <= 0:
                    continue
                x_center = (left + box_w / 2.0) / width
                y_center = (top + box_h / 2.0) / height
                label_lines.append(
                    f"{VISDRONE_TO_YOLO[cls]} {x_center:.6f} {y_center:.6f} {box_w / width:.6f} {box_h / height:.6f}"
                )
        else:
            raise FileNotFoundError(f"Missing annotation: {annotation_path}")
        target_image = out_images / image_path.name
        shutil.copyfile(image_path, target_image)
        (out_labels / f"{image_path.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    for split in args.splits:
        convert_split(args.source_root, args.output_root, split)


if __name__ == "__main__":
    main()
