from __future__ import annotations

import argparse
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
                x_center = (left + box_w / 2.0) / width
                y_center = (top + box_h / 2.0) / height
                label_lines.append(
                    f"{VISDRONE_TO_YOLO[cls]} {x_center:.6f} {y_center:.6f} {box_w / width:.6f} {box_h / height:.6f}"
                )
        target_image = out_images / image_path.name
        if not target_image.exists():
            target_image.write_bytes(image_path.read_bytes())
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
