from pathlib import Path
import pytest
from PIL import Image
from scripts.validate_dataset import validate_dataset, validate_label
from scripts.prepare_visdrone import convert_split


def make_dataset(tmp_path):
    for split, color in [("train", "red"), ("val", "blue")]:
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (10, 10), color).save(tmp_path / "images" / split / "a.png")
        (tmp_path / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.4 0.4")
    cfg = tmp_path / "data.yaml"
    cfg.write_text(f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnames: {{0: person}}\n")
    return cfg


def test_dataset_valid_and_detects_cross_split_duplicates(tmp_path):
    cfg = make_dataset(tmp_path)
    assert validate_dataset(cfg)["train"]["boxes"] == 1
    (tmp_path / "images/val/a.png").write_bytes((tmp_path / "images/train/a.png").read_bytes())
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset(cfg)


@pytest.mark.parametrize("row", ["0 nan .5 .2 .2", "0 .1 .5 .5 .5", "1 .5 .5 .2 .2", "0 .5 .5 0 .2", "0 .5 .5 .2"])
def test_bad_labels_fail(tmp_path, row):
    p = tmp_path / "label.txt"; p.write_text(row)
    with pytest.raises(ValueError):
        validate_label(p, 1)


def test_missing_labels_fail_instead_of_becoming_background(tmp_path):
    cfg = make_dataset(tmp_path)
    (tmp_path / "labels/train/a.txt").unlink()
    with pytest.raises(FileNotFoundError):
        validate_dataset(cfg)


def test_visdrone_clips_boundary_and_requires_annotations(tmp_path):
    source = tmp_path / "VisDrone2019-DET-train"
    (source / "images").mkdir(parents=True)
    (source / "annotations").mkdir()
    Image.new("RGB", (100, 100)).save(source / "images/a.jpg")
    label = source / "annotations/a.txt"
    label.write_text("-10,0,30,20,1,1,0,0\n")
    out = tmp_path / "converted"
    convert_split(tmp_path, out, "train")
    assert validate_label(out / "labels/train/a.txt", 10)[0] == [0, 0.1, 0.1, 0.2, 0.2]
    label.unlink()
    with pytest.raises(FileNotFoundError):
        convert_split(tmp_path, out, "train")
