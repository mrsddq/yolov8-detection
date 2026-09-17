"""Validate and train YOLO using the exact dataset paths checked by the gate."""
import argparse
import hashlib
from pathlib import Path
import yaml
from scripts.utils import validate_train_config, validate_yolo_data_config
from scripts.validate_dataset import validate_dataset


def main(cfg_path):
    cfg_path = Path(cfg_path).resolve()
    cfg = validate_train_config(cfg_path)
    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = cfg_path.parent.parent / data_path
    validate_dataset(data_path)
    data_cfg = validate_yolo_data_config(data_path)
    root = Path(data_cfg.get("path", "."))
    if not root.is_absolute():
        root = data_path.resolve().parent.parent / root
    data_cfg["path"] = str(root.resolve())
    for split in ("train", "val", "test"):
        if split in data_cfg:
            value = Path(data_cfg[split])
            data_cfg[split] = str((value if value.is_absolute() else root / value).resolve())
    serialized = yaml.safe_dump(data_cfg, sort_keys=True)
    digest = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    destination = Path(cfg.get("project", "runs/detect")) / "validated-data" / f"dataset-{digest}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    cfg["data"] = str(destination.resolve())
    from ultralytics import YOLO
    model = YOLO(cfg.pop("model"))
    return model.train(**cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yolov8.yaml")
    main(p.parse_args().config)
