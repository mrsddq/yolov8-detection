from pathlib import Path

import yaml


REQUIRED_TRAIN_KEYS = {"model", "data", "epochs", "imgsz", "batch"}
REQUIRED_DATA_KEYS = {"path", "train", "val", "names"}


def load_yaml(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return data


def validate_keys(config, required_keys, label):
    missing = sorted(required_keys - set(config))
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")


def validate_yolo_data_config(path):
    config = load_yaml(path)
    validate_keys(config, REQUIRED_DATA_KEYS, "YOLO data config")
    names = config["names"]
    if not isinstance(names, dict) or not names:
        raise ValueError("YOLO data config must define class names as a non-empty mapping")
    if "nc" in config and int(config["nc"]) != len(names):
        raise ValueError("YOLO data config nc must match the number of class names")
    expected_indices = set(range(len(names)))
    actual_indices = {int(index) for index in names}
    if actual_indices != expected_indices:
        raise ValueError("YOLO class names must use contiguous zero-based indices")
    return config


def validate_train_config(path):
    config_path = Path(path)
    config = load_yaml(path)
    validate_keys(config, REQUIRED_TRAIN_KEYS, "YOLO train config")
    for key in ("epochs", "imgsz", "batch"):
        if int(config[key]) <= 0:
            raise ValueError(f"YOLO train config {key} must be positive")
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = config_path.parent.parent / data_path
    validate_yolo_data_config(data_path)
    return config
