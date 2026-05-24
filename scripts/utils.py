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
    return config


def validate_train_config(path):
    config = load_yaml(path)
    validate_keys(config, REQUIRED_TRAIN_KEYS, "YOLO train config")
    validate_yolo_data_config(config["data"])
    return config
