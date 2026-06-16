from scripts.utils import validate_train_config, validate_yolo_data_config


def test_data_config_is_valid():
    config = validate_yolo_data_config("configs/data.yaml")

    assert config["nc"] == 3
    assert set(config["names"].values()) == {"car", "pedestrian", "road_sign"}


def test_train_config_is_valid():
    config = validate_train_config("configs/yolov8.yaml")

    assert config["model"] == "yolov8s.pt"
    assert config["data"] == "configs/data.yaml"


def test_data_config_rejects_nc_mismatch(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text(
        "path: data\ntrain: images/train\nval: images/val\nnc: 2\nnames:\n  0: car\n",
        encoding="utf-8",
    )

    try:
        validate_yolo_data_config(config)
    except ValueError as exc:
        assert "nc must match" in str(exc)
    else:
        raise AssertionError("Expected nc mismatch to fail")


def test_train_config_rejects_non_positive_epochs(tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text(
        "path: data\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: car\n",
        encoding="utf-8",
    )
    train = tmp_path / "train.yaml"
    train.write_text(
        f"model: yolov8n.pt\ndata: {data}\nepochs: 0\nimgsz: 416\nbatch: 4\n",
        encoding="utf-8",
    )

    try:
        validate_train_config(train)
    except ValueError as exc:
        assert "epochs must be positive" in str(exc)
    else:
        raise AssertionError("Expected non-positive epochs to fail")
