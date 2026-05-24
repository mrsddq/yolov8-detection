from scripts.utils import validate_train_config, validate_yolo_data_config


def test_data_config_is_valid():
    config = validate_yolo_data_config("configs/data.yaml")

    assert config["nc"] == 3
    assert set(config["names"].values()) == {"car", "pedestrian", "road_sign"}


def test_train_config_is_valid():
    config = validate_train_config("configs/yolov8.yaml")

    assert config["model"] == "yolov8s.pt"
    assert config["data"] == "configs/data.yaml"
