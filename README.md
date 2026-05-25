# YOLOv8 Object Detection

[![CI](https://github.com/mrsddq/yolov8-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsddq/yolov8-detection/actions/workflows/ci.yml)

Portfolio-ready YOLOv8 object detection pipeline for a custom street-scene dataset.

The repository is structured for training, validation, and inference with Ultralytics YOLOv8. It does not ship private datasets or model weights; it provides reusable code, configs, and a result-reporting template.

## Highlights

- YOLO-format dataset configuration
- Training, evaluation, and inference scripts
- Config validation utilities
- Pytest checks for config integrity
- Results template for honest experiment tracking

## Classes

| ID | Name |
|---:|---|
| 0 | car |
| 1 | pedestrian |
| 2 | road_sign |

## Structure

```text
configs/
  data.yaml
  yolov8.yaml
docs/
  ABLATION_PLAN.md
  ARCHITECTURE_RATIONALE.md
  DEPLOYMENT_NOTES.md
  REPRODUCIBILITY.md
  RESULTS_TEMPLATE.md
scripts/
  train.py
  evaluate.py
  infer.py
  utils.py
tests/
  test_configs.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Layout

The scripts expect YOLO-format labels:

```text
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each label file should contain normalized `class_id cx cy width height` rows.

## Train

```bash
python -m scripts.train --config configs/yolov8.yaml
```

Equivalent Ultralytics CLI command:

```bash
yolo detect train data=configs/data.yaml model=yolov8s.pt epochs=100 imgsz=416 batch=16
```

## Evaluate

```bash
python -m scripts.evaluate --weights runs/detect/train/weights/best.pt --split val
```

For test-set evaluation:

```bash
python -m scripts.evaluate --weights runs/detect/train/weights/best.pt --split test
```

## Inference

```bash
python -m scripts.infer --weights runs/detect/train/weights/best.pt --source assets/sample.jpg
```

Predictions are saved under `runs/detect/infer/` by default.

## Testing

```bash
pytest
```

## Results

No public model weights or verified metrics are committed in this repository. After training, record real metrics in [docs/RESULTS_TEMPLATE.md](docs/RESULTS_TEMPLATE.md) and add sample prediction images under `assets/`.

Research support docs:

- [Reproducibility Plan](docs/REPRODUCIBILITY.md)
- [Architecture Rationale](docs/ARCHITECTURE_RATIONALE.md)
- [Ablation Plan](docs/ABLATION_PLAN.md)
- [Deployment Notes](docs/DEPLOYMENT_NOTES.md)

`outputs/metrics/smoke_test_results.csv` is a schema artifact only, not a benchmark.

Recommended artifacts:

- `assets/prediction-clean.png`
- `assets/prediction-crowded.png`
- `assets/confusion-matrix.png`
- `assets/pr-curve.png`
- `assets/results-plot.png`
- `assets/failure-case.png`

## Limitations

- Dataset is not included.
- Model weights are not included.
- Reported metrics should be treated as experiment-specific until reproduced.
- This is frame-level detection only; it does not include tracking or temporal smoothing.
