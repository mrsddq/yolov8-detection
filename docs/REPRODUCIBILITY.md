# Reproducibility Plan

## Environment

- Python: 3.10
- Dependencies: pinned in `requirements.txt`
- Training config: `configs/yolov8.yaml`
- Dataset contract: `configs/data.yaml`

## Dataset Contract

Do not commit private images or labels. For each real run, record:

- dataset name and version
- image count by split
- class balance for `car`, `pedestrian`, and `road_sign`
- label format validation result
- checksum or DVC hash for the dataset snapshot

## Run Order

1. Validate YAML configs with `pytest`.
2. Train with `python -m scripts.train --config configs/yolov8.yaml`.
3. Evaluate validation and test splits.
4. Export prediction images and confusion/PR curves.
5. Fill `docs/RESULTS_TEMPLATE.md`.

## Non-Benchmark Artifact

`outputs/metrics/smoke_test_results.csv` documents the expected metrics schema only. It is not mAP evidence.
