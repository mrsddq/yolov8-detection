# YOLOv8 detection engineering runbook

This repository wraps Ultralytics detection with dataset validation and VisDrone
conversion. See the [README setup](../README.md#setup),
[dataset gates](../README.md#dataset-gates-before-training), and
[training command](../README.md#train). Full detector execution uses
`requirements.txt`; the lightweight test dependencies do not install Ultralytics.

## Local verification

Run from the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Dependency installation needs package-network access. Once installed, the test
suite runs on CPU with generated fixtures and does not download model weights or
datasets. On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell
and run `python -m pytest -q`.

Tests use generated images/labels and a mocked trainer to check class IDs, box
bounds, missing annotations, exact duplicate files across splits, VisDrone box
clipping, and consistent dataset paths from another working directory. They do
not train a detector or measure mAP.

## Data and artifact contract

```bash
python -m scripts.validate_dataset --data configs/data.yaml --splits train val
python -m scripts.train --config configs/yolov8.yaml
```

- Use matching `images/<split>` and `labels/<split>` trees. Each image needs a label
  file; an empty file explicitly marks a verified background image.
- The training wrapper runs validation before loading weights. It writes an
  absolute, content-addressed dataset YAML under `<project>/validated-data/` and
  passes that exact configuration to Ultralytics. Retain it with run artifacts.
- Direct Ultralytics calls bypass this wrapper. Do not assume its global dataset
  root matches the repository-relative paths checked by this validator.
- Keep original images, downloaded/trained weights and `runs/` outside version
  control. Record dataset license, scene/video split, weights source and exact
  command when reporting real metrics.
- Hash checks detect byte-identical images, not near-duplicate video frames. Use
  scene/video-level splits and inspect suspicious overlap separately.

A passing software suite does not establish detection quality on a real dataset.
