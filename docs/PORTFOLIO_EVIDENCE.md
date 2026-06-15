# Portfolio Evidence Plan

This project should be shown as an object detection pipeline with dataset preparation, training, evaluation, and inference. Do not claim mAP or tracking quality until the exact run is documented.

## Reproducible Demo

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_portfolio_contract.py"
python -m scripts.prepare_visdrone --input data/raw --output data/visdrone
python -m scripts.train --config configs/yolov8.yaml
python -m scripts.evaluate --weights outputs/best.pt --data configs/data.yaml
python -m scripts.infer --weights outputs/best.pt --source data/samples --output outputs/predictions
```

## Evidence To Capture

| Artifact | Portfolio Use |
|---|---|
| `assets/dataset-sample.png` | Shows object classes and annotation density. |
| `assets/detection-grid.png` | Shows model predictions on held-out images. |
| `outputs/metrics/yolov8_eval.json` | Records precision, recall, mAP50, and mAP50-95. |
| `docs/RESULTS.md` | Summarizes verified training and evaluation runs. |

## Demo Narrative

1. Explain the data conversion path and class mapping.
2. Show the YOLO config and augmentation choices.
3. Present mAP and qualitative detections from the same run.
4. Discuss failure cases: small objects, occlusion, motion blur, and class imbalance.

## Evidence Checklist Before Pinning

- [ ] Public dataset or small reproducible sample identified.
- [ ] Class mapping verified in `configs/data.yaml`.
- [ ] Detection grid added to `assets/`.
- [ ] Real metric table added to `docs/RESULTS.md`.
- [ ] CI badge green on the latest commit.
