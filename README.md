# YOLOv8 Object Detection

Multi-class object detection on a custom street-scene dataset using YOLOv8. Covers the full pipeline: dataset preparation, annotation, training, evaluation, and inference.

## Results

| Metric | Value |
|---|---|
| mAP@0.5 | 89% |
| Image Resolution | 416×416 |
| Classes | Cars, Pedestrians, Road Signs |
| Precision | _add per-class breakdown_ |
| Recall | _add per-class breakdown_ |

## Classes

| ID | Name |
|---|---|
| 0 | car |
| 1 | pedestrian |
| 2 | road_sign |

## Quickstart

```bash
git clone https://github.com/your-username/yolov8-detection
cd yolov8-detection
pip install -r requirements.txt
```

## Data

Custom annotated dataset. Annotations in YOLO format (one `.txt` per image, normalised `cx cy w h`).

```
data/
  images/
    train/    ← training images
    val/      ← validation images
    test/     ← test images
  labels/
    train/    ← YOLO .txt annotations
    val/
    test/
```

Dataset is private. For a public equivalent use [COCO](https://cocodataset.org/) or [Open Images](https://storage.googleapis.com/openimages/web/index.html) filtered to the three classes above.

## Training

```bash
python scripts/train.py --config configs/yolov8.yaml
```

Or directly with the Ultralytics CLI:

```bash
yolo detect train data=configs/data.yaml model=yolov8s.pt epochs=100 imgsz=416 batch=16
```

Key config in `configs/yolov8.yaml`:

```yaml
model: yolov8s.pt
epochs: 100
imgsz: 416
batch: 16
optimizer: SGD
lr0: 0.01
augment: true
```

## Evaluation

```bash
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt
```

Auto-generated outputs in `runs/detect/val/`:
- `confusion_matrix.png`
- `PR_curve.png`
- `results.png`

## Inference

```bash
python scripts/infer.py --weights runs/detect/train/weights/best.pt --source assets/sample.jpg
```

## Sample Outputs

| File | Contents |
|---|---|
| `assets/01_prediction_clean.png` | Clean detection: all 3 classes correctly boxed |
| `assets/02_prediction_crowded.png` | Dense scene with NMS |
| `assets/03_confusion_matrix.png` | Class-level confusion matrix |
| `assets/04_pr_curve.png` | Precision-Recall curve (mAP@0.5 labelled) |
| `assets/05_results_plot.png` | Loss + mAP across all training epochs |
| `assets/06_dataset_sample.png` | Training mosaic with GT boxes |
| `assets/07_failure_case.png` | Missed detection + annotated reason |

## Limitations

- mAP reported at 416×416; higher resolution improves small-object recall
- Dataset is single-domain (street scene); cross-domain generalisation not tested
- No tracking or temporal smoothing — per-frame detection only

## Environment

```
Python 3.10
ultralytics==8.0.196
torch==2.1.0
CUDA 11.8
```
