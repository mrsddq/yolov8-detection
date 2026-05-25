# Model Card: VisDrone YOLOv8 Detection

## Dataset
Target dataset: VisDrone2019-DET converted to YOLO format with `scripts/prepare_visdrone.py`.

## Model
Baseline: YOLOv8n. Main model: YOLOv8s with standard Ultralytics augmentation.

## Evaluation
Primary metrics: mAP50 and mAP50-95. Report precision and recall from Ultralytics validation output.

## Inference
Image, video, directory, webcam, and optional ByteTrack tracking are supported by `scripts/infer.py`.

## Limitations
Aerial small-object detection is difficult; this repository should report dataset size, class imbalance, and failure cases instead of claiming production accuracy.

## Ethical Considerations
Drone footage may contain people and vehicles. Use public datasets and avoid surveillance framing in demos.
