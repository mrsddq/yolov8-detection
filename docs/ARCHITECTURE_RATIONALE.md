# Architecture Rationale

## Why YOLOv8

YOLOv8 is a strong baseline for real-time object detection because it balances speed, accuracy, and practical deployment support.

## Class Scope

This repo keeps a narrow three-class street-scene taxonomy:

- car
- pedestrian
- road_sign

The narrow scope makes labeling, error analysis, and class-balance reporting manageable for a portfolio-sized experiment.

## Extension Path

- add ByteTrack/BoT-SORT for video tracking
- add DVC for dataset versioning
- add `scripts/infer_video.py` for video files and webcams
- add model export to ONNX for deployment
