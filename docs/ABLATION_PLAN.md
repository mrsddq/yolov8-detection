# Ablation Plan

| Experiment | Variable | Fixed Controls | Metric |
|---|---|---|---|
| baseline | `yolov8s.pt` | image size 416, same data split | mAP50-95 |
| image size | 416 vs 640 | model, split, epochs | mAP and latency |
| augmentation | mosaic on/off | model, split | mAP by class |
| model size | n/s/m | image size, split | mAP vs inference time |
| confidence | threshold sweep | weights, split | precision/recall curve |

Every ablation should save the command, config, seed, metrics CSV, and representative failure images.
