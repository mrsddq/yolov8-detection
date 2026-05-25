# Deployment Notes

## Demo Target

Best first demo: a Gradio or Streamlit app where a user uploads an image and receives bounding boxes plus per-class confidence.

## Production Path

1. Export trained weights to ONNX.
2. Validate inference parity against PyTorch.
3. Package a FastAPI endpoint.
4. Add confidence thresholds and max image size checks.
5. Log predictions without storing private images.

## Artifact Policy

Commit small screenshots and charts under `assets/`. Do not commit large datasets or model weights.
