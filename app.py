from __future__ import annotations

import gradio as gr


def status() -> str:
    return "YOLOv8 VisDrone demo scaffold is ready. Add trained weights before deploying."


demo = gr.Interface(fn=status, inputs=None, outputs="text", title="VisDrone YOLOv8 Detector")


if __name__ == "__main__":
    demo.launch()
