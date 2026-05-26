# Monet - An AI Detection Tool

A browser extension + local Python backend that detects AI-generated video content in real time. As you scroll through YouTube Shorts (or any video-enabled page), Monet analyzes each video and overlays a color-coded border: green for real, orange for suspicious, red for AI-generated.

The extension captures frames from videos playing in your browser and sends them over a local WebSocket to a Python backend. The backend runs them through a stacked ensemble of analyzers and returns a score and breakdown in milliseconds.

The pipeline covers texture, biometrics, motion, color, semantics, and YouTube metadata, alongside a Swin Transformer (ViT) model running via ONNX Runtime, with all outputs combined through a trained Random Forest classifier.

## Setup

Install dependencies:

```bash
pip install fastapi uvicorn opencv-python numpy Pillow scikit-learn mediapipe transformers "optimum[onnxruntime]" torch
```

Start the server:

```bash
python server.py
```

Load the extension:

- Firefox: `about:debugging` > "This Firefox" > "Load Temporary Add-on" > select `manifest.json`
- Chrome: `chrome://extensions` > enable "Developer mode" > "Load unpacked" > select the Chrome extension folder (not included in this repo)

Then open any YouTube video or Short and the analysis runs automatically.

## Retraining

Add URLs to the text files in `training_data/` and run:

```bash
python train_model.py
```
