# Video Depth & Obstacle Detection Web Demo

## Overview

This application is an interactive **Streamlit** dashboard developed to visualize and validate the real-time perceptual pipeline of **Project Ishara**.

It performs:
1. **Side-by-Side Video Processing**: Displays original camera footage alongside the monocular depth map and object detection bounding boxes.
2. **Dynamic Zone & Distance Calculation**: Partitions the field of view into three horizontal sectors (`Left`, `Center`, `Right`) and three depth tiers (`Near < 1.5m`, `Mid < 4.0m`, `Far >= 4.0m`).
3. **Real-time Navigation Guidance**: Generates human-readable navigational alerts (e.g., *"Person detected Center - Near. Caution!"*).

---

## Prerequisites

Install the required Python dependencies:

```bash
cd apps/video_depth_app
pip install -r requirements.txt
```

---

## Launching the Web Application

To start the local Streamlit web server:

```bash
# From apps/video_depth_app/
streamlit run app.py

# Or from repository root
streamlit run apps/video_depth_app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Application Structure

```
apps/video_depth_app/
├── app.py                           # Streamlit UI & dual-model inference loop
├── requirements.txt                 # Python dependencies (streamlit, opencv, ultralytics)
├── packages.txt                     # Linux system dependencies (ffmpeg, libgl)
└── README.md                        # This document
```

---

## Hardware Acceleration

By default, the application loads OpenVINO compiled models from `models/base/weights/` if available, offering up to **3x-5x speedup** on standard x86/Intel CPUs compared to standard PyTorch FP32 inference. If OpenVINO binaries are not present, it automatically falls back to raw PyTorch `.pt` models.
