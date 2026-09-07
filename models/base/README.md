# Open-Source Base Models & Edge Benchmarks

## Overview

This module houses the pretrained, open-source computer vision models evaluated for the **Ishara** assistive system. The pipeline integrates two foundational perceptual components:
1. **Object Detection**: Fast edge detection models (YOLO series) trained on standard object classes (people, chairs, obstacles, doorways).
2. **Monocular Depth Estimation**: Real-time relative depth estimation (YOLO-depth) predicting pixel-wise distance to gauge obstacle proximity.

---

## Directory Organization

```
models/base/
├── weights/                         # Checkpoints & model formats (.gitignored)
│   ├── .gitkeep                     # Keeps folder structure in Git
│   ├── yolo26n.pt                   # Pretrained PyTorch detection weights
│   ├── yolo26s-depth.pt             # Pretrained PyTorch monocular depth weights
│   ├── yolo26n_openvino_model/      # OpenVINO compiled IR model (optimized for Intel/x86 CPU)
│   └── yolo26s-depth_openvino_model/# OpenVINO compiled depth model
├── scripts/
│   ├── detect.py                    # Side-by-side detection & depth video processing
│   ├── detect2.py                   # High-contrast obstacle zone overlay script
│   └── benchmark.py                 # Latency, FPS, and hardware utilization profiler
└── README.md                        # This document
```

---

## Model Specifications

| Model Name | Task | Framework | Parameters / Size | Recommended Runtime |
| :--- | :--- | :--- | :--- | :--- |
| `yolo26n` | Real-time Object Detection | PyTorch / OpenVINO | ~2.6M (~5.5 MB) | CPU (OpenVINO) / Mobile NPU |
| `yolo26s-depth` | Monocular Depth Estimation | PyTorch / OpenVINO | ~9.4M (~26.8 MB) | CPU (OpenVINO) / Mobile NPU |

---

## Quickstart & Usage

### 1. Running Dual Detection + Depth Inference
The `detect.py` script runs real-time object detection and monocular depth side-by-side on test campus videos, computing proximity zones (Near, Mid, Far):

```bash
python models/base/scripts/detect.py
```

### 2. Exporting to OpenVINO for CPU Acceleration
To compile PyTorch checkpoints into high-throughput OpenVINO IR representations:

```bash
# Export detection model
yolo export model=models/base/weights/yolo26n.pt format=openvino

# Export depth estimation model
yolo export model=models/base/weights/yolo26s-depth.pt format=openvino
```

### 3. Benchmarking Inference Latency
Measure inference FPS, per-stage latency (preprocess, inference, postprocess), and memory consumption across PyTorch vs. OpenVINO runtimes:

```bash
python models/base/scripts/benchmark.py --video datasets/raw_videos/IMG_5081.MOV
```

---

## Weights Management
Large model binaries (`.pt`, `.bin`, OpenVINO directories) are excluded from Git via `.gitignore` to maintain a lightweight repository. For production distribution, weights are released as GitHub Release Assets or hosted on Hugging Face Hub.
