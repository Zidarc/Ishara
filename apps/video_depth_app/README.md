# Ishara Assistive Navigation & Depth Studio (Healthcare Track)

## Overview

This application is an interactive **Streamlit** dashboard developed to visualize, validate, and evaluate the real-time perceptual pipeline of **Project Ishara** within the **Healthcare & Assistive Technology** track.

It performs:
1. **Model Selection**: Switch seamlessly between the fine-tuned campus navigation model (`models/trained/checkpoints/best (1).pt`) and open-source base models (`yolo26n`), with automatic OpenVINO/PyTorch acceleration.
2. **Healthcare Privacy Shield**: Built-in automated pedestrian and facial anonymization (Gaussian blur) preventing biometric identification in public university corridors.
3. **Side-by-Side Video Processing**: Displays original camera footage alongside the monocular depth map and object detection bounding boxes.
4. **Intelligent Resizing**: Automatically normalizes 4K and 1080p uploads to standard processing dimensions (e.g. max 640px) while preserving native aspect ratio, preventing CPU inference freezes.
5. **Live Perceptual Stream**: Frame-by-frame live visual updates and spatial telemetry during analysis.
6. **In-App Browser Playback & Download**: Encodes using universally compatible H.264 MP4 format for direct HTML5 browser playback and instant export.

---

## 🩺 Healthcare & Assistive Aid Advisory (Non-Reliance Notice)

> [!WARNING]
> **Auxiliary Assistive Aid Only — Not for Sole Reliance**
>
> Project Ishara is an auxiliary computer-vision perceptual aid engineered solely to augment spatial awareness. **It is NOT a certified medical diagnostic device or a primary orientation and mobility substitute.**
>
> Blind and low-vision users must **NEVER** place sole reliance on this software for life safety or solitary navigation. It is designed to complement, and must always be used in tandem with, primary mobility aids including the **white cane, guide dog, and certified Orientation & Mobility (O&M) training**.

---

## 🔒 Healthcare Privacy & Data Ethics

In institutional and healthcare environments, user and bystander privacy is paramount:
- **100% On-Device Processing**: All inference executes locally in system memory. No video feeds, biometric signatures, or telemetry are ever transmitted to external servers or cloud services.
- **Pedestrian Anonymization Shield**: Dynamically blurs facial and body silhouettes of detected individuals, ensuring compliance with institutional privacy standards and IRB guidelines.
- **Zero Data Retention**: Uploaded videos and temporary processing buffers are automatically sanitized and purged from the filesystem immediately after session use.

---

## Prerequisites & Installation

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

# Or from the repository root
streamlit run apps/video_depth_app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Application Structure

```
apps/video_depth_app/
├── app.py                           # Streamlit UI, live frame stream, and dual-model inference
├── requirements.txt                 # Python dependencies (streamlit, ultralytics, opencv, av)
├── packages.txt                     # Linux system dependencies (ffmpeg, libgl)
└── README.md                        # This guide
```

---

## Hardware Acceleration

The application dynamically detects OpenVINO compiled models from `models/base/weights/` if available, offering up to **3x-5x speedup** on standard x86/Intel CPUs compared to standard PyTorch FP32 inference. If OpenVINO binaries are not present, it automatically falls back to raw PyTorch `.pt` models.
