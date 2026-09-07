# Ishara Campus Video Datasets & Preprocessing Pipeline

## Overview

Assistive navigation research datasets historically suffer from an indoor-realism gap: existing benchmarks are captured in static, staged environments that do not reflect dynamic, crowded campus corridors.

The **Ishara Video Dataset** captures realistic university environments across varying lighting conditions, foot-traffic densities, staircases, doorways, and outdoor-indoor transitions.

---

## Directory Organization

```
datasets/
├── raw_videos/                      # Raw recorded video footage (.gitignored)
│   ├── .gitkeep                     # Preserves directory in Git
│   ├── batch_videos/                # Benchmark series V01.MOV - V28.MOV
│   ├── afloor.mp4                   # Floor transition test video
│   ├── campustour.mp4               # Full indoor building walkthrough
│   ├── darkspot.MOV                 # Low-light obstacle test
│   └── IMG_5081.MOV                 # High-resolution benchmark recording
│
├── processed/                       # Extracted frames & YOLO annotations (.gitignored)
│   ├── .gitkeep
│   ├── raw/                         # Raw extracted frame slices
│   └── processed/                   # Letterboxed and normalized training sets
│
├── pipeline/                        # Preprocessing & Data Wrangling Scripts
│   ├── framegen.py                  # Extracts letterboxed frames from video at fixed intervals
│   ├── frameimprove.py              # Frame enhancement & noise filtering
│   ├── data_merger.py               # Merges multiple YOLO-format annotation sets with label union
│   └── remap.py                     # Remaps class IDs across different dataset taxonomies
│
└── README.md                        # This document
```

---

## Video Collection Protocol

When capturing supplementary video footage for dataset expansion:

1. **Mounting / Camera Position**: Chest or head-mounted camera at eye/chest level (approx. 1.2m - 1.5m from floor), simulating a cane user or wearable assistive rig.
2. **Resolution & Framerate**: 1080p or 4K at 30 FPS / 60 FPS.
3. **Pacing**: Natural human walking pace (~1.0 - 1.4 m/s).
4. **Scenarios**:
   - Class change rushes (dense pedestrian dynamics)
   - Hallways with propped doors and maintenance equipment
   - Stairwell ascents and descents
   - Transition thresholds (glare from glass entrances, shadows under staircases)

---

## Preprocessing Pipeline

### 1. Extract Letterboxed Frames (`framegen.py`)
Extracts frames at regular stride intervals (e.g. every 10th frame) and applies letterbox resizing to preserve native aspect ratio:

```bash
python datasets/pipeline/framegen.py
```

Generates datasets formatted at `640x640` and `1280x1280` inside `datasets/processed/`.

### 2. Dataset Merging & Label Remapping (`data_merger.py`)
When combining university-specific annotations with open-source indoor datasets:

```bash
python datasets/pipeline/data_merger.py \
    --dataset-a datasets/processed/set_a \
    --dataset-b datasets/processed/set_b \
    --output datasets/processed/unified_dataset
```

---

## Gitignore Policy & Dataset Storage

> [!NOTE]
> All raw video recordings (`.mp4`, `.mov`, `.MOV`, `.avi`, `.zip`), extracted frame caches, and benchmark video sets are strictly excluded from Git tracking via `.gitignore`.
>
> The raw datasets are mirrored on cloud storage (Google Drive / OneDrive / Academic Data Repository). To download the raw archive `batch_videos.zip`, please refer to project releases or contact the repository maintainers.
