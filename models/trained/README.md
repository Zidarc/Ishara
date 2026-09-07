# Custom Trained Models — Campus Obstacle Navigation

## Overview

This directory contains fine-tuned and domain-adapted computer vision models specifically trained on campus corridor walkthroughs, dynamic obstacles, and university building environments for **Project Ishara**.

Unlike generic COCO-trained models, these checkpoints are specialized for:
- Low-lying corridor hazards (trash bins, wet floor signs, backpacks, loose cables)
- Dynamic pedestrian avoidance in crowded campus hallways
- Doorways, staircases, and transition thresholds

---

## Directory Organization

```
models/trained/
├── checkpoints/                     # Model weights & training artifacts (.gitignored)
│   ├── .gitkeep                     # Preserves directory in Git
│   └── best (1).pt                  # Fine-tuned checkpoint (or best.pt)
├── scripts/
│   ├── model.py                     # Batch inference script on validation video series
│   └── benchmarkv1.py               # Comprehensive multi-video batch evaluation suite
└── README.md                        # This document
```

---

## Model Details & Training Specifications

| Property | Value |
| :--- | :--- |
| **Base Architecture** | YOLO (Fine-tuned for campus navigation) |
| **Input Resolution** | 1280x1280 (High-res spatial resolution for distant hazards) / 640x640 (Mobile) |
| **Target Classes** | Obstacles, pedestrians, stairways, doors, navigation hazards |
| **Training Dataset** | Curated campus footage (`datasets/raw_videos/batch_videos/`) |
| **Evaluation Metrics** | Precision, Recall, mAP@0.5, mAP@0.5:0.95 |

---

## Running Inference

To run the custom model on the validation video sequence:

```bash
python models/trained/scripts/model.py
```

Outputs are automatically saved to `test_results/` with annotated bounding boxes, confidence scores, and hazard classifications.

---

## Batch Evaluation & Benchmarking

The `benchmarkv1.py` suite evaluates model accuracy, frame stability, and inference latency across 28+ benchmark campus clips (`V01.MOV` through `V28.MOV`):

```bash
# Run benchmark across all batch videos
python models/trained/scripts/benchmarkv1.py --batch-dir datasets/raw_videos/batch_videos/
```

Results are exported to CSV and JSON summaries inside `datasets/benchmark_outputs/`.

---

## Checkpoint Distribution Policy

Trained weights are managed in accordance with open-source machine learning best practices:
- Binary `.pt` files and training checkpoint directories are excluded from Git via `.gitignore`.
- Official checkpoints can be downloaded from our **GitHub Releases** page or **Hugging Face Model Hub**.
- Checkpoint files should be placed in `models/trained/checkpoints/` for automated discovery by inference scripts.
