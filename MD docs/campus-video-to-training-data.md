# Campus Video → Training Data Pipeline

Pipeline for turning campus video footage into labeled image data for training/fine-tuning models (object detection, depth estimation, segmentation).

---

## 1. Extract Frames from Video

### Option A — Fixed frame rate (simplest)
```bash
ffmpeg -i campus_video.mp4 -vf "fps=2" frames/frame_%04d.jpg
```

### Option B — Scene-change based (skips near-duplicate static frames)
```bash
ffmpeg -i campus_video.mp4 -vf "select='gt(scene,0.02)'" -vsync vfr frames/frame_%04d.jpg
```

### Option C — Python (more control, e.g. every Nth frame)
```python
import cv2, os
os.makedirs("frames", exist_ok=True)
cap = cv2.VideoCapture("campus_video.mp4")
i, saved = 0, 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if i % 15 == 0:  # adjust interval
        cv2.imwrite(f"frames/frame_{saved:04d}.jpg", frame)
        saved += 1
    i += 1
cap.release()
```

**Install:** `sudo dnf install ffmpeg` (Fedora) or `pip install opencv-python --break-system-packages` for the Python route.

---

## 2. Remove Blurry Frames

### Option A — Laplacian variance (fast, general purpose)
```python
import cv2, os, shutil

INPUT_DIR, OUTPUT_DIR, THRESHOLD = "frames", "frames_sharp", 100.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() >= THRESHOLD:
        shutil.copy(os.path.join(INPUT_DIR, fname), os.path.join(OUTPUT_DIR, fname))
```
Tune `THRESHOLD` (80–150 typical range) by checking how many frames get kept/dropped and eyeballing a sample.

### Option B — Sobel gradient method (alternative metric, good on textured/outdoor scenes)
```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sharpness = sobelx.var() + sobely.var()
```
Swap this in for the Laplacian check if Option A over/under-filters.

### Option C — No-code / GUI
Manually review with a fast image viewer (`nomacs` or `geeqie` on Fedora) if the dataset is small enough.

---

## 3. Resize / Standardize (optional, recommended before labeling)
```bash
mkdir frames_resized
for f in frames_sharp/*.jpg; do
  ffmpeg -i "$f" -vf "scale=640:640" "frames_resized/$(basename "$f")"
done
```

---

## 4. Labeling: Images Alone vs. Labeled Data

Whether you need labels depends on the model type:

| Task | Need labels? | Notes |
|---|---|---|
| Object detection / segmentation ("this is a door") | **Yes** | Raw images alone are useless; model needs boxes/masks |
| Depth estimation (transfer learning from MiDaS, Depth Anything) | Usually no | Pretrained models work without manual labels |
| Depth fine-tuning against ground truth | Yes (depth data) | Requires LiDAR/stereo capture alongside video, different pipeline |
| General vision-language fine-tuning on campus scenes | Sometimes | Captions may suffice instead of per-object boxes |

For an object detection + depth pipeline (AR obstacle-guidance use case), you'll need labeled boxes for detection.

### Labeling Tool Alternatives

| Tool | Best for | Notes |
|---|---|---|
| **Roboflow** | Fastest start, has auto-labeling | Free tier, exports YOLO/COCO directly |
| **CVAT** | Full control, self-hosted, teams | Steeper setup, no vendor lock-in |
| **LabelImg** | Simple, offline, lightweight | Boxes only, no segmentation |
| **Label Studio** | Flexible, multiple task types | Good if adding classification/captioning later |

**Recommended path:** Roboflow → export in YOLOv8 format (fits lightweight CPU-friendly detection models well).

---

## 5. One-Shot Combined Script (extract → blur filter → resize)

```bash
#!/bin/bash
mkdir -p frames frames_sharp frames_resized

ffmpeg -i campus_video.mp4 -vf "fps=2" frames/frame_%04d.jpg

python3 - <<'EOF'
import cv2, os, shutil
INPUT_DIR, OUTPUT_DIR, THRESHOLD = "frames", "frames_sharp", 100.0
for fname in os.listdir(INPUT_DIR):
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() >= THRESHOLD:
        shutil.copy(os.path.join(INPUT_DIR, fname), os.path.join(OUTPUT_DIR, fname))
EOF

for f in frames_sharp/*.jpg; do
  ffmpeg -i "$f" -vf "scale=640:640" "frames_resized/$(basename "$f")"
done

echo "Done. Upload frames_resized/ to Roboflow or CVAT."
```
