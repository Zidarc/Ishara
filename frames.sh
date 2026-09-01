#!/bin/bash
VIDEO="/home/Ali/D/Git Projects/Ishara/Videos/AB2Afloor_1.mp4"

mkdir -p frames frames_sharp frames_resized

ffmpeg -i "$VIDEO" -vf "fps=2" frames/frame_%04d.jpg

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