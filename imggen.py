import os
import shutil
import cv2

INPUT_DIR = "frames"
SHARP_DIR = "frames_sharp"
RESIZED_DIR = "frames_resized"
THRESHOLD = 100.0  # Blur detection threshold

os.makedirs(SHARP_DIR, exist_ok=True)
os.makedirs(RESIZED_DIR, exist_ok=True)

# Step 1: Filter out blurry frames into 'frames_sharp'
print("Filtering blurry frames into frames_sharp...")
for fname in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() >= THRESHOLD:
        shutil.copy(img_path, os.path.join(SHARP_DIR, fname))

# Step 2: Resize non-blurry frames to 640x640 into 'frames_resized'
print("Resizing sharp frames to 640x640 into frames_resized...")
for fname in os.listdir(SHARP_DIR):
    in_file = os.path.join(SHARP_DIR, fname)
    out_file = os.path.join(RESIZED_DIR, fname)
    img = cv2.imread(in_file)
    if img is not None:
        resized = cv2.resize(img, (640, 640))
        cv2.imwrite(out_file, resized)

