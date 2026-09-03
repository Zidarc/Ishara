import os
import cv2
import numpy as np
from pathlib import Path

# Dynamically find the root directory of the Ishara repository (where this script is saved)
BASE_DIR = Path(__file__).resolve().parent

# Cross-platform video path (Ishara/Models/videos/IMG_5081.MOV)
video_path = BASE_DIR / "Models" / "videos" / "IMG_5081.MOV"

# Output dataset directories created directly in the Ishara root folder
output_640 = BASE_DIR / "dataset_640x640"
output_1280 = BASE_DIR / "dataset_1280x1280"

# Create output folders inside the root directory
output_640.mkdir(parents=True, exist_ok=True)
output_1280.mkdir(parents=True, exist_ok=True)

frame_interval = 10  # Take every 10th frame

def letterbox_resize(image, target_size=(640, 640), color=(114, 114, 114)):
    """Resizes image with aspect-ratio preservation using letterbox padding."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image keeping aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded canvas
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    # Center the resized image on the canvas
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    
    return canvas

# Pass string representation of Path object to OpenCV
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Error: Could not open video file at: {video_path}")
    print("Note: On Linux, file names and extensions are case-sensitive (.MOV vs .mov).")
    exit()

frame_count = 0
saved_count = 0

print(f"Processing video: {video_path}")
print("Extracting frames...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Generate padded 640x640 frame
        frame_640 = letterbox_resize(frame, target_size=(640, 640))
        
        # Generate padded 1280x1280 frame
        frame_1280 = letterbox_resize(frame, target_size=(1280, 1280))
        
        # Save both versions into root dataset directories
        filename = f"frame_{saved_count:05d}.jpg"
        cv2.imwrite(str(output_640 / filename), frame_640)
        cv2.imwrite(str(output_1280 / filename), frame_1280)
        
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Done! Extracted {saved_count} frames into:\n - {output_640}\n - {output_1280}")