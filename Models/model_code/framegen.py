import os
import cv2
import numpy as np

# Configuration
video_path = "campus_video.mp4"  # Replace with your actual video path
frame_interval = 10              # Take every 10th frame

output_640 = "dataset_640x640"
output_1280 = "dataset_1280x1280"

os.makedirs(output_640, exist_ok=True)
os.makedirs(output_1280, exist_ok=True)

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

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

frame_count = 0
saved_count = 0

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
        
        # Save both versions
        filename = f"frame_{saved_count:05d}.jpg"
        cv2.imwrite(os.path.join(output_640, filename), frame_640)
        cv2.imwrite(os.path.join(output_1280, filename), frame_1280)
        
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Done! Extracted {saved_count} frames into '{output_640}' and '{output_1280}'.")