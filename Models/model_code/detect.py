import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the models
# (Pro-tip: run `yolo export model=yolo26n.pt format=openvino` in terminal first, 
# then load the '_openvino_model' directory here for a massive CPU speed boost)
det_model = YOLO("yolo26n.pt") 
depth_model = YOLO("yolo26s-depth.pt")

# 2. Setup video capture and writer
input_video_path = "darkspot_f.MOV"
output_video_path = "output_processedk.mp4"

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Dynamically get original video dimensions
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output resolution: Side-by-side means doubling the original width, keeping height the same
out_width = orig_width * 2  
out_height = orig_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps / 3, (out_width, out_height))

frame_count = 0
skip_frames = 3 # Process 1 frame, skip the next 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Skip frames to speed up CPU processing
    if frame_count % skip_frames != 0:
        continue

    # 3. Run Detection Inference (imgsz=640 is standard for nano models)
    det_results = det_model(frame, imgsz=640, device="cpu", verbose=False)
    # Get the frame with bounding boxes drawn
    det_frame = det_results[0].plot()

    # 4. Run Depth Inference (YOLO26 depth is optimized at 768)
    depth_results = depth_model(frame, imgsz=768, device="cpu", verbose=False)
    
    # Extract depth map (absolute distance in meters)
    depth_map = depth_results[0].depth.data.cpu().numpy()
    
    # 5. Process Depth Map for Visualization
    # Normalize the depth map to 0-255 for rendering
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = np.uint8(depth_normalized)
    
    # Apply a colormap (INFERNO or MAGMA work best for depth visualization)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    
    # 6. Resize back to original frame dimensions (prevents stretching)
    det_frame_resized = cv2.resize(det_frame, (orig_width, orig_height))
    depth_colored_resized = cv2.resize(depth_colored, (orig_width, orig_height))
    
    # Concatenate images horizontally (side-by-side)
    combined_frame = np.hstack((det_frame_resized, depth_colored_resized))
    
    # Write to output video
    out.write(combined_frame)

cap.release()
out.release()
print("Processing complete. Video saved to:", output_video_path)