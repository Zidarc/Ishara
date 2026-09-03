import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the models
# (Pro-tip: run `yolo export model=yolo26n.pt format=openvino` in terminal first, 
# then load the '_openvino_model' directory here for a massive CPU speed boost)
det_model = YOLO("yolo26n.pt") 
depth_model = YOLO("yolo26s-depth.pt")

# 2. Setup video capture and writer
input_video_path = "videos/IMG_5081.MOV"
output_video_path = "videos/IMG_5081_processed.mp4"

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

# --- Distance thresholds in meters — tune these after watching your test output ---
NEAR_THRESHOLD = 1.5
MID_THRESHOLD = 4.0

# --- Overlay text settings ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2.2       # bumped up further — was too small at 1.2
FONT_THICKNESS = 4
LINE_GAP = 20          # vertical gap between stacked lines
TOP_MARGIN = 60
RIGHT_MARGIN = 20
TOP_N = 3              # how many priority messages to show


def get_zone_and_distance(box, depth_map, frame_width):
    """Work out which horizontal zone a detection is in, and how far away it is."""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        h_zone = "left"
    elif center_x < 2 * frame_width / 3:
        h_zone = "center"
    else:
        h_zone = "right"

    depth_crop = depth_map[y1:y2, x1:x2]
    distance = float(np.median(depth_crop)) if depth_crop.size > 0 else None

    if distance is None:
        d_zone = "unknown"
    elif distance < NEAR_THRESHOLD:
        d_zone = "near"
    elif distance < MID_THRESHOLD:
        d_zone = "mid"
    else:
        d_zone = "far"

    return h_zone, d_zone, distance


def pick_priority_detections(detections, n=TOP_N):
    """Out of everything detected this frame, rank the most urgent ones.
    Priority = closest distance, full stop. Center-near beats left-far, etc.
    Returns up to n detections, nearest first."""
    valid = [d for d in detections if d["distance_m"] is not None]
    if not valid:
        return []
    return sorted(valid, key=lambda d: d["distance_m"])[:n]


def build_guidance_message(detection):
    """Turn one structured detection into a short, speakable instruction."""
    if detection is None:
        return "Path clear"

    cls = detection["class"]
    h_zone = detection["h_zone"]
    d_zone = detection["d_zone"]
    dist = detection["distance_m"]

    if d_zone == "far":
        # Don't bother alerting on far-away objects — not actionable yet
        return "Path clear"

    urgency = "Caution" if d_zone == "near" else "Notice"
    return f"{urgency}: {cls} {h_zone}, {dist:.1f}m"


def draw_right_aligned_lines(frame, lines, top_margin=TOP_MARGIN, right_margin=RIGHT_MARGIN):
    """Draw a stack of lines in the top-right corner, each right-aligned."""
    frame_width = frame.shape[1]
    y = top_margin
    for line in lines:
        (text_w, text_h), baseline = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)
        x = frame_width - right_margin - text_w
        cv2.putText(
            frame, line, (x, y),
            FONT, FONT_SCALE, (0, 255, 255), FONT_THICKNESS, cv2.LINE_AA
        )
        y += text_h + LINE_GAP


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Skip frames to speed up CPU processing
    if frame_count % skip_frames != 0:
        continue

    frame_width = frame.shape[1]

    # 3. Run Detection Inference (imgsz=640 is standard for nano models)
    det_results = det_model(frame, imgsz=640, device="cpu", verbose=False)
    # Get the frame with bounding boxes drawn
    det_frame = det_results[0].plot()

    # 4. Run Depth Inference (YOLO26 depth is optimized at 768)
    depth_results = depth_model(frame, imgsz=768, device="cpu", verbose=False)
    
    # Extract depth map (absolute distance in meters)
    depth_map = depth_results[0].depth.data.cpu().numpy()

    # 4b. Structure detections with zone + distance, then rank top-N priority messages
    structured_detections = []
    for box in det_results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = det_model.names[cls_id]
        confidence = float(box.conf[0])
        h_zone, d_zone, distance = get_zone_and_distance(box, depth_map, frame_width)

        structured_detections.append({
            "class": class_name,
            "confidence": confidence,
            "h_zone": h_zone,
            "d_zone": d_zone,
            "distance_m": distance
        })

    top_detections = pick_priority_detections(structured_detections)
    if top_detections:
        messages = [f"{i+1}) {build_guidance_message(d)}" for i, d in enumerate(top_detections)]
    else:
        messages = ["Path clear"]
    print(f"Frame {frame_count}: {messages}  |  all detections: {structured_detections}")
    
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

    # 6b. Burn the guidance messages onto the frame, top-right, stacked by priority
    draw_right_aligned_lines(combined_frame, messages)
    
    # Write to output video
    out.write(combined_frame)

cap.release()
out.release()
print("Processing complete. Video saved to:", output_video_path)