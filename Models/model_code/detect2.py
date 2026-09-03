import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the models
det_model = YOLO("yolo26n.pt")
depth_model = YOLO("yolo26s-depth.pt")

# 2. Setup video capture and writer
input_video_path = "darkspot_f.MOV"
output_video_path = "output_processed_dark.mp4"

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

out_width = 1920
out_height = 540
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps / 3, (out_width, out_height))

frame_count = 0
skip_frames = 3

# --- Distance thresholds in meters — tune these after watching your test output ---
NEAR_THRESHOLD = 1.5
MID_THRESHOLD = 4.0


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


def pick_priority_detection(detections):
    """Out of everything detected this frame, pick the single most urgent one.
    Priority = closest distance, full stop. Center-near beats left-far, etc."""
    valid = [d for d in detections if d["distance_m"] is not None]
    if not valid:
        return None
    return min(valid, key=lambda d: d["distance_m"])


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


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    frame_width = frame.shape[1]

    # 3. Detection
    det_results = det_model(frame, imgsz=640, device="cpu", verbose=False)
    det_frame = det_results[0].plot()

    # 4. Depth
    depth_results = depth_model(frame, imgsz=768, device="cpu", verbose=False)
    depth_map = depth_results[0].depth.data.cpu().numpy()

    # 5. Structure detections with zone + distance
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

    # 6. Priority + guidance message (this is the "actionable insight" step)
    top_detection = pick_priority_detection(structured_detections)
    message = build_guidance_message(top_detection)
    print(f"Frame {frame_count}: {message}  |  all detections: {structured_detections}")

    # 7. Depth visualization (unchanged)
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = np.uint8(depth_normalized)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    det_frame_resized = cv2.resize(det_frame, (960, 540))
    depth_colored_resized = cv2.resize(depth_colored, (960, 540))
    combined_frame = np.hstack((det_frame_resized, depth_colored_resized))

    # 8. Burn the guidance message onto the frame so it's visible when you watch it back
    cv2.putText(
        combined_frame, message, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA
    )

    out.write(combined_frame)

cap.release()
out.release()
print("Processing complete. Video saved to:", output_video_path)