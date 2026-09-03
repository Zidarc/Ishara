import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

NEAR_THRESHOLD = 1.5
MID_THRESHOLD = 4.0
SKIP_FRAMES = 8
DET_IMGSZ = 256
DEPTH_IMGSZ = 256




@st.cache_resource
def load_models():
    detection_model = YOLO("models/yolo26n_openvino_model")
    depth_model = YOLO("models/yolo26s-depth_openvino_model")
    return detection_model, depth_model



def get_zone_and_distance(box, depth_map, frame_width):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    # Prevent invalid crops at frame boundaries
    height, width = depth_map.shape[:2]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        horizontal_zone = "left"
    elif center_x < 2 * frame_width / 3:
        horizontal_zone = "center"
    else:
        horizontal_zone = "right"

    depth_crop = depth_map[y1:y2, x1:x2]
    distance = float(np.median(depth_crop)) if depth_crop.size else None

    if distance is None:
        distance_zone = "unknown"
    elif distance < NEAR_THRESHOLD:
        distance_zone = "near"
    elif distance < MID_THRESHOLD:
        distance_zone = "mid"
    else:
        distance_zone = "far"

    return horizontal_zone, distance_zone, distance


def build_guidance_message(detections):
    valid_detections = [
        detection
        for detection in detections
        if detection["distance_m"] is not None
    ]

    if not valid_detections:
        return "Path clear"

    closest = min(valid_detections, key=lambda item: item["distance_m"])

    if closest["d_zone"] == "far":
        return "Path clear"

    urgency = "Caution" if closest["d_zone"] == "near" else "Notice"
    return (
        f"{urgency}: {closest['class']} "
        f"{closest['h_zone']}, {closest['distance_m']:.1f}m"
    )

def process_video(input_path, output_path, det_model, depth_model, progress_bar):
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_fps = fps / SKIP_FRAMES
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width * 2, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create the output video.")

    frame_number = 0
    written_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1

        # Run inference only on this retained frame.
        det_results = det_model(
            frame, imgsz=DET_IMGSZ, device="cpu", verbose=False
        )
        depth_results = depth_model(
            frame, imgsz=DEPTH_IMGSZ, device="cpu", verbose=False
        )

        det_frame = det_results[0].plot()
        depth_map = depth_results[0].depth.data.cpu().numpy()
        depth_map = cv2.resize(depth_map, (width, height))

        detections = []
        for box in det_results[0].boxes:
            class_id = int(box.cls[0])
            horizontal_zone, distance_zone, distance = get_zone_and_distance(
                box, depth_map, width
            )

            detections.append(
                {
                    "class": det_model.names[class_id],
                    "h_zone": horizontal_zone,
                    "d_zone": distance_zone,
                    "distance_m": distance,
                }
            )

        message = build_guidance_message(detections)

        depth_normalized = cv2.normalize(
            depth_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        depth_colored = cv2.applyColorMap(
            depth_normalized, cv2.COLORMAP_INFERNO
        )

        det_frame = cv2.resize(det_frame, (width, height))
        depth_colored = cv2.resize(depth_colored, (width, height))
        combined = np.hstack((det_frame, depth_colored))

        cv2.putText(
            combined,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # `combined` exists now, so write it here.
        writer.write(combined)
        written_frames += 1

        # Skip the next frames without retrieving them into Python.
        reached_end = False
        for _ in range(SKIP_FRAMES - 1):
            if not cap.grab():
                reached_end = True
                break
            frame_number += 1

        if total_frames:
            progress_bar.progress(min(frame_number / total_frames, 1.0))

        if reached_end:
            break

    cap.release()
    writer.release()

    if written_frames == 0:
        raise RuntimeError("No frames were written to the output video.")


st.set_page_config(page_title="Video Depth Guidance", layout="wide")
st.title("Video Depth Guidance")
st.write("Upload a video to generate detection, depth visualization, and guidance text.")

uploaded_video = st.file_uploader(
    "Choose a video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video:
    st.video(uploaded_video)

    if st.button("Process video", type="primary"):
        detection_model, depth_model = load_models()
        progress_bar = st.progress(0)
        status = st.empty()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_path = temp_dir / f"input{Path(uploaded_video.name).suffix}"
            output_path = temp_dir / "processed_video.mp4"

            input_path.write_bytes(uploaded_video.getbuffer())

            try:
                status.info("Processing video on CPU...")
                process_video(
                    input_path,
                    output_path,
                    detection_model,
                    depth_model,
                    progress_bar,
                )

                output_bytes = output_path.read_bytes()
                status.success("Processing complete.")

                st.subheader("Processed video")
                st.video(output_bytes)

                st.download_button(
                    "Download processed video",
                    data=output_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )
            except Exception as error:
                status.error(f"Processing failed: {error}")
