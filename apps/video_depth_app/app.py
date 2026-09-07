"""
Ishara — Assistive Navigation & Depth Studio (Healthcare Track)
==============================================================
Auxiliary computer-vision spatial awareness tool for blind and low-vision students.
Combines real-time object detection, monocular depth estimation, and privacy anonymization.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Optional high-compatibility video encoder
try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

# -----------------------------------------------------------------------------
# Configuration & Path Resolution
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
BASE_WEIGHTS_DIR = REPO_ROOT / "models" / "base" / "weights"
TRAINED_WEIGHTS_DIR = REPO_ROOT / "models" / "trained" / "checkpoints"


# -----------------------------------------------------------------------------
# Streamlit Page Setup & Healthcare Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ishara — Assistive Navigation Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Modern Healthcare & Accessibility Styling */
    .healthcare-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 24px;
        color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .healthcare-header h1 {
        color: #e0f2fe;
        font-size: 2rem;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .healthcare-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .medical-disclaimer {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 16px 20px;
        border-radius: 8px;
        color: #991b1b;
        margin-bottom: 24px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .medical-disclaimer strong {
        color: #7f1d1d;
        font-size: 1rem;
    }
    .privacy-badge {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        display: inline-block;
        margin-bottom: 15px;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model Discovery & Loading
# -----------------------------------------------------------------------------
def get_available_detection_models():
    """Discover available detection models (custom trained vs base)."""
    models = {}

    # Check for fine-tuned campus model
    if TRAINED_WEIGHTS_DIR.exists():
        for item in TRAINED_WEIGHTS_DIR.iterdir():
            if item.name.lower().endswith(".pt") or (item.is_dir() and "best" in item.name.lower()):
                models[f"Custom Trained Campus Model ({item.name})"] = str(item)

    # Check for OpenVINO base model
    ov_det = BASE_WEIGHTS_DIR / "yolo26n_openvino_model"
    if ov_det.exists():
        models["Base YOLO26n (OpenVINO Accelerated)"] = str(ov_det)

    # Check for raw PyTorch base weights
    pt_det = BASE_WEIGHTS_DIR / "yolo26n.pt"
    if pt_det.exists():
        models["Base YOLO26n (PyTorch)"] = str(pt_det)

    if not models:
        models["Default YOLOv8n (Download on demand)"] = "yolo26n.pt"

    return models


def get_available_depth_models():
    """Discover available monocular depth models."""
    models = {}

    ov_depth = BASE_WEIGHTS_DIR / "yolo26s-depth_openvino_model"
    if ov_depth.exists():
        models["YOLO26s-Depth (OpenVINO Accelerated)"] = str(ov_depth)

    pt_depth = BASE_WEIGHTS_DIR / "yolo26s-depth.pt"
    if pt_depth.exists():
        models["YOLO26s-Depth (PyTorch)"] = str(pt_depth)

    if not models:
        models["Default YOLO26s-Depth"] = "yolo26s-depth.pt"

    return models


@st.cache_resource(show_spinner="Loading perceptual models into memory...")
def load_models(det_path: str, depth_path: str):
    """Load object detection and monocular depth models."""
    try:
        det_model = YOLO(det_path)
    except Exception as exc:
        st.warning(f"Failed to load detection model from {det_path}: {exc}. Falling back to default.")
        det_model = YOLO("yolo26n.pt")

    try:
        depth_model = YOLO(depth_path)
    except Exception as exc:
        st.warning(f"Failed to load depth model from {depth_path}: {exc}. Falling back to default.")
        depth_model = YOLO("yolo26s-depth.pt")

    return det_model, depth_model


# -----------------------------------------------------------------------------
# Spatial Reasoning & Privacy Utilities
# -----------------------------------------------------------------------------
def get_zone_and_distance(box, depth_map, frame_width, near_thresh=1.5, mid_thresh=4.0):
    """Compute lateral spatial zone (Left/Center/Right) and depth category."""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    height, width = depth_map.shape[:2]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        horizontal_zone = "Left"
    elif center_x < 2 * frame_width / 3:
        horizontal_zone = "Center"
    else:
        horizontal_zone = "Right"

    depth_crop = depth_map[y1:y2, x1:x2]
    distance = float(np.median(depth_crop)) if depth_crop.size else None

    if distance is None:
        distance_zone = "Unknown"
    elif distance < near_thresh:
        distance_zone = "Near"
    elif distance < mid_thresh:
        distance_zone = "Mid"
    else:
        distance_zone = "Far"

    return horizontal_zone, distance_zone, distance


def apply_privacy_blur(frame, box, blur_face_only=True):
    """
    Healthcare Privacy Shield: Anonymize human pedestrians in video feeds.
    Applies Gaussian blurring to the face/head region (or full box).
    """
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return frame

    if blur_face_only:
        # Blur the upper 40% of the bounding box (head/facial area)
        target_y2 = y1 + max(10, int((y2 - y1) * 0.40))
        roi = frame[y1:target_y2, x1:x2]
    else:
        roi = frame[y1:y2, x1:x2]

    if roi.size > 0:
        # Kernel size must be odd and proportional to ROI size
        kw = max(15, (roi.shape[1] // 3) * 2 + 1)
        kh = max(15, (roi.shape[0] // 3) * 2 + 1)
        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 25)
        if blur_face_only:
            frame[y1:target_y2, x1:x2] = blurred_roi
        else:
            frame[y1:y2, x1:x2] = blurred_roi

    return frame


def build_guidance_message(detections):
    """Synthesize priority spatial guidance alert for non-visual navigation."""
    valid = [d for d in detections if d["distance_m"] is not None]
    if not valid:
        return "Path Clear — Proceed"

    closest = min(valid, key=lambda item: item["distance_m"])
    if closest["d_zone"] == "Far":
        return "Path Clear — Distant objects detected"

    urgency = "⚠️ CAUTION" if closest["d_zone"] == "Near" else "ℹ️ NOTICE"
    return f"{urgency}: {closest['class'].upper()} {closest['h_zone']} ({closest['distance_m']:.1f}m)"


def resize_maintaining_aspect_ratio(image, max_dim=640):
    """Intelligently scale down large frames while preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        # Ensure dimensions are even for video encoders
        target_w = w - (w % 2)
        target_h = h - (h % 2)
        if target_w != w or target_h != h:
            return cv2.resize(image, (target_w, target_h))
        return image

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Ensure even dimensions (required by H.264 / mp4v video writers)
    new_w -= new_w % 2
    new_h -= new_h % 2

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def write_video_pyav(frames, output_path, fps):
    """Write browser-playable H.264 MP4 using PyAV."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=int(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for f in frames:
        # Input frames from cv2 are BGR, convert to RGB
        rgb_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return True


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
# Header Banner
st.markdown(
    """
    <div class="healthcare-header">
        <h1>🩺 Project Ishara — Assistive Navigation Studio</h1>
        <p>Healthcare & Assistive Technology Track | Spatial Perception, Monocular Depth & Dynamic Obstacle Guidance</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory Healthcare Advisory Banner
st.markdown(
    """
    <div class="medical-disclaimer">
        <strong>⚠️ Healthcare & Assistive Aid Advisory (Non-Reliance Notice)</strong><br>
        Project Ishara is an <em>auxiliary perceptual assistance technology</em> engineered to augment environmental awareness for blind and low-vision users. 
        <strong>It is NOT a certified medical diagnostic device, clinical prosthesis, or primary mobility substitute.</strong> 
        Users must <strong>NEVER</strong> place complete reliance on this system for life safety or solitary navigation. 
        It is designed to complement, and must always be used alongside, primary mobility aids such as the <strong>white cane, guide dog, and certified Orientation & Mobility (O&M) training</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.title("⚙️ Navigation Studio Settings")

# Model Discovery
avail_det = get_available_detection_models()
avail_depth = get_available_depth_models()

st.sidebar.subheader("🧠 Perceptual Models")
selected_det_name = st.sidebar.selectbox("Obstacle Detection Model", list(avail_det.keys()))
selected_depth_name = st.sidebar.selectbox("Monocular Depth Model", list(avail_depth.keys()))

st.sidebar.subheader("🔒 Healthcare Privacy & Ethics")
enable_privacy_shield = st.sidebar.checkbox(
    "Enable Privacy Shield (Pedestrian Anonymization)",
    value=True,
    help="Applies Gaussian blurring to detected pedestrians and facial regions to protect personal privacy in public and institutional spaces.",
)
blur_scope = st.sidebar.radio(
    "Anonymization Scope",
    ["Face / Head Region Only", "Full Pedestrian Silhouette"],
    index=0,
    disabled=not enable_privacy_shield,
)

st.sidebar.subheader("📏 Spatial & Proximity Thresholds")
near_thresh = st.sidebar.slider("Critical Near Threshold (m)", 0.5, 3.0, 1.5, 0.1)
mid_thresh = st.sidebar.slider("Cautionary Mid Threshold (m)", 2.0, 7.0, 4.0, 0.5)
conf_thresh = st.sidebar.slider("Detection Confidence", 0.15, 0.80, 0.25, 0.05)
skip_frames = st.sidebar.select_slider("Frame Processing Stride (Skip)", options=[1, 2, 4, 6, 8], value=4)
target_max_dim = st.sidebar.select_slider(
    "Max Video Dimension (Scaling)", options=[480, 640, 720, 1080], value=640
)

st.sidebar.markdown(
    """
    ---
    <div class="privacy-badge">
        🛡️ <strong>Local Processing Guarantee</strong><br>
        All inference executes entirely on-device in local memory. No camera frames or biometrics are ever uploaded to cloud servers.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load Models
det_model, depth_model = load_models(avail_det[selected_det_name], avail_depth[selected_depth_name])

# -----------------------------------------------------------------------------
# Video Upload & Processing Section
# -----------------------------------------------------------------------------
st.subheader("📹 Campus Video Input & Analysis")

uploaded_video = st.file_uploader(
    "Upload campus corridor or walkway video footage (.mp4, .mov, .avi, .mkv)",
    type=["mp4", "mov", "avi", "mkv"],
    help="Upload pre-recorded campus walking footage to preview real-time detection, depth maps, and navigational guidance.",
)

if uploaded_video is not None:
    # Save uploaded bytes to a temporary input file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_file:
        tmp_file.write(uploaded_video.getbuffer())
        temp_input_path = Path(tmp_file.name)

    cap_probe = cv2.VideoCapture(str(temp_input_path))
    orig_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.metric("Native Resolution", f"{orig_w} × {orig_h}")
    with col_meta2:
        st.metric("Frame Rate", f"{orig_fps:.1f} FPS")
    with col_meta3:
        st.metric("Total Frames", f"{total_frames}")
    with col_meta4:
        st.metric("Processing Size", f"Max {target_max_dim}px")

    col_preview, col_controls = st.columns([1, 1])
    with col_preview:
        st.markdown("**Original Video Preview:**")
        st.video(uploaded_video)

    with col_controls:
        st.markdown("**Operational Pipeline:**")
        st.markdown(
            f"""
            - **Detection:** `{selected_det_name}`
            - **Depth Engine:** `{selected_depth_name}`
            - **Privacy Shield:** `{'Active (' + blur_scope + ')' if enable_privacy_shield else 'Disabled'}`
            - **Zone Calculation:** Near `< {near_thresh}m`, Mid `< {mid_thresh}m`
            """
        )
        process_btn = st.button("🚀 Start Assistive Analysis", type="primary", use_container_width=True)

    if process_btn:
        st.markdown("---")
        st.subheader("⚡ Real-Time Perceptual Live Stream")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Telemetry columns
        col_tel1, col_tel2, col_tel3 = st.columns(3)
        tel_hazard = col_tel1.empty()
        tel_closest = col_tel2.empty()
        tel_alert = col_tel3.empty()

        # Live visual frame container
        live_preview = st.empty()

        processed_frames = []
        frame_idx = 0
        written_count = 0
        start_time = time.time()

        cap = cv2.VideoCapture(str(temp_input_path))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Resize frame maintaining aspect ratio to prevent 4K memory bottlenecks
            frame_resized = resize_maintaining_aspect_ratio(frame, max_dim=target_max_dim)
            proc_h, proc_w = frame_resized.shape[:2]

            # 1. Object Detection Inference
            det_res = det_model(
                frame_resized,
                conf=conf_thresh,
                imgsz=min(proc_w, proc_h),
                verbose=False,
            )

            # 2. Monocular Depth Inference
            depth_res = depth_model(
                frame_resized,
                imgsz=min(proc_w, proc_h),
                verbose=False,
            )

            # 3. Healthcare Privacy Shield: Blur pedestrians if enabled
            if enable_privacy_shield and len(det_res[0].boxes) > 0:
                for box in det_res[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = det_model.names.get(cls_id, "").lower()
                    if cls_name in ["person", "pedestrian", "human"]:
                        frame_resized = apply_privacy_blur(
                            frame_resized, box, blur_face_only=(blur_scope == "Face / Head Region Only")
                        )

            # Generate annotated detection view
            annotated_frame = det_res[0].plot()

            # Process depth map
            try:
                depth_raw = depth_res[0].depth.data.cpu().numpy()
                depth_map = cv2.resize(depth_raw, (proc_w, proc_h))
            except Exception:
                # Fallback if depth object schema differs
                depth_map = np.ones((proc_h, proc_w), dtype=np.float32) * 5.0

            # 4. Spatial Zone Reasoning
            detections = []
            for box in det_res[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = det_model.names.get(cls_id, f"Object-{cls_id}")
                h_zone, d_zone, dist_m = get_zone_and_distance(
                    box, depth_map, proc_w, near_thresh, mid_thresh
                )
                detections.append(
                    {
                        "class": cls_name,
                        "h_zone": h_zone,
                        "d_zone": d_zone,
                        "distance_m": dist_m,
                    }
                )

            guidance_message = build_guidance_message(detections)

            # Colormap depth for intuitive visualization (Inferno colormap)
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

            # Side-by-side composite: [Detection + Depth]
            annotated_frame = cv2.resize(annotated_frame, (proc_w, proc_h))
            combined = np.hstack((annotated_frame, depth_colored))

            # Guidance Banner overlay
            banner_bg = (30, 30, 30)
            cv2.rectangle(combined, (0, 0), (proc_w * 2, 50), banner_bg, -1)
            text_color = (0, 69, 255) if "CAUTION" in guidance_message else ((0, 215, 255) if "NOTICE" in guidance_message else (0, 255, 127))
            cv2.putText(
                combined,
                guidance_message,
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                text_color,
                2,
                cv2.LINE_AA,
            )

            # Store composite frame
            processed_frames.append(combined)
            written_count += 1

            # Update Live Visual Stream in Streamlit
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            live_preview.image(
                combined_rgb,
                caption=f"Live Feed — Frame {frame_idx}/{total_frames} | {guidance_message}",
                use_container_width=True,
            )

            # Update Telemetry Widgets
            hazards_count = sum(1 for d in detections if d["d_zone"] in ["Near", "Mid"])
            tel_hazard.metric("Active Obstacles", f"{hazards_count}")
            valid_dists = [d for d in detections if d["distance_m"] is not None]
            if valid_dists:
                closest_obj = min(valid_dists, key=lambda x: x["distance_m"])
                tel_closest.metric("Closest Hazard", f"{closest_obj['class']} ({closest_obj['distance_m']:.1f}m)")
            else:
                tel_closest.metric("Closest Hazard", "None")
            tel_alert.metric("Navigational Status", guidance_message.split(":")[0])

            # Frame Skip optimization (advance cap without decoding unused frames)
            for _ in range(skip_frames - 1):
                if not cap.grab():
                    break
                frame_idx += 1

            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
                elapsed = time.time() - start_time
                fps_curr = written_count / max(elapsed, 0.001)
                status_text.text(f"Processing frame {frame_idx}/{total_frames} ({fps_curr:.1f} FPS)...")

        cap.release()
        try:
            temp_input_path.unlink()
        except Exception:
            pass

        progress_bar.progress(1.0)
        status_text.success("✅ Video processing and spatial reasoning complete!")

        # ---------------------------------------------------------------------
        # Final Video Assembly, In-App Browser Playback & Download
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🎬 Final Processed Video & Download")

        if processed_frames:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_tmp:
                out_path = Path(out_tmp.name)

            output_playback_fps = max(5, int(orig_fps / skip_frames))
            encoding_success = False

            # Attempt 1: PyAV H.264 (Universally compatible with HTML5 video players)
            if HAS_PYAV:
                try:
                    encoding_success = write_video_pyav(processed_frames, out_path, output_playback_fps)
                except Exception as av_err:
                    st.warning(f"PyAV encoding encountered an issue: {av_err}. Attempting OpenCV fallback.")
                    encoding_success = False

            # Attempt 2: OpenCV VideoWriter Fallback
            if not encoding_success:
                h_out, w_out = processed_frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_playback_fps,
                    (w_out, h_out),
                )
                for f in processed_frames:
                    writer.write(f)
                writer.release()

            # Read final video bytes
            with open(out_path, "rb") as f_out:
                final_video_bytes = f_out.read()

            try:
                out_path.unlink()
            except Exception:
                pass

            col_vid, col_dl = st.columns([3, 1])
            with col_vid:
                st.video(final_video_bytes)

            with col_dl:
                st.markdown("**Export Ready:**")
                st.download_button(
                    label="💾 Download Processed Video (.mp4)",
                    data=final_video_bytes,
                    file_name=f"ishara_guidance_{Path(uploaded_video.name).stem}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
                st.info(
                    f"**Summary:**\n- Processed Frames: {len(processed_frames)}\n"
                    f"- Playback Rate: {output_playback_fps} FPS\n"
                    f"- Privacy Anonymization: {'Active' if enable_privacy_shield else 'Inactive'}"
                )
        else:
            st.error("No valid frames were processed from the uploaded video.")
