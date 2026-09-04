#!/usr/bin/env python3
"""Batch benchmark wrapper for the existing detection/depth video pipeline."""

import argparse
import csv
import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import psutil
except ImportError:
    psutil = None


# Edit these defaults or provide --model-paths model_paths.json.
DEFAULT_MODEL_PATHS = {
    "det": {
        "raw": {
            "nano": "yolo26n.pt",
            "small": "yolo26s.pt",
            "medium": "yolo26m.pt",
        },
        "openvino": {
            "nano": "yolo26n_openvino_model",
            "small": "yolo26s_openvino_model",
            "medium": "yolo26m_openvino_model",
        },
    },
    "depth": {
        "raw": {
            "nano": "yolo26n-depth.pt",
            "small": "yolo26s-depth.pt",
            "medium": "yolo26m-depth.pt",
        },
        "openvino": {
            "nano": "yolo26n-depth_openvino_model",
            "small": "yolo26s-depth_openvino_model",
            "medium": "yolo26m-depth_openvino_model",
        },
    },
}

REFERENCE_HARDWARE = {
    "machine": "HP ProBook 440 G3",
    "cpu": "Intel Core i5-6300U @ 2.40GHz (2 cores / 4 threads, max 3.0GHz)",
    "gpu": "Intel HD 520 integrated graphics; no CUDA/discrete GPU",
    "cache": "L1 64KiB, L2 512KiB, L3 3MiB",
    "ram": "8GB DDR4-2133",
    "storage": "256GB SATA SSD",
    "benchmark_device": "cpu",
}

# Estimated relative throughput versus the measured i5-6300U result. These are
# planning ranges, not measured phone results. "Accelerated" assumes conversion
# to a phone-native NPU/GPU runtime; OpenVINO itself is generally not a phone runtime.
MOBILE_ESTIMATE_FACTORS = {
    "cpu_fp32": (0.5, 1.0),
    "cpu_fp16": (0.7, 1.4),
    "accelerated_fp32": (1.5, 4.0),
    "accelerated_fp16": (3.0, 8.0),
}

# --- Distance thresholds in meters — tune these after watching your test output ---
NEAR_THRESHOLD = 1.5
MID_THRESHOLD = 4.0

# --- Overlay text settings ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2.2
FONT_THICKNESS = 4
LINE_GAP = 20
TOP_MARGIN = 60
RIGHT_MARGIN = 20
TOP_N = 3


# The four functions below are unchanged from the supplied code.
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


@dataclass(frozen=True)
class PassConfig:
    name: str
    test_group: str
    test_variable: str
    test_value: str
    skip_frames: int = 3
    det_imgsz: int = 640
    depth_imgsz: int = 768
    model_source: str = "raw"
    det_model_size: str = "nano"
    depth_model_size: str = "small"
    device: str = "cpu"
    frame_scale: float = 1.0
    output_scale: float = 1.0
    output_fps_divisor: int = 3
    codec: str = "mp4v"
    half: bool = False


BASELINE = PassConfig(
    name="balanced",
    test_group="preset",
    test_variable="preset",
    test_value="balanced",
)


class PassUnavailable(RuntimeError):
    pass


def parse_video_selection(spec):
    numbers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
            if not match:
                raise argparse.ArgumentTypeError(f"Invalid range: {part}")
            start, end = map(int, match.groups())
            step = 1 if start <= end else -1
            numbers.extend(range(start, end + step, step))
        elif part.isdigit():
            numbers.append(int(part))
        else:
            raise argparse.ArgumentTypeError(f"Invalid video number: {part}")
    if not numbers:
        raise argparse.ArgumentTypeError("At least one video number is required")
    return list(dict.fromkeys(numbers))


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-_")


def build_passes(mode):
    passes = []

    def add_isolated(variable, values, field):
        for value in values:
            kwargs = {field: value}
            passes.append(replace(
                BASELINE,
                name=f"isolated_{variable}_{safe_name(value)}",
                test_group="isolated",
                test_variable=variable,
                test_value=str(value),
                **kwargs,
            ))

    if mode in ("full", "isolated"):
        add_isolated("skip_frames", [1, 2, 3, 5, 8, 10], "skip_frames")
        add_isolated("det_imgsz", [320, 480, 640, 960], "det_imgsz")
        add_isolated("depth_imgsz", [384, 512, 640, 768], "depth_imgsz")
        add_isolated("model_source", ["raw", "openvino"], "model_source")
        add_isolated("det_model_size", ["nano", "small", "medium"], "det_model_size")
        add_isolated("depth_model_size", ["small", "nano", "medium"], "depth_model_size")
        add_isolated("device", ["cpu"], "device")
        add_isolated("frame_scale", [1.0, 0.75, 0.5, 0.25], "frame_scale")
        add_isolated("output_scale", [1.0, 0.75, 0.5], "output_scale")
        add_isolated("output_fps_divisor", [3, 2, 1], "output_fps_divisor")
        add_isolated("codec", ["mp4v", "avc1"], "codec")
        add_isolated("precision", [False, True], "half")

    if mode in ("full", "presets"):
        passes.extend([
            replace(
                BASELINE,
                name="max_quality",
                test_value="max_quality",
                det_imgsz=960,
                depth_imgsz=768,
                skip_frames=1,
            ),
            BASELINE,
            replace(
                BASELINE,
                name="medium",
                test_value="medium",
                det_imgsz=480,
                depth_imgsz=640,
                skip_frames=5,
                frame_scale=0.75,
            ),
            replace(
                BASELINE,
                name="fast",
                test_value="fast",
                det_imgsz=320,
                depth_imgsz=384,
                skip_frames=8,
                frame_scale=0.5,
                half=True,
                model_source="openvino",
            ),
            replace(
                BASELINE,
                name="mobile_target",
                test_value="mobile_target",
                det_imgsz=320,
                depth_imgsz=384,
                skip_frames=10,
                frame_scale=0.25,
                half=True,
                model_source="openvino",
            ),
        ])

    return passes


def deep_merge(base, override):
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_model_paths(path):
    if path is None:
        return DEFAULT_MODEL_PATHS
    with path.open("r", encoding="utf-8") as handle:
        return deep_merge(DEFAULT_MODEL_PATHS, json.load(handle))


def resolve_input(videos_dir, number):
    padded = f"{number:02d}"
    candidate = videos_dir / f"V{padded}.MOV"
    if candidate.is_file():
        return candidate
    candidate = videos_dir / f"V{padded}.mov"
    if candidate.is_file():
        return candidate
    return videos_dir / f"V{padded}.MOV"


def model_path_for(model_paths, kind, source, size):
    try:
        return model_paths[kind][source][size]
    except KeyError as exc:
        raise PassUnavailable(
            f"No model path configured for {kind}/{source}/{size}"
        ) from exc


def get_model(model_cache, path):
    path_key = str(path)
    if path_key not in model_cache:
        if not Path(path).exists():
            raise PassUnavailable(f"Model not found: {path}")
        model_cache[path_key] = YOLO(path_key)
    return model_cache[path_key]


def thermal_snapshot():
    snapshot = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cpu_frequency_mhz": None,
        "cpu_temperature_c": None,
    }
    if psutil is None:
        snapshot["monitoring_note"] = "psutil is not installed"
        return snapshot

    try:
        frequency = psutil.cpu_freq()
        if frequency is not None:
            snapshot["cpu_frequency_mhz"] = round(float(frequency.current), 2)
    except (AttributeError, OSError):
        pass

    try:
        temperatures = psutil.sensors_temperatures()
        readings = [
            entry.current
            for entries in temperatures.values()
            for entry in entries
            if entry.current is not None
        ]
        if readings:
            snapshot["cpu_temperature_c"] = round(float(max(readings)), 2)
    except (AttributeError, OSError):
        snapshot["monitoring_note"] = "CPU temperature unavailable on this OS/hardware"

    return snapshot


def format_timestamp(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def mobile_estimate(config, source_equivalent_fps):
    if config.model_source == "openvino":
        key = "accelerated_fp16" if config.half else "accelerated_fp32"
        runtime_note = (
            "Estimated for a converted phone-native NPU/GPU model; "
            "OpenVINO itself is not assumed to run on the phone."
        )
    else:
        key = "cpu_fp16" if config.half else "cpu_fp32"
        runtime_note = "Estimated mobile CPU execution without a dedicated accelerator."

    low, high = MOBILE_ESTIMATE_FACTORS[key]
    low_fps = source_equivalent_fps * low
    high_fps = source_equivalent_fps * high
    return {
        "mobile_factor_low": low,
        "mobile_factor_high": high,
        "mobile_estimated_source_fps_low": round(low_fps, 4),
        "mobile_estimated_source_fps_high": round(high_fps, 4),
        "mobile_realtime_30fps": "likely" if low_fps >= 30 else ("possible" if high_fps >= 30 else "unlikely"),
        "mobile_estimate_note": runtime_note,
    }


def should_encode_temporary(config):
    return config.test_group == "isolated" and config.test_variable in {
        "output_scale", "output_fps_divisor", "codec"
    }


def open_writer(path, config, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*config.codec)
    writer = cv2.VideoWriter(
        str(path), fourcc, fps / config.output_fps_divisor, (width * 2, height)
    )
    if not writer.isOpened():
        writer.release()
        raise PassUnavailable(
            f"Codec {config.codec!r} is unavailable for output {path.suffix}"
        )
    return writer


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_csv(path, row, fieldnames):
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def write_caution_log(path, caution_events):
    fields = [
        "timestamp_seconds", "timestamp", "source_frame", "object",
        "zone", "distance_m", "confidence", "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(caution_events)


def validate_video(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Video contains no readable frames: {path}")
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video metadata: {path}")
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_total": frame_total,
        }, frame
    finally:
        cap.release()


def warm_up(det_model, depth_model, frame, config):
    warm_frame = frame
    if config.frame_scale != 1.0:
        warm_frame = cv2.resize(
            frame, None, fx=config.frame_scale, fy=config.frame_scale,
            interpolation=cv2.INTER_AREA,
        )
    det_model(
        warm_frame,
        imgsz=config.det_imgsz,
        device=config.device,
        half=config.half,
        verbose=False,
    )
    depth_model(
        warm_frame,
        imgsz=config.depth_imgsz,
        device=config.device,
        half=config.half,
        verbose=False,
    )


def process_pass(
    video_number,
    input_path,
    metadata,
    first_frame,
    config,
    save_video,
    output_label,
    attempt_dir,
    model_paths,
    model_cache,
    pass_index,
):
    det_path = model_path_for(
        model_paths, "det", config.model_source, config.det_model_size
    )
    depth_path = model_path_for(
        model_paths, "depth", config.model_source, config.depth_model_size
    )
    det_model = get_model(model_cache, det_path)
    depth_model = get_model(model_cache, depth_path)

    # The throwaway calls are intentionally outside the timed section.
    warm_up(det_model, depth_model, first_frame, config)

    if config.test_group == "preset":
        base_name = f"{video_number}_processed_{output_label}"
    else:
        base_name = f"{video_number}_{config.name}"

    output_path = attempt_dir / f"{base_name}.MOV" if save_video else None
    temporary_path = None
    writer = None
    cap = None
    caution_events = []
    detection_count = 0
    confidence_sum = 0.0
    source_frames_read = 0
    processed_frames = 0
    written_frames = 0
    pass_completed = False

    output_width = max(1, round(metadata["width"] * config.output_scale))
    output_height = max(1, round(metadata["height"] * config.output_scale))

    try:
        if save_video:
            writer = open_writer(
                output_path, config, metadata["fps"], output_width, output_height
            )
        elif should_encode_temporary(config):
            # Encoding-dependent isolated tests require a real encoder sink. The
            # temporary video is deleted after timing and is never retained.
            suffix = ".mp4" if config.codec in ("mp4v", "avc1") else ".MOV"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="benchmark_discard_", suffix=suffix, dir=str(attempt_dir)
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            writer = open_writer(
                temporary_path, config, metadata["fps"], output_width, output_height
            )

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video: {input_path}")

        thermal_start = thermal_snapshot()
        started = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            source_frames_read += 1
            frame_count = source_frames_read

            # Skip frames to speed up CPU processing
            if frame_count % config.skip_frames != 0:
                continue

            if config.frame_scale != 1.0:
                frame = cv2.resize(
                    frame, None, fx=config.frame_scale, fy=config.frame_scale,
                    interpolation=cv2.INTER_AREA,
                )

            frame_width = frame.shape[1]

            # Run Detection Inference
            det_results = det_model(
                frame,
                imgsz=config.det_imgsz,
                device=config.device,
                half=config.half,
                verbose=False,
            )
            det_frame = det_results[0].plot()

            # Run Depth Inference
            depth_results = depth_model(
                frame,
                imgsz=config.depth_imgsz,
                device=config.device,
                half=config.half,
                verbose=False,
            )
            depth_map = depth_results[0].depth.data.cpu().numpy()

            # Structure detections with zone + distance, then rank top-N.
            structured_detections = []
            for box in det_results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = det_model.names[cls_id]
                confidence = float(box.conf[0])
                h_zone, d_zone, distance = get_zone_and_distance(
                    box, depth_map, frame_width
                )

                detection = {
                    "class": class_name,
                    "confidence": confidence,
                    "h_zone": h_zone,
                    "d_zone": d_zone,
                    "distance_m": distance,
                }
                structured_detections.append(detection)
                detection_count += 1
                confidence_sum += confidence

                if save_video and d_zone == "near":
                    timestamp_seconds = (frame_count - 1) / metadata["fps"]
                    caution_events.append({
                        "timestamp_seconds": round(timestamp_seconds, 3),
                        "timestamp": format_timestamp(timestamp_seconds),
                        "source_frame": frame_count,
                        "object": class_name,
                        "zone": h_zone,
                        "distance_m": round(distance, 4) if distance is not None else None,
                        "confidence": round(confidence, 6),
                        "message": build_guidance_message(detection),
                    })

            top_detections = pick_priority_detections(structured_detections)
            if top_detections:
                messages = [
                    f"{i+1}) {build_guidance_message(d)}"
                    for i, d in enumerate(top_detections)
                ]
            else:
                messages = ["Path clear"]

            # Process Depth Map for Visualization
            depth_normalized = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX
            )
            depth_uint8 = np.uint8(depth_normalized)
            depth_colored = cv2.applyColorMap(
                depth_uint8, cv2.COLORMAP_INFERNO
            )

            # Resize and concatenate side-by-side.
            det_frame_resized = cv2.resize(
                det_frame, (output_width, output_height)
            )
            depth_colored_resized = cv2.resize(
                depth_colored, (output_width, output_height)
            )
            combined_frame = np.hstack(
                (det_frame_resized, depth_colored_resized)
            )
            draw_right_aligned_lines(combined_frame, messages)

            if writer is not None:
                writer.write(combined_frame)
                written_frames += 1
            processed_frames += 1

        elapsed = time.perf_counter() - started
        thermal_end = thermal_snapshot()

        if processed_frames == 0:
            raise RuntimeError("No frames were processed in this pass")

        processed_fps = processed_frames / elapsed if elapsed else 0.0
        source_equivalent_fps = source_frames_read / elapsed if elapsed else 0.0
        source_duration = source_frames_read / metadata["fps"]
        real_time_factor = source_duration / elapsed if elapsed else 0.0

        result = {
            "pass_index": pass_index,
            "video_number": video_number,
            "input_video": str(input_path),
            "status": "success",
            "error": "",
            "base_name": base_name,
            "output_video": str(output_path) if output_path else "",
            "video_saved": save_video,
            "temporary_encoding_benchmark": temporary_path is not None,
            "elapsed_seconds": round(elapsed, 6),
            "source_frames_read": source_frames_read,
            "processed_frames": processed_frames,
            "written_frames": written_frames,
            "processed_fps": round(processed_fps, 4),
            "source_equivalent_fps": round(source_equivalent_fps, 4),
            "source_video_duration_seconds": round(source_duration, 4),
            "real_time_factor": round(real_time_factor, 4),
            "detection_count": detection_count,
            "average_confidence": (
                round(confidence_sum / detection_count, 6)
                if detection_count else None
            ),
            "caution_event_count": len(caution_events) if save_video else None,
            "det_model_path": str(det_path),
            "depth_model_path": str(depth_path),
            "thermal_start": thermal_start,
            "thermal_end": thermal_end,
            **asdict(config),
        }
        result.update(mobile_estimate(config, source_equivalent_fps))

        if save_video:
            write_caution_log(
                attempt_dir / f"{base_name}_cautions.csv", caution_events
            )

        pass_completed = True
        return result, base_name
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if not pass_completed and output_path is not None and output_path.exists():
            output_path.unlink()


def failed_result(video_number, input_path, config, pass_index, error, status="failed"):
    return {
        "pass_index": pass_index,
        "video_number": video_number,
        "input_video": str(input_path),
        "status": status,
        "error": str(error),
        "base_name": "",
        "output_video": "",
        "video_saved": False,
        "temporary_encoding_benchmark": False,
        "elapsed_seconds": None,
        "source_frames_read": None,
        "processed_frames": None,
        "written_frames": None,
        "processed_fps": None,
        "source_equivalent_fps": None,
        "source_video_duration_seconds": None,
        "real_time_factor": None,
        "detection_count": None,
        "average_confidence": None,
        "caution_event_count": None,
        "det_model_path": "",
        "depth_model_path": "",
        "thermal_start": None,
        "thermal_end": None,
        "mobile_factor_low": None,
        "mobile_factor_high": None,
        "mobile_estimated_source_fps_low": None,
        "mobile_estimated_source_fps_high": None,
        "mobile_realtime_30fps": "unknown",
        "mobile_estimate_note": "No estimate because the pass did not complete.",
        **asdict(config),
    }


SUMMARY_FIELDS = [
    "pass_index", "video_number", "input_video", "status", "error",
    "test_group", "test_variable", "test_value", "name",
    "video_saved", "output_video", "temporary_encoding_benchmark",
    "elapsed_seconds", "source_frames_read", "processed_frames", "written_frames",
    "processed_fps", "source_equivalent_fps", "source_video_duration_seconds",
    "real_time_factor", "detection_count", "average_confidence",
    "caution_event_count", "skip_frames", "det_imgsz", "depth_imgsz",
    "model_source", "det_model_size", "depth_model_size", "det_model_path",
    "depth_model_path", "device", "frame_scale", "output_scale",
    "output_fps_divisor", "codec", "half", "thermal_start", "thermal_end",
    "mobile_factor_low", "mobile_factor_high",
    "mobile_estimated_source_fps_low", "mobile_estimated_source_fps_high",
    "mobile_realtime_30fps", "mobile_estimate_note",
]


def environment_record():
    record = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "ultralytics": None,
        "psutil": getattr(psutil, "__version__", None),
    }
    try:
        import ultralytics
        record["ultralytics"] = ultralytics.__version__
    except (ImportError, AttributeError):
        pass
    return record


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch video benchmark for detection + depth inference"
    )
    parser.add_argument(
        "--videos-dir", type=Path, default=Path("batch_videos"),
        help="Folder containing input videos V01.MOV ... V28.MOV",
    )

    parser.add_argument(
        "--videos", type=parse_video_selection, required=True,
        help="Selection such as 1-5 or 1,3,7-10",
    )
    parser.add_argument(
        "--reference", type=int, required=True,
        help="Selected video whose output is saved for every combined preset",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("benchmark_outputs"),
        help="Parent directory for attempt_<timestamp>",
    )
    parser.add_argument(
        "--model-paths", type=Path,
        help="Optional JSON overriding DEFAULT_MODEL_PATHS",
    )
    parser.add_argument(
        "--mode", choices=("full", "isolated", "presets"), default="full",
        help="Run all tests, isolated tests only, or presets only",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference not in args.videos:
        raise SystemExit("--reference must be included in --videos")

    run_started = datetime.now()
    timestamp = run_started.strftime("%Y%m%d_%H%M%S_%f")
    args.output_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = args.output_root / f"attempt_{timestamp}"
    attempt_dir.mkdir(exist_ok=False)

    model_paths = load_model_paths(args.model_paths)
    passes = build_passes(args.mode)
    summary_path = attempt_dir / "benchmark_summary.csv"
    model_cache = {}
    results = []
    succeeded_videos = []
    failed_videos = []
    pass_failures = []
    pass_index = 0

    config_record = {
        "run_timestamp": run_started.isoformat(timespec="microseconds"),
        "attempt_directory": str(attempt_dir),
        "videos_directory": str(args.videos_dir),
        "selected_videos": args.videos,
        "reference_video": args.reference,
        "mode": args.mode,
        "reference_hardware": REFERENCE_HARDWARE,
        "runtime_environment": environment_record(),
        "model_paths": model_paths,
        "baseline": asdict(BASELINE),
        "passes": [asdict(item) for item in passes],
        "mobile_estimation_factors": MOBILE_ESTIMATE_FACTORS,
        "mobile_estimation_warning": (
            "Planning estimates only. Validate on the target phone and use a "
            "phone-native runtime such as Core ML, NNAPI, TFLite, or vendor SDK."
        ),
    }
    write_json(attempt_dir / "config.json", config_record)

    print(f"Output folder: {attempt_dir}")
    print(f"Videos: {args.videos}; reference: {args.reference}; passes/video: {len(passes)}")

    try:
        for video_number in args.videos:
            input_path = resolve_input(args.videos_dir, video_number)
            print(f"\nVideo {video_number}: {input_path}")

            try:
                metadata, first_frame = validate_video(input_path)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failed_videos.append({"video_number": video_number, "error": error})
                video_row = failed_result(
                    video_number, input_path, BASELINE, pass_index, error
                )
                video_row["test_group"] = "video"
                video_row["test_variable"] = "validation"
                video_row["test_value"] = ""
                append_csv(summary_path, video_row, SUMMARY_FIELDS)
                results.append(video_row)
                print(f"  FAILED: {error}")
                continue

            successful_passes = 0
            for config in passes:
                pass_index += 1
                save_video = (
                    config.test_group == "preset"
                    and (
                        video_number == args.reference
                        or config.name == "balanced"
                    )
                )
                # The storage rule explicitly names non-reference baseline files
                # *_processed_baseline.MOV; the measured preset remains "balanced".
                output_label = (
                    "baseline"
                    if save_video and video_number != args.reference and config.name == "balanced"
                    else config.name
                )
                print(
                    f"  [{pass_index}] {config.test_group}/{config.name}"
                    f"{' (saving video)' if save_video else ''}"
                )

                try:
                    result, base_name = process_pass(
                        video_number=video_number,
                        input_path=input_path,
                        metadata=metadata,
                        first_frame=first_frame,
                        config=config,
                        save_video=save_video,
                        output_label=output_label,
                        attempt_dir=attempt_dir,
                        model_paths=model_paths,
                        model_cache=model_cache,
                        pass_index=pass_index,
                    )
                    successful_passes += 1
                    print(
                        f"    {result['elapsed_seconds']:.2f}s, "
                        f"detections={result['detection_count']}, "
                        f"source-equivalent FPS={result['source_equivalent_fps']:.2f}"
                    )
                except PassUnavailable as exc:
                    result = failed_result(
                        video_number, input_path, config, pass_index, exc,
                        status="unavailable",
                    )
                    base_name = (
                        f"{video_number}_processed_{output_label}"
                        if config.test_group == "preset"
                        else f"{video_number}_{config.name}"
                    )
                    pass_failures.append({
                        "video_number": video_number,
                        "pass": config.name,
                        "status": "unavailable",
                        "error": str(exc),
                    })
                    print(f"    UNAVAILABLE: {exc}")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    result = failed_result(
                        video_number, input_path, config, pass_index, error
                    )
                    base_name = (
                        f"{video_number}_processed_{output_label}"
                        if config.test_group == "preset"
                        else f"{video_number}_{config.name}"
                    )
                    result["traceback"] = traceback.format_exc()
                    pass_failures.append({
                        "video_number": video_number,
                        "pass": config.name,
                        "status": "failed",
                        "error": error,
                    })
                    print(f"    FAILED: {error}")

                write_json(attempt_dir / f"{base_name}.json", result)
                append_csv(summary_path, result, SUMMARY_FIELDS)
                results.append(result)

            if successful_passes:
                succeeded_videos.append(video_number)
            else:
                failed_videos.append({
                    "video_number": video_number,
                    "error": "All benchmark passes failed or were unavailable",
                })
    except KeyboardInterrupt:
        print("\nInterrupted; writing partial run summary.", file=sys.stderr)
    finally:
        run_finished = datetime.now()
        run_summary = {
            "run_started": run_started.isoformat(timespec="microseconds"),
            "run_finished": run_finished.isoformat(timespec="microseconds"),
            "elapsed_seconds": round((run_finished - run_started).total_seconds(), 3),
            "attempt_directory": str(attempt_dir),
            "selected_videos": args.videos,
            "reference_video": args.reference,
            "succeeded_videos": succeeded_videos,
            "failed_videos": failed_videos,
            "successful_passes": sum(r.get("status") == "success" for r in results),
            "failed_or_unavailable_passes": pass_failures,
            "benchmark_summary": str(summary_path),
        }
        write_json(attempt_dir / "run_summary.json", run_summary)

    print(f"\nRun complete: {attempt_dir}")
    print(f"Succeeded videos: {succeeded_videos}")
    if failed_videos:
        print(f"Failed videos: {failed_videos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())