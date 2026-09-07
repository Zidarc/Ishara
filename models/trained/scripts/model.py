import os
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR.parent / "checkpoints"
DATASETS_DIR = SCRIPT_DIR.parent.parent.parent / "datasets" / "raw_videos"

# Check for checkpoints (best (1).pt or any .pt file in checkpoints directory)
model_path = CHECKPOINTS_DIR / "best (1).pt"
if not model_path.exists():
    model_path = CHECKPOINTS_DIR / "best.pt"
if not model_path.exists():
    model_path = "best (1).pt"

video_dir = DATASETS_DIR / "batch_videos"
if not video_dir.exists():
    video_dir = SCRIPT_DIR / "batch_videos"

model = YOLO(str(model_path))

# List actual video files so you can see exact names/extensions
all_videos = sorted([f.name for f in Path(video_dir).glob("*") if f.suffix.lower() in ('.mov', '.mp4')])
print("Videos found in folder:", all_videos)

# Pick the one(s) you want to test — use exact name from the list above
selected_videos = ["V26.MOV"]

for vid in selected_videos:
    vid_path = os.path.join(video_dir, vid)
    print(f"Running inference on: {vid}")

    results = model.predict(
        source=vid_path,
        save=True,
        imgsz=1280,
        conf=0.25,
        vid_stride=30,   # process every 5th frame — raise this (e.g. 15, 30) for even fewer frames
        project="test_results",
        name=os.path.splitext(vid)[0]
    )

print("Done. Check the test_results folder for output video(s).")