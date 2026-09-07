from ultralytics import YOLO
import os

MODEL_PATH = "best (1).pt"
video_dir = "batch_videos"   # videos are directly in the current folder (batch_videos)

model = YOLO(MODEL_PATH)

# List actual video files so you can see exact names/extensions
all_videos = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.mov', '.mp4'))])
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