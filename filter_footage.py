import os
import sys
import subprocess
import cv2
import torch
import imageio_ffmpeg
from ultralytics import YOLO

# Suppress noisy C++ level decode logs cleanly without breaking Python stderr
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)

# Dynamically select the best available accelerator hardware
if torch.cuda.is_available():
    DEVICE = 0          # NVIDIA or AMD ROCm
elif torch.backends.mps.is_available():
    DEVICE = "mps"      # Apple Silicon
else:
    DEVICE = "cpu"      # Universal CPU fallback

print(f"Using compute device: {DEVICE}")

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# Upgrade to extra-large model for superior IR feature detection
model = YOLO("yolo11x.pt")

INPUT_DIR = "./repaired_videos"
OUTPUT_DIR = "./person_clips"
PAD_SECONDS = 15
STRIDE = 5
CONF_THRESHOLD = 0.50        # Balanced threshold for IR night vision & daytime detection
REQUIRED_CONSECUTIVE = 2     # Must detect in 2 sampled frames in a row
os.makedirs(OUTPUT_DIR, exist_ok=True)

def merge_intervals(intervals, max_duration):
    if not intervals:
        return []
    padded = [
        (max(0.0, start - PAD_SECONDS), min(max_duration, end + PAD_SECONDS))
        for start, end in intervals
    ]
    padded.sort(key=lambda x: x[0])
    merged = [padded[0]]
    for current_start, current_end in padded[1:]:
        prev_start, prev_end = merged[-1]
        if current_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged

if not os.path.exists(INPUT_DIR):
    print(f"Input directory '{INPUT_DIR}' not found. Please run repair_clips.py or place footage there.")
    sys.exit(0)

video_extensions = (".mp4", ".mkv", ".avi", ".mov")
video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(video_extensions)]
print(f"Found {len(video_files)} video(s) to process.\n")

for idx, filename in enumerate(video_files, 1):
    video_path = os.path.join(INPUT_DIR, filename)
    name_stem, ext = os.path.splitext(filename)
    print(f"[{idx}/{len(video_files)}] Scanning: {filename}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  -> WARNING: Could not open {filename}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    detected_timestamps = []
    pending_streak = []
    frame_idx = 0
    last_timestamp = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Use actual presentation timestamp (PTS) to prevent VFR drift; fallback to frame count
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec > 0:
                current_time = pos_msec / 1000.0
            else:
                current_time = frame_idx / fps
            last_timestamp = max(last_timestamp, current_time)

            if frame_idx % STRIDE != 0:
                continue

            results = model.predict(
                source=frame,
                classes=[0],           # 0 = person
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

            if len(results[0].boxes) > 0:
                pending_streak.append(current_time)
                if len(pending_streak) >= REQUIRED_CONSECUTIVE:
                    # Flush all accumulated timestamps in streak (including first detection)
                    detected_timestamps.extend(pending_streak)
                    pending_streak = []
            else:
                pending_streak = []

    finally:
        cap.release()

    if not detected_timestamps:
        print("  -> No person detected.")
        continue

    # Deduplicate and sort timestamps
    detected_timestamps = sorted(list(set(detected_timestamps)))
    video_duration = last_timestamp if last_timestamp > 0 else (frame_idx / fps)
    raw_events = [(t, t) for t in detected_timestamps]
    segments = merge_intervals(raw_events, video_duration)

    print(f"  -> Found {len(detected_timestamps)} detection points. Extracting {len(segments)} padded segment(s)...")

    for seg_idx, (start_sec, end_sec) in enumerate(segments, 1):
        duration = end_sec - start_sec
        # Always output standard .mp4 for clean playable clips
        out_filename = f"{name_stem}_person_{seg_idx:02d}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        # Re-encode video and audio to guarantee frame-accurate cuts and zero keyframe macroblocking
        cmd = [
            ffmpeg_exe, "-y",
            "-err_detect", "ignore_err",
            "-ss", f"{start_sec:.2f}",
            "-i", video_path,
            "-t", f"{duration:.2f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            out_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0 and os.path.exists(out_path):
            print(f"     Saved: {out_filename} [{start_sec:.1f}s to {end_sec:.1f}s | Duration: {duration:.1f}s]")
        else:
            print(f"     ERROR: Failed to extract segment {seg_idx} for {filename}")

print("\nAll videos processed. Padded clips saved to ./person_clips")