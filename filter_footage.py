import os
import sys
import subprocess
import cv2
import imageio_ffmpeg
from ultralytics import YOLO

# Suppress Python-level warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
model = YOLO("yolo11m.pt")

INPUT_DIR = "./repaired_videos"
OUTPUT_DIR = "./person_clips"
PAD_SECONDS = 15
STRIDE = 5
CONF_THRESHOLD = 0.45
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
    frame_idx = 0

    # Temporarily redirect low-level C stderr (FFmpeg console chatter)
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull, stderr_fd)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % STRIDE != 0:
                continue

            results = model.predict(
                source=frame,
                classes=[0],
                conf=CONF_THRESHOLD,
                device=0,
                verbose=False
            )

            if len(results[0].boxes) > 0:
                timestamp = frame_idx / fps
                detected_timestamps.append(timestamp)

    finally:
        # Restore normal console stderr output
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)
        os.close(devnull)
        cap.release()

    if not detected_timestamps:
        print("  -> No person detected.")
        continue

    # Use actual decoded duration rather than metadata in case of truncation
    video_duration = frame_idx / fps
    raw_events = [(t, t) for t in detected_timestamps]
    segments = merge_intervals(raw_events, video_duration)

    print(f"  -> Found {len(detected_timestamps)} detection frames. Extracting {len(segments)} padded segment(s)...")

    for seg_idx, (start_sec, end_sec) in enumerate(segments, 1):
        duration = end_sec - start_sec
        out_filename = f"{name_stem}_person_{seg_idx:02d}{ext}"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        cmd = [
            ffmpeg_exe, "-y",
            "-err_detect", "ignore_err",   # Tell FFmpeg to ignore trailing malformed slices
            "-ss", f"{start_sec:.2f}",
            "-i", video_path,
            "-t", f"{duration:.2f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_path
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"     Saved: {out_filename} [{start_sec:.1f}s to {end_sec:.1f}s | Duration: {duration:.1f}s]")

print("\nAll videos processed. Padded clips saved to ./person_clips")