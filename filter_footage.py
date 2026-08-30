import os
import sys
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan security video clips for people and extract trimmed, frame-accurate clips."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="./repaired_videos",
        help="Directory containing prepared input videos (default: ./repaired_videos)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./person_clips",
        help="Directory to save extracted person clips (default: ./person_clips)"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11x.pt",
        help="YOLO model weights path (default: yolo11x.pt)"
    )
    parser.add_argument(
        "--select-model",
        action="store_true",
        default=False,
        help="Interactively select model (1 for yolo11x.pt, 2 for yolo11m.pt, 3 for yolo11s.pt)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.50,
        help="Detection confidence threshold (default: 0.50)"
    )
    parser.add_argument(
        "--stride", "-s",
        type=int,
        default=5,
        help="Process every Nth frame during scan (default: 5)"
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="Number of consecutive sampled frames required to confirm detection (default: 2)"
    )
    parser.add_argument(
        "--pad", "-p",
        type=float,
        default=15.0,
        help="Padding in seconds added before and after detected events (default: 15.0)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        default=False,
        help="Directly compact/downscale extracted clips in a single pass (default: False)"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Maximum width for output video when compacting (default: 1280, aspect ratio preserved)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target framerate when compacting (default: 15)"
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=22,
        help="H.264 CRF quality value, 0-51 where lower is higher quality (default: 22)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (0, 'mps', 'cpu'). If omitted, automatically selects fastest available."
    )
    return parser.parse_args()

def prompt_model_selection():
    """Interactively prompt user to select a YOLO model."""
    print("\n" + "=" * 50)
    print(" Select a YOLO Model:")
    print("  [1] yolo11x.pt (Extra-Large: Maximum Accuracy / Best for GPU & Night IR)")
    print("  [2] yolo11m.pt (Medium: Fast & Balanced)")
    print("  [3] yolo11s.pt (Small: Very Fast & Lightweight)")
    print("=" * 50)
    while True:
        try:
            choice = input("Enter choice [1, 2, or 3] (default: 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled. Defaulting to yolo11x.pt.")
            return "yolo11x.pt"

        if choice in ("", "1"):
            print("Selected: yolo11x.pt\n")
            return "yolo11x.pt"
        elif choice == "2":
            print("Selected: yolo11m.pt\n")
            return "yolo11m.pt"
        elif choice == "3":
            print("Selected: yolo11s.pt\n")
            return "yolo11s.pt"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def resolve_device(device_override=None):
    if device_override is not None:
        return device_override
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def merge_intervals(intervals, max_duration, pad_seconds):
    if not intervals:
        return []
    padded = [
        (max(0.0, start - pad_seconds), min(max_duration, end + pad_seconds))
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

def process_footage(args):
    device = resolve_device(args.device)
    print(f"Using compute device: {device}")
    print(f"Loading YOLO model: {args.model}...")
    model = YOLO(args.model)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_dir):
        print(f"Input directory '{args.input_dir}' not found. Please run repair_clips.py or place footage there.")
        return

    video_extensions = (".mp4", ".mkv", ".avi", ".mov")
    video_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(video_extensions)]

    if not video_files:
        print(f"No video files found in '{args.input_dir}'.")
        return

    print(f"Found {len(video_files)} video(s) to process.\n")

    for idx, filename in enumerate(video_files, 1):
        video_path = os.path.join(args.input_dir, filename)
        name_stem, _ = os.path.splitext(filename)
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

                if frame_idx % args.stride != 0:
                    continue

                results = model.predict(
                    source=frame,
                    classes=[0],           # 0 = person
                    conf=args.conf,
                    device=device,
                    verbose=False
                )

                if len(results[0].boxes) > 0:
                    pending_streak.append(current_time)
                    if len(pending_streak) >= args.consecutive:
                        detected_timestamps.extend(pending_streak)
                        pending_streak = []
                else:
                    pending_streak = []

        finally:
            cap.release()

        if not detected_timestamps:
            print("  -> No person detected.")
            continue

        detected_timestamps = sorted(list(set(detected_timestamps)))
        video_duration = last_timestamp if last_timestamp > 0 else (frame_idx / fps)
        raw_events = [(t, t) for t in detected_timestamps]
        segments = merge_intervals(raw_events, video_duration, args.pad)

        print(f"  -> Found {len(detected_timestamps)} detection points. Extracting {len(segments)} padded segment(s)...")

        for seg_idx, (start_sec, end_sec) in enumerate(segments, 1):
            duration = end_sec - start_sec
            out_filename = f"{name_stem}_person_{seg_idx:02d}.mp4"
            out_path = os.path.join(args.output_dir, out_filename)

            # Build FFmpeg command with frame-accurate re-encoding & optional one-pass compaction
            cmd = [
                ffmpeg_exe, "-y",
                "-err_detect", "ignore_err",
                "-ss", f"{start_sec:.2f}",
                "-i", video_path,
                "-t", f"{duration:.2f}",
            ]

            if args.compact:
                # Proportional scaling without distortion + framerate reduction
                vf_filters = f"scale='min({args.max_width},iw)':-2,fps={args.fps}"
                cmd.extend(["-vf", vf_filters])

            cmd.extend([
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", str(args.crf),
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                out_path
            ])

            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0 and os.path.exists(out_path):
                out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
                print(f"     Saved: {out_filename} [{start_sec:.1f}s to {end_sec:.1f}s | Duration: {duration:.1f}s | {out_size_mb:.1f} MB]")
            else:
                print(f"     ERROR: Failed to extract segment {seg_idx} for {filename}")

    print(f"\nAll videos processed. Clips saved to '{args.output_dir}'")

if __name__ == "__main__":
    args = parse_args()
    if args.select_model:
        args.model = prompt_model_selection()
    process_footage(args)