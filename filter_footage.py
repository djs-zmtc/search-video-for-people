import os
import sys
import re
import argparse
import subprocess
from datetime import datetime
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
        "--log-dir", "-l",
        default="./log",
        help="Directory to save run logs (default: ./log)"
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
        help="H.264 CRF quality level, 0-51 where lower is higher quality (default: 22)"
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

def format_time_hms(seconds):
    """Format seconds into HH:MM:SS.ss string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

def build_camera_manifest(video_files):
    """
    Pre-parse video files to detect unique camera/stream prefixes (e.g. name_{GUID}_)
    and assign concise aliases like [CAM-01], [CAM-02], etc.
    """
    guid_pattern = re.compile(r'^(.*\{[0-9a-fA-F\-]{32,38}\}_?)(.*)$')
    fallback_bracket_pattern = re.compile(r'^(.*\{[^\}]+\}_?)(.*)$')

    prefixes = []
    file_mapping = {}

    for filename in video_files:
        match = guid_pattern.match(filename) or fallback_bracket_pattern.match(filename)
        if match:
            prefix, remainder = match.group(1), match.group(2)
            if prefix not in prefixes:
                prefixes.append(prefix)
            file_mapping[filename] = (prefix, remainder)
        else:
            file_mapping[filename] = (None, filename)

    prefix_to_alias = {p: f"[CAM-{idx:02d}]" for idx, p in enumerate(prefixes, 1)}

    manifest_lines = []
    if prefixes:
        manifest_lines.append("--- Active Camera Streams ---")
        for p in prefixes:
            alias = prefix_to_alias[p]
            clean_name = p.rstrip('_')
            manifest_lines.append(f" {alias} {clean_name}")
        manifest_lines.append("-----------------------------")

    file_to_display = {}
    for filename, (prefix, remainder) in file_mapping.items():
        if prefix and prefix in prefix_to_alias:
            alias = prefix_to_alias[prefix]
            file_to_display[filename] = f"{alias} {remainder}"
        else:
            file_to_display[filename] = filename

    return file_to_display, manifest_lines

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
    start_time = datetime.now()
    datestamp = start_time.strftime("%Y%m%dT%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    log_filename = f"person-found_{datestamp}.log"
    log_path = os.path.join(args.log_dir, log_filename)

    if not os.path.exists(args.input_dir):
        print(f"Input directory '{args.input_dir}' not found. Please run repair_clips.py or place footage there.")
        return

    video_extensions = (".mp4", ".mkv", ".avi", ".mov")
    video_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(video_extensions)]

    if not video_files:
        print(f"No video files found in '{args.input_dir}'.")
        return

    file_to_display, manifest_lines = build_camera_manifest(video_files)

    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"=== Session Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')} (ISO: {start_time.isoformat()}) ===\n")
    if manifest_lines:
        log_file.write("\n".join(manifest_lines) + "\n")
    log_file.write("\n")
    log_file.flush()

    try:
        device = resolve_device(args.device)
        print(f"Using compute device: {device}")
        print(f"Loading YOLO model: {args.model}...")
        model = YOLO(args.model)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        os.makedirs(args.output_dir, exist_ok=True)

        if manifest_lines:
            print("\n".join(manifest_lines))
        print(f"\nSession log: {log_path}")
        print(f"Found {len(video_files)} video(s) to process.\n")

        for idx, filename in enumerate(video_files, 1):
            video_path = os.path.join(args.input_dir, filename)
            name_stem, _ = os.path.splitext(filename)
            display_name = file_to_display.get(filename, filename)
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

                # Find earliest detection timestamp within this segment
                detections_in_segment = [t for t in detected_timestamps if start_sec <= t <= end_sec]
                first_detect = min(detections_in_segment) if detections_in_segment else start_sec

                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                clip_label = f"person_{seg_idx:02d}.mp4"
                log_entry = (
                    f"[{now_str}] {display_name} | "
                    f"Detected: {format_time_hms(first_detect)} ({first_detect:.2f}s) | "
                    f"Clip: {clip_label} ({format_time_hms(start_sec)} to {format_time_hms(end_sec)})\n"
                )
                log_file.write(log_entry)
                log_file.flush()

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
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        duration_str = str(duration).split('.')[0]
        log_file.write(f"\n=== Session Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (ISO: {end_time.isoformat()}) | Duration: {duration_str} ===\n")
        log_file.flush()
        log_file.close()
        print(f"Session log saved to '{log_path}'.")

if __name__ == "__main__":
    args = parse_args()
    if args.select_model:
        args.model = prompt_model_selection()
    process_footage(args)