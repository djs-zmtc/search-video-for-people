import os
import subprocess
import imageio_ffmpeg

# Resolve bundled FFmpeg executable path
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

INPUT_DIR = "./person_clips"
OUTPUT_DIR = "./person_clips_720p"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 15
CRF_VALUE = "23"        # Balanced visual quality & compression for security review
ENCODER_PRESET = "medium"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scan for video clips
video_extensions = (".mp4", ".mkv", ".avi", ".mov")
video_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(video_extensions) and os.path.isfile(os.path.join(INPUT_DIR, f))
]

print(f"Found {len(video_files)} clip(s) to convert in '{INPUT_DIR}'.\n")

for idx, filename in enumerate(sorted(video_files), 1):
    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    in_size_mb = os.path.getsize(in_path) / (1024 * 1024)
    print(f"[{idx}/{len(video_files)}] Converting: {filename} ({in_size_mb:.1f} MB)...")

    # FFmpeg command:
    # - scale: ensures 1280x720 aspect compliance
    # - fps: downsamples framerate to 15 fps
    # - c:a copy: keeps original audio track untouched (if present)
    cmd = [
        ffmpeg_exe, "-y",
        "-err_detect", "ignore_err",
        "-i", in_path,
        "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT},fps={TARGET_FPS}",
        "-c:v", "libx264",
        "-crf", CRF_VALUE,
        "-preset", ENCODER_PRESET,
        "-c:a", "copy",
        "-movflags", "+faststart",
        out_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode == 0 and os.path.exists(out_path):
        out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        reduction = (1 - (out_size_mb / in_size_mb)) * 100 if in_size_mb > 0 else 0
        print(f"  -> Finished: {out_size_mb:.1f} MB ({reduction:.1f}% reduction)")
    else:
        print(f"  -> ERROR: Failed to encode {filename}")

print(f"\nAll conversions finished. Scaled files are saved in '{OUTPUT_DIR}'.")
print(f"Original files remain intact in '{INPUT_DIR}'.")