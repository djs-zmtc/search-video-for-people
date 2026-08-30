import os
import argparse
import subprocess
import imageio_ffmpeg

def parse_args():
    parser = argparse.ArgumentParser(
        description="Downscale and compress video clips for lightweight storage and fast review."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="./person_clips",
        help="Input folder containing video clips (default: ./person_clips)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./person_clips_720p",
        help="Output folder for compressed video clips (default: ./person_clips_720p)"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Maximum width in pixels while maintaining original aspect ratio (default: 1280)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target framerate (default: 15)"
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 CRF quality level (default: 23)"
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="FFmpeg x264 encoding preset (default: medium)"
    )
    return parser.parse_args()

def compact_videos(args):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_dir):
        print(f"Input directory '{args.input_dir}' not found.")
        return

    video_extensions = (".mp4", ".mkv", ".avi", ".mov")
    video_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(video_extensions) and os.path.isfile(os.path.join(args.input_dir, f))
    ]

    if not video_files:
        print(f"No video clips found in '{args.input_dir}'.")
        return

    print(f"Found {len(video_files)} clip(s) to convert in '{args.input_dir}'.\n")

    for idx, filename in enumerate(sorted(video_files), 1):
        in_path = os.path.join(args.input_dir, filename)
        name_stem, _ = os.path.splitext(filename)
        out_filename = f"{name_stem}.mp4"
        out_path = os.path.join(args.output_dir, out_filename)

        in_size_mb = os.path.getsize(in_path) / (1024 * 1024)
        print(f"[{idx}/{len(video_files)}] Converting: {filename} ({in_size_mb:.1f} MB)...")

        # FFmpeg command:
        # - scale='min(max_w,iw)':-2 preserves exact aspect ratio without stretching or upscaling
        # - fps: downsamples framerate
        # - c:a aac: ensures universal compatibility even if input audio is PCM/G.711
        cmd = [
            ffmpeg_exe, "-y",
            "-err_detect", "ignore_err",
            "-i", in_path,
            "-vf", f"scale='min({args.max_width},iw)':-2,fps={args.fps}",
            "-c:v", "libx264",
            "-crf", str(args.crf),
            "-preset", args.preset,
            "-c:a", "aac",
            "-b:a", "128k",
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

    print(f"\nAll conversions finished. Scaled files are saved in '{args.output_dir}'.")
    print(f"Original files remain intact in '{args.input_dir}'.")

if __name__ == "__main__":
    args = parse_args()
    compact_videos(args)