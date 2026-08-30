import os
import sys
import shutil
import argparse
import subprocess
import cv2
import imageio_ffmpeg

def parse_args():
    parser = argparse.ArgumentParser(
        description="Safely inspect, remux, and prepare security camera video clips."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="./input_videos",
        help="Path to folder containing raw camera dumps (default: ./input_videos)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./repaired_videos",
        help="Path to folder for repaired/prepared videos (default: ./repaired_videos)"
    )
    parser.add_argument(
        "--archive-dir", "-a",
        default=None,
        help="Optional path to move original source videos to after successful processing"
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        default=False,
        help="Permanently delete original source video after successful processing (default: False)"
    )
    return parser.parse_args()

def is_valid_video(file_path):
    """Verify that a video file can be opened and contains readable frames."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret

def repair_videos(input_dir, output_dir, archive_dir=None, delete_source=False):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)

    video_extensions = (".mp4", ".mkv", ".avi", ".mov")
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]

    if not video_files:
        print(f"No video files found in '{input_dir}'.")
        return

    print(f"Starting repair & preparation for {len(video_files)} video(s)...")
    if delete_source:
        print("  [NOTE] --delete-source is enabled: Originals will be removed after verified processing.")
    elif archive_dir:
        print(f"  [NOTE] Originals will be moved to archive directory: '{archive_dir}'")
    else:
        print("  [NOTE] Safe mode: Original files in input directory will be preserved.")
    print()

    for idx, filename in enumerate(video_files, 1):
        in_file = os.path.join(input_dir, filename)
        out_file = os.path.join(output_dir, filename)
        temp_out_file = os.path.join(output_dir, f"temp_{filename}")

        if not os.path.exists(in_file):
            continue

        in_size = os.path.getsize(in_file)
        print(f"[{idx}/{len(video_files)}] Processing: {filename} ({in_size / (1024*1024):.1f} MB)")

        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)

        # Step 1: Attempt lossless container remux to fix corrupt timestamps/indices
        cmd = [
            ffmpeg_exe, "-y",
            "-err_detect", "ignore_err",
            "-i", in_file,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            temp_out_file
        ]

        repair_successful = False
        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0 and os.path.exists(temp_out_file):
                out_size = os.path.getsize(temp_out_file)
                # Remuxed container size should be close to original and decodable
                if out_size > (in_size * 0.70) and is_valid_video(temp_out_file):
                    repair_successful = True
        except Exception as e:
            print(f"  -> Remux error: {e}")

        # Step 2: Finalize output file
        if repair_successful:
            os.replace(temp_out_file, out_file)
            print("  -> Remux successful and verified.")
        else:
            # Fallback: Copy untouched original if remux fails or produces invalid output
            print("  -> Remux skipped or failed validation; copying original file as fallback...")
            if os.path.exists(temp_out_file):
                os.remove(temp_out_file)
            shutil.copy2(in_file, out_file)

        # Step 3: Handle source file retention / archiving safely
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            if delete_source:
                os.remove(in_file)
                print(f"  -> Original deleted from '{input_dir}'.")
            elif archive_dir:
                shutil.move(in_file, os.path.join(archive_dir, filename))
                print(f"  -> Original moved to archive: '{archive_dir}'.")
            else:
                print("  -> Original preserved in input directory.")
        else:
            print(f"  -> WARNING: Destination verification failed for {filename}. Original untouched.")

    print(f"\nProcessing complete. Ready files are in '{output_dir}'.")

if __name__ == "__main__":
    args = parse_args()
    repair_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        archive_dir=args.archive_dir,
        delete_source=args.delete_source
    )