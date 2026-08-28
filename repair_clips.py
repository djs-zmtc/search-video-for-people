import os
import shutil
import subprocess
import imageio_ffmpeg

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
INPUT_DIR = "./input_videos"
REPAIRED_DIR = "./repaired_videos"
os.makedirs(REPAIRED_DIR, exist_ok=True)

video_extensions = (".mp4", ".mkv", ".avi", ".mov")
video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(video_extensions)]

print(f"Starting safe repair & migration for {len(video_files)} video(s)...\n")

for idx, filename in enumerate(video_files, 1):
    in_file = os.path.join(INPUT_DIR, filename)
    out_file = os.path.join(REPAIRED_DIR, filename)
    temp_out_file = os.path.join(REPAIRED_DIR, f"temp_{filename}")

    # Safety check: Skip if input file is missing/inaccessible
    if not os.path.exists(in_file):
        continue

    in_size = os.path.getsize(in_file)
    print(f"[{idx}/{len(video_files)}] Processing: {filename} ({in_size / (1024*1024):.1f} MB)")

    # Clean up any leftover temp files from prior interrupted runs
    if os.path.exists(temp_out_file):
        os.remove(temp_out_file)

    # Step 1: Attempt lossless container remux to temp file
    cmd = [
        ffmpeg_exe, "-y",
        "-err_detect", "ignore_err",
        "-i", in_file,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        temp_out_file
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    repair_successful = False

    # Step 2: Validate remux success
    if result.returncode == 0 and os.path.exists(temp_out_file):
        out_size = os.path.getsize(temp_out_file)
        # Verify output exists and retains a reasonable portion of the data (> 50% threshold)
        if out_size > 0 and out_size > (in_size * 0.5):
            repair_successful = True

    # Step 3: Finalize destination file
    if repair_successful:
        os.replace(temp_out_file, out_file)
        print("  -> Remux successful.")
    else:
        # Fallback: If remux failed, copy the original untouched file directly
        print("  -> Remux skipped or failed; performing standard copy as fallback...")
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)
        shutil.copy2(in_file, out_file)

    # Step 4: Final verification before deleting source
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        os.remove(in_file)
        print(f"  -> Safe: Original removed from {INPUT_DIR} to reclaim disk space.")
    else:
        print(f"  -> CRITICAL WARNING: Verification failed for {filename}. Original retained in {INPUT_DIR}.")

print("\nProcessing complete. All active files are now in ./repaired_videos")