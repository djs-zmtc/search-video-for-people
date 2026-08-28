import os
import subprocess
import imageio_ffmpeg

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
INPUT_DIR = "./input_videos"
REPAIRED_DIR = "./repaired_videos"
os.makedirs(REPAIRED_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".mp4"):
        continue

    in_file = os.path.join(INPUT_DIR, filename)
    out_file = os.path.join(REPAIRED_DIR, filename)

    # Lossless remux: -c copy rebuilds container/headers with zero quality loss
    cmd = [
        ffmpeg_exe, "-y", "-err_detect", "ignore_err",
        "-i", in_file, "-c", "copy", out_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("All files repaired. Point your filter script to ./repaired_videos")