import os
import shutil
import argparse
import cv2
import pytest
from compact_video_clips import compact_videos

def test_compact_videos_standard(tmp_path, day_video_path):
    input_dir = tmp_path / "clips"
    output_dir = tmp_path / "compacted"
    input_dir.mkdir()

    shutil.copy2(day_video_path, input_dir / "sample_clip.mp4")

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        max_width=640,
        fps=15,
        crf=23,
        preset="fast"
    )

    compact_videos(args)

    assert os.path.exists(output_dir / "sample_clip.mp4")
    out_path = str(output_dir / "sample_clip.mp4")

    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    assert width <= 640
    assert abs(fps - 15) <= 1.0

def test_compact_videos_empty_dir(tmp_path):
    input_dir = tmp_path / "empty_in"
    output_dir = tmp_path / "empty_out"
    input_dir.mkdir()

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        max_width=640,
        fps=15,
        crf=23,
        preset="fast"
    )

    # Should exit cleanly without throwing an exception
    compact_videos(args)
    assert os.path.exists(output_dir)
