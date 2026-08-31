import os
import shutil
import argparse
import pytest
import cv2
from repair_clips import repair_videos
from filter_footage import process_footage
from compact_video_clips import compact_videos

def test_full_pipeline_integration(tmp_path, day_video_path, ir_video_path, yolo_test_model):
    """
    End-to-end integration test validating the entire video processing chain:
    1. Raw NVR dump inspection and container remuxing (repair_clips)
    2. YOLO person scanning and timestamp-accurate extraction (filter_footage)
    3. Standalone downscaling and compression (compact_video_clips)
    """
    input_dir = tmp_path / "input_videos"
    repaired_dir = tmp_path / "repaired_videos"
    clips_dir = tmp_path / "person_clips"
    compact_dir = tmp_path / "person_clips_720p"
    log_dir = tmp_path / "log"

    input_dir.mkdir()

    # Place day and IR test files with realistic NVR-style names
    day_file = "FrontGate_{a1b2c3d4-e5f6-7890-abcd-ef0123456789}_20260831.mp4"
    ir_file = "BackYard_{b2c3d4e5-f6a7-8901-bcde-f0123456789a}_20260831.mp4"
    shutil.copy2(day_video_path, input_dir / day_file)
    shutil.copy2(ir_video_path, input_dir / ir_file)

    # -------------------------------------------------------------
    # Step 1: Repair / Prepare videos
    # -------------------------------------------------------------
    repair_videos(str(input_dir), str(repaired_dir))

    repaired_files = sorted([f for f in os.listdir(repaired_dir) if f.endswith(".mp4")])
    assert len(repaired_files) == 2
    for rf in repaired_files:
        cap = cv2.VideoCapture(str(repaired_dir / rf))
        assert cap.isOpened(), f"Repaired file {rf} is not decodable"
        cap.release()

    # -------------------------------------------------------------
    # Step 2: Scan for people and extract clips
    # -------------------------------------------------------------
    filter_args = argparse.Namespace(
        input_dir=str(repaired_dir),
        output_dir=str(clips_dir),
        log_dir=str(log_dir),
        model=yolo_test_model,
        conf=0.35,
        stride=5,
        consecutive=2,
        pad=2.0,
        compact=False,
        max_width=1280,
        fps=15,
        crf=22,
        device=None,
        select_model=False,
    )
    process_footage(filter_args)

    extracted_clips = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
    assert len(extracted_clips) > 0, "Pipeline should have extracted at least 1 person clip"

    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    assert len(log_files) == 1
    with open(log_dir / log_files[0], "r", encoding="utf-8") as lf:
        log_text = lf.read()
        assert "[CAM-01]" in log_text
        assert "[CAM-02]" in log_text
        assert "=== Session Ended:" in log_text

    # -------------------------------------------------------------
    # Step 3: Compact and downscale extracted clips
    # -------------------------------------------------------------
    compact_args = argparse.Namespace(
        input_dir=str(clips_dir),
        output_dir=str(compact_dir),
        max_width=640,
        fps=15,
        crf=24,
        preset="fast"
    )
    compact_videos(compact_args)

    compacted_files = [f for f in os.listdir(compact_dir) if f.endswith(".mp4")]
    assert len(compacted_files) == len(extracted_clips), "Every extracted clip should be compacted"

    for cf in compacted_files:
        cap = cv2.VideoCapture(str(compact_dir / cf))
        assert cap.isOpened()
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert w <= 640
        assert abs(fps - 15) <= 1.0
