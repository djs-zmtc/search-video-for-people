import os
import shutil
import argparse
import pytest
import cv2
from filter_footage import (
    format_time_hms,
    build_camera_manifest,
    merge_intervals,
    process_footage,
)

def test_format_time_hms():
    assert format_time_hms(0) == "00:00:00.00"
    assert format_time_hms(65.5) == "00:01:05.50"
    assert format_time_hms(3661.25) == "01:01:01.25"

def test_build_camera_manifest():
    files = [
        "FrontDoor_{12345678-1234-1234-1234-123456789abc}_20260830.mp4",
        "FrontDoor_{12345678-1234-1234-1234-123456789abc}_20260831.mp4",
        "BackYard_{87654321-4321-4321-4321-cba987654321}_20260830.mp4",
        "simple_unformatted_name.mp4"
    ]
    file_to_display, manifest_lines = build_camera_manifest(files)
    
    assert len(manifest_lines) > 0
    assert "[CAM-01]" in file_to_display[files[0]]
    assert "[CAM-01]" in file_to_display[files[1]]
    assert "[CAM-02]" in file_to_display[files[2]]
    assert file_to_display[files[3]] == "simple_unformatted_name.mp4"

def test_merge_intervals():
    # Two separate intervals that do not overlap with 2.0s padding
    intervals = [(10.0, 12.0), (30.0, 32.0)]
    merged = merge_intervals(intervals, max_duration=60.0, pad_seconds=2.0)
    assert len(merged) == 2
    assert merged[0] == (8.0, 14.0)
    assert merged[1] == (28.0, 34.0)

    # Overlapping intervals should merge
    overlapping = [(10.0, 15.0), (14.0, 20.0)]
    merged_ov = merge_intervals(overlapping, max_duration=60.0, pad_seconds=1.0)
    assert len(merged_ov) == 1
    assert merged_ov[0] == (9.0, 21.0)

def test_filter_footage_day_detection(tmp_path, day_video_path, yolo_test_model):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "clips"
    log_dir = tmp_path / "log"
    input_dir.mkdir()

    shutil.copy2(day_video_path, input_dir / "FrontDoor_{11111111-2222-3333-4444-555555555555}_day.mp4")

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
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

    process_footage(args)

    extracted_clips = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
    assert len(extracted_clips) > 0, "Expected at least one extracted person clip from daytime fixture"

    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    assert len(log_files) == 1
    with open(log_dir / log_files[0], "r", encoding="utf-8") as lf:
        content = lf.read()
        assert "[CAM-01]" in content
        assert "Detected:" in content

def test_filter_footage_ir_detection(tmp_path, ir_video_path, yolo_test_model):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "clips"
    log_dir = tmp_path / "log"
    input_dir.mkdir()

    shutil.copy2(ir_video_path, input_dir / "Driveway_{99999999-8888-7777-6666-555555555555}_ir.mp4")

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
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

    process_footage(args)

    extracted_clips = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
    assert len(extracted_clips) > 0, "Expected at least one extracted clip from IR fixture"

def test_filter_footage_empty_negative_control(tmp_path, empty_video_path, yolo_test_model):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "clips"
    log_dir = tmp_path / "log"
    input_dir.mkdir()

    shutil.copy2(empty_video_path, input_dir / "empty.mp4")

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        log_dir=str(log_dir),
        model=yolo_test_model,
        conf=0.50,
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

    process_footage(args)

    extracted_clips = [f for f in os.listdir(output_dir) if f.endswith(".mp4")] if os.path.exists(output_dir) else []
    assert len(extracted_clips) == 0, "Negative control should not produce extracted person clips"

def test_filter_footage_compact_mode(tmp_path, day_video_path, yolo_test_model):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "clips_compact"
    log_dir = tmp_path / "log"
    input_dir.mkdir()

    shutil.copy2(day_video_path, input_dir / "day_compact.mp4")

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        log_dir=str(log_dir),
        model=yolo_test_model,
        conf=0.35,
        stride=5,
        consecutive=2,
        pad=2.0,
        compact=True,
        max_width=640,
        fps=15,
        crf=24,
        device=None,
        select_model=False,
    )

    process_footage(args)

    extracted_clips = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
    assert len(extracted_clips) > 0

    clip_path = str(output_dir / extracted_clips[0])
    cap = cv2.VideoCapture(clip_path)
    assert cap.isOpened()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    assert width <= 640
    assert abs(fps - 15) <= 1.0
