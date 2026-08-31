import os
import shutil
import cv2
import pytest
from repair_clips import is_valid_video, repair_videos

def test_is_valid_video_with_valid_fixture(day_video_path):
    assert is_valid_video(day_video_path) is True

def test_is_valid_video_with_nonexistent_file(tmp_path):
    non_existent = str(tmp_path / "non_existent.mp4")
    assert is_valid_video(non_existent) is False

def test_is_valid_video_with_corrupt_file(corrupt_video_path):
    assert is_valid_video(corrupt_video_path) is False

def test_repair_videos_standard(tmp_path, day_video_path, ir_video_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "repaired"
    input_dir.mkdir()

    shutil.copy2(day_video_path, input_dir / "day.mp4")
    shutil.copy2(ir_video_path, input_dir / "ir.mp4")

    repair_videos(str(input_dir), str(output_dir))

    repaired_files = os.listdir(output_dir)
    assert "day.mp4" in repaired_files
    assert "ir.mp4" in repaired_files

    for f in repaired_files:
        path = str(output_dir / f)
        assert is_valid_video(path) is True
        assert os.path.getsize(path) > 0

    # Ensure source files were preserved by default
    assert os.path.exists(input_dir / "day.mp4")
    assert os.path.exists(input_dir / "ir.mp4")

def test_repair_videos_with_archive_dir(tmp_path, day_video_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "repaired"
    archive_dir = tmp_path / "archive"
    input_dir.mkdir()

    shutil.copy2(day_video_path, input_dir / "day.mp4")

    repair_videos(str(input_dir), str(output_dir), archive_dir=str(archive_dir))

    assert os.path.exists(output_dir / "day.mp4")
    assert not os.path.exists(input_dir / "day.mp4")
    assert os.path.exists(archive_dir / "day.mp4")

def test_repair_videos_with_delete_source(tmp_path, day_video_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "repaired"
    input_dir.mkdir()

    shutil.copy2(day_video_path, input_dir / "day.mp4")

    repair_videos(str(input_dir), str(output_dir), delete_source=True)

    assert os.path.exists(output_dir / "day.mp4")
    assert not os.path.exists(input_dir / "day.mp4")
