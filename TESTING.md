# Testing Guide

This project includes a comprehensive, automated test suite built with **`pytest`** to verify that changes to the scripts do not introduce regressions or failure points in the video processing pipeline.

---

## Overview of the Testing Framework

The testing suite validates all three stages of the pipeline:
1. **Source Inspection & Remuxing** ([repair_clips.py](repair_clips.py))
2. **YOLO Detection & Timestamp Extraction** ([filter_footage.py](filter_footage.py))
3. **Clip Compacting & Compression** ([compact_video_clips.py](compact_video_clips.py))
4. **End-to-End Integration** (Full multi-stage pipeline flow)

Tests run in isolated temporary directories (`tmp_path`) provided automatically by `pytest`. Test executions will never touch or pollute your actual footage directories (`./input_videos`, `./repaired_videos`, etc.).

---

## Test Fixture Management & Caching

Testing video processing requires realistic video files, but real NVR dumps (512MB+) are too large for rapid testing. The testing suite uses lightweight, targeted video clips (1–4MB each) managed automatically by **[`tests/conftest.py`](tests/conftest.py)**:

* **Download on First Run**: When you run `pytest` for the first time, `conftest.py` downloads the required test videos into `tests/fixtures/`.
* **Local Caching**: On all subsequent runs, the test runner checks `tests/fixtures/` and reuses cached files immediately without network activity.
* **Git Excluded**: The `tests/fixtures/` folder is listed in `.gitignore` so no large binary video files are ever tracked or committed to Git.

### Fixtures Included

| Fixture File | Source / Type | Purpose |
|---|---|---|
| `test_day_person.mp4` | Intel IoT DevKit sample video (~3.3MB) | Tests daytime person detection and clip trimming. |
| `test_ir_night.mp4` | Generated monochrome IR video (~3.3MB) | Tests night-vision/infrared person detection. |
| `test_empty_scene.mp4` | Synthetically generated static video (~3s) | Negative control test verifying that scenes with no people produce zero false detections. |
| `test_corrupt.mp4` | Truncated binary header | Verifies error handling and fallback behavior in `repair_clips.py`. |

---

## Test Suite Structure

```text
search-video-for-people/
├── tests/
│   ├── conftest.py           # Session fixtures & automatic download/caching logic
│   ├── fixtures/             # Auto-populated test video directory (git-ignored)
│   ├── test_repair.py        # Tests for repair_clips.py (remuxing, validation, error handling)
│   ├── test_filter.py        # Tests for filter_footage.py (YOLO day/IR detections, manifest, logs)
│   ├── test_compact.py       # Tests for compact_video_clips.py (downscaling, fps, CRF)
│   └── test_pipeline_e2e.py  # Full sequential integration pipeline test
```

### Module Breakdown

- **`tests/test_repair.py`**:
  - `test_is_valid_video_*`: Validates video decodability checks for valid, missing, and corrupt files.
  - `test_repair_videos_standard`: Validates lossless remuxing and output verification.
  - `test_repair_videos_with_archive_dir`: Tests moving source files to an archive directory.
  - `test_repair_videos_with_delete_source`: Tests permanent source deletion upon verified remuxing.

- **`tests/test_filter.py`**:
  - `test_format_time_hms`: Validates timestamp conversion to `HH:MM:SS.ss`.
  - `test_build_camera_manifest`: Validates camera GUID detection and `[CAM-XX]` prefix alias generation.
  - `test_merge_intervals`: Validates padding and overlapping interval merges.
  - `test_filter_footage_day_detection`: Validates person detection and session log creation on daytime footage.
  - `test_filter_footage_ir_detection`: Validates person detection on IR night footage.
  - `test_filter_footage_empty_negative_control`: Validates that empty footage produces 0 extracted clips.
  - `test_filter_footage_compact_mode`: Validates single-pass downscaling (`--compact`) during extraction.

- **`tests/test_compact.py`**:
  - `test_compact_videos_standard`: Validates downscaling (e.g. `--max-width 640`), framerate reduction (`--fps 15`), and H.264 CRF encoding.
  - `test_compact_videos_empty_dir`: Validates graceful handling when no input videos exist.

- **`tests/test_pipeline_e2e.py`**:
  - `test_full_pipeline_integration`: Full end-to-end regression test running files through repair -> filter -> compact in sequence.

---

## Running the Tests

Ensure your environment dependencies are installed via `uv sync`.

### Run All Tests
```powershell
uv run pytest -v
```

### Run a Specific Test Module
```powershell
# Test only clip repair
uv run pytest tests/test_repair.py -v

# Test only YOLO detection and filtering
uv run pytest tests/test_filter.py -v

# Test only clip compacting
uv run pytest tests/test_compact.py -v

# Run only the end-to-end integration test
uv run pytest tests/test_pipeline_e2e.py -v
```

### Run a Specific Test by Name Pattern
```powershell
uv run pytest -k "test_filter_footage_day_detection" -v
```

### Run with Short Summary Output
```powershell
uv run pytest
```
