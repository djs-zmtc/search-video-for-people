# Scripts to Process Security Video

A pipeline to prepare, scan, and extract lightweight video clips of detected people from security camera NVR dumps using YOLO and FFmpeg.

---

## Installation & Setup

For full platform-specific setup instructions (Windows PowerShell and Linux bash), see **[INSTALL.md](INSTALL.md)**.

### Quick Setup:
```bash
# Install dependencies with uv (automatically configures PyTorch + CUDA)
uv sync
```

---

## Quick Start / Run Order

### Step 1: Prepare & Verify Source Videos
Inspect and remux raw camera video dumps to fix corrupt container timestamps/headers without deleting originals:
```bash
uv run python repair_clips.py
```
* **Input**: `./input_videos`
* **Output**: `./repaired_videos`
* *(Optional)* To automatically move original files to an archive folder after verification:
  ```bash
  uv run python repair_clips.py --archive-dir ./archive_videos
  ```

---

### Step 2: Scan for People & Extract Clips
Scan prepared footage with YOLO and extract frame-accurate, re-encoded video clips for all detected events:

#### Standard Extraction:
```bash
uv run python filter_footage.py
```
* **Input**: `./repaired_videos`
* **Output**: `./person_clips`

#### One-Pass Compact Mode (Recommended):
Scan and extract directly into compressed, downscaled clips in a single pass without extra disk I/O:
```bash
uv run python filter_footage.py --compact --output-dir ./person_clips_720p
```

---

## Standalone Utilities

### Compact Existing Clips
If you already have extracted clips in `./person_clips` and want to compress/downscale them separately:
```bash
uv run python compact_video_clips.py --input-dir ./person_clips --output-dir ./person_clips_720p
```

---

## CLI Options Reference

### `repair_clips.py`
| Argument | Default | Description |
|---|---|---|
| `-i`, `--input-dir` | `./input_videos` | Source directory with raw camera dumps |
| `-o`, `--output-dir` | `./repaired_videos` | Destination directory for repaired videos |
| `-a`, `--archive-dir`| `None` | Optional directory to move originals to after validation |
| `--delete-source` | `False` | Permanently delete source files after validation |

### `filter_footage.py`
| Argument | Default | Description |
|---|---|---|
| `-i`, `--input-dir` | `./repaired_videos` | Directory of prepared videos to scan |
| `-o`, `--output-dir`| `./person_clips` | Directory to save extracted clips |
| `-m`, `--model` | `yolo11x.pt` | YOLO model weights (`yolo11x.pt`, `yolo11m.pt`, etc.) |
| `-c`, `--conf` | `0.50` | Detection confidence threshold |
| `-s`, `--stride` | `5` | Scan every Nth frame |
| `--consecutive` | `2` | Consecutive positive samples required to confirm detection |
| `-p`, `--pad` | `15.0` | Seconds padded before and after detected events |
| `--compact` | `False` | Enable direct downscaling and framerate optimization |
| `--max-width` | `1280` | Max width for downscaling (preserves aspect ratio) |
| `--fps` | `15` | Framerate for compact mode |
| `--crf` | `22` | H.264 CRF quality level (lower = higher quality) |
| `--device` | `auto` | Compute device (`0`, `mps`, `cpu`) |
