import os
import sys
import numpy as np
import cv2
import pytest
import urllib.request

FIXTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures"))

REMOTE_FIXTURES = {
    "test_day_person.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4",
}

def download_file(url, dest_path):
    """Download a file with User-Agent header and progress indicator."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    print(f"\n[FIXTURE] Downloading {os.path.basename(dest_path)} from {url}...")
    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
        chunk_size = 1024 * 64
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[FIXTURE] Saved {os.path.basename(dest_path)} ({size_mb:.2f} MB)")

def create_ir_night_fixture(source_video_path, dest_path):
    """
    Generate a realistic IR / night-vision monochrome video from source footage.
    Applies grayscale luminance mapping and night contrast curve.
    """
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_adjusted = cv2.equalizeHist(gray)
        ir_bgr = cv2.cvtColor(ir_adjusted, cv2.COLOR_GRAY2BGR)
        out.write(ir_bgr)

    cap.release()
    out.release()

def create_synthetic_empty_video(dest_path, width=640, height=360, fps=25, duration_sec=3):
    """Generate a clean synthetic empty video (no humans) for negative testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
    total_frames = fps * duration_sec
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [40 + int(i * 0.1) % 20, 45, 50]
        cv2.rectangle(frame, (100, 200), (540, 320), (70, 75, 80), -1)
        cv2.line(frame, (0, 250), (640, 250), (30, 30, 30), 2)
        out.write(frame)
    out.release()

def create_corrupt_video(dest_path):
    """Create a corrupt truncated file for error-handling verification."""
    with open(dest_path, "wb") as f:
        f.write(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08freeCORRUPT_DATA_HEADER_TRUNCATED")

def ensure_test_fixtures():
    """Ensure all required test videos are downloaded, generated, and cached."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    # 1. Download remote daytime sample video if missing
    for filename, url in REMOTE_FIXTURES.items():
        dest = os.path.join(FIXTURES_DIR, filename)
        if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
            download_file(url, dest)

    # 2. Generate local IR night vision fixture from source if missing
    day_source = os.path.join(FIXTURES_DIR, "test_day_person.mp4")
    ir_dest = os.path.join(FIXTURES_DIR, "test_ir_night.mp4")
    if os.path.exists(day_source) and (not os.path.exists(ir_dest) or os.path.getsize(ir_dest) == 0):
        create_ir_night_fixture(day_source, ir_dest)

    # 3. Generate local synthetic empty scene if missing
    empty_dest = os.path.join(FIXTURES_DIR, "test_empty_scene.mp4")
    if not os.path.exists(empty_dest) or os.path.getsize(empty_dest) == 0:
        create_synthetic_empty_video(empty_dest)

    # 4. Generate corrupt test file if missing
    corrupt_dest = os.path.join(FIXTURES_DIR, "test_corrupt.mp4")
    if not os.path.exists(corrupt_dest) or os.path.getsize(corrupt_dest) == 0:
        create_corrupt_video(corrupt_dest)

    return FIXTURES_DIR

@pytest.fixture(scope="session")
def fixtures_dir():
    """Session-scoped fixture to ensure all fixtures exist and return directory path."""
    return ensure_test_fixtures()

@pytest.fixture(scope="session")
def day_video_path(fixtures_dir):
    return os.path.join(fixtures_dir, "test_day_person.mp4")

@pytest.fixture(scope="session")
def ir_video_path(fixtures_dir):
    return os.path.join(fixtures_dir, "test_ir_night.mp4")

@pytest.fixture(scope="session")
def empty_video_path(fixtures_dir):
    return os.path.join(fixtures_dir, "test_empty_scene.mp4")

@pytest.fixture(scope="session")
def corrupt_video_path(fixtures_dir):
    return os.path.join(fixtures_dir, "test_corrupt.mp4")

@pytest.fixture(scope="session")
def yolo_test_model():
    """Returns the most lightweight available YOLO model for rapid testing."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for model_name in ["yolo11s.pt", "yolo11m.pt", "yolo11x.pt"]:
        model_path = os.path.join(root_dir, model_name)
        if os.path.exists(model_path):
            return model_name
    return "yolo11s.pt"
