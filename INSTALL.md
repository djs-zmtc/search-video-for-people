# Installation & Environment Setup Guide

This guide walks you through configuring the Python and YOLO environment from scratch after cloning this repository to a new machine.

---

## Prerequisites & Requirements

* **Operating System**: Windows 10/11 or Linux (Ubuntu 20.04+, Debian 11+, Fedora 38+, etc.)
* **Python**: Version `3.12` or newer (automatically installed and managed if using `uv`)
* **Hardware Acceleration (Optional, but recommended)**:
  * **NVIDIA GPU** with CUDA support and up-to-date NVIDIA drivers.
  * *CPU mode is fully supported as an automatic fallback on all platforms.*
* **Package Manager**: [`uv`](https://docs.astral.sh/uv/) is the recommended package and project manager for fast, deterministic setup.

---

## Method 1: Quick Setup using `uv` (Recommended)

### 🪟 Windows (PowerShell)

1. **Install `uv`** (if not already installed):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   *Or via Windows Package Manager:*
   ```powershell
   winget install --id=astral-sh.uv -e
   ```

2. **Clone the repository and enter the directory**:
   ```powershell
   git clone <REPO_URL>
   cd search-video-for-people
   ```

3. **Install all dependencies and create the virtual environment**:
   ```powershell
   uv sync
   ```
   *`uv` will automatically fetch Python 3.12+ (if needed), install PyTorch with CUDA 12.4 support, and install all dependencies in `.venv`.*

4. **Verify GPU / CUDA acceleration**:
   ```powershell
   uv run python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

---

### 🐧 Linux (Bash)

1. **Install system dependencies** (required by OpenCV for headless video decoding):
   * **Ubuntu / Debian**:
     ```bash
     sudo apt update
     sudo apt install -y libgl1 libglib2.0-0 ffmpeg git curl
     ```
   * **Fedora / RHEL**:
     ```bash
     sudo dnf install -y mesa-libGL glib2 ffmpeg git curl
     ```
   * **Arch Linux**:
     ```bash
     sudo pacman -Syu --noconfirm mesa glib2 ffmpeg git curl
     ```

2. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

3. **Clone the repository and enter the directory**:
   ```bash
   git clone <REPO_URL>
   cd search-video-for-people
   ```

4. **Install all dependencies and create the virtual environment**:
   ```bash
   uv sync
   ```

5. **Verify GPU / CUDA acceleration**:
   ```bash
   uv run python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

---

## Method 2: Traditional Setup using standard `python -m venv` and `pip`

If you prefer not to use `uv`, you can configure the environment using standard Python tools:

### 🪟 Windows (PowerShell)

```powershell
# 1. Create and activate a virtual environment (Python 3.12+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install ultralytics opencv-python imageio-ffmpeg
```

### 🐧 Linux (Bash)

```bash
# 1. Install system prerequisites (Debian/Ubuntu example)
sudo apt update && sudo apt install -y python3-venv libgl1 libglib2.0-0 ffmpeg

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining dependencies
pip install ultralytics opencv-python imageio-ffmpeg
```

---

## Model Weights & First Run

The YOLO model weights (e.g. `yolo11x.pt` or `yolo11m.pt`) are automatically downloaded from Ultralytics assets the first time you run [`filter_footage.py`](file:///c:/Users/dougl/Projects/search-video-for-people/filter_footage.py).

If you are deploying to an offline machine or wish to pre-fetch the weights:
```bash
# Windows / Linux with uv:
uv run python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

---

## Running the Pipeline

Once installed, place your raw security footage into `./input_videos/` and execute:

```bash
# Step 1: Safe container remux / preparation
uv run python repair_clips.py

# Step 2: Detection and compact clip extraction
uv run python filter_footage.py --compact --output-dir ./person_clips_720p
```
