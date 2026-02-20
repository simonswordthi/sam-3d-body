# SAM 3D Body – Integration

This project integrates Meta's **SAM 3D Body** model
([facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body))
with a Python desktop application for single-image full-body 3D human mesh
recovery.

## Background

SAM 3D Body (3DB) is a promptable model that reconstructs a full-body 3D human
mesh from a single image.  It uses a DINOv3 encoder-decoder architecture and
outputs a **Momentum Human Rig (MHR)** mesh covering body, feet, and hands.
Auxiliary prompts (2D keypoints, segmentation masks) can be supplied to guide
inference, similar to the SAM family of models.

- Paper: <https://arxiv.org/abs/2602.15989>
- Blog post: <https://ai.meta.com/blog/sam-3d/>
- Live demo: <https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d>
- Rerun visualisation playground: <https://github.com/rerun-io/sam3d-body-rerun>

## Installation

### 1. Clone and set up the upstream SAM 3D Body package

```bash
git clone https://github.com/facebookresearch/sam-3d-body
cd sam-3d-body
pip install -e .
```

Follow [INSTALL.md](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md)
to request access to the gated Hugging Face checkpoints.

### 2. Install this project's dependencies

```bash
cd /path/to/this/repo
pip install -r requirements.txt
```

Alternatively, use the provided helper scripts:

```bash
# Linux / macOS
bash setup_venv.sh

# Windows
setup_venv.bat
```

### 3. Download the model checkpoint

```bash
huggingface-cli download facebook/sam-3d-body-dinov3 \
    --local-dir checkpoints/sam-3d-body-dinov3
```

## Usage

### Python API

```python
import cv2
from src.model.sam3d_processor import SAM3DProcessor

processor = SAM3DProcessor.from_pretrained(
    hf_repo_id="facebook/sam-3d-body-dinov3"
)

img_bgr = cv2.imread("path/to/image.jpg")
outputs = processor.process_frame(img_bgr)

vertices = outputs["vertices"]   # Vx3 float32 – MHR mesh vertices
faces    = outputs["faces"]      # Fx3 int32   – triangle faces
joints   = outputs["joints"]     # Jx3 float32 – 3-D joint positions
```

### Desktop Application

```bash
python src/main.py
```

### 3D Viewer

```python
from src.visualization.body_viewer import BodyMeshViewer

viewer = BodyMeshViewer(vertices=outputs["vertices"], faces=outputs["faces"])
viewer.visualize()
```

## Project Structure

```
src/
├── main.py                    # Application entry point
├── model/
│   └── sam3d_processor.py     # SAM 3D Body processor (wraps Meta's estimator)
├── ui/
│   └── app.py                 # SAM3DApp – PyQt6 main window
└── visualization/
    └── body_viewer.py         # BodyMeshViewer / BodyPointCloudViewer (Open3D)
```
