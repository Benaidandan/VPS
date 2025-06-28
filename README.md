# VPS: Visual Positioning System

A coarse-to-fine visual positioning system that combines the power of Hierarchical-Localization (Hloc) for visual place recognition and multiple pose estimation methods (VGGT, MASt3R) for accurate pose estimation.

## Features

- Efficient visual place recognition using Hloc's NetVLAD
- Multiple pose estimation methods:
  - VGGT for accurate pose estimation
  - MASt3R for robust pose estimation
- Scale recovery using ground truth depth information
- Easy-to-use pipeline for visual localization
- Configurable pose estimation methods

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VPS.git
cd VPS
```

2. Create a conda environment:
```bash
conda create -n vps python=3.10
conda activate vps
```

3. Install dependencies:
```bash
pip install -e .
```

4. (Optional) For MASt3R support, ensure the MASt3R model is available in `third_party/mast3r/checkpoints/`

## Usage

### Basic Usage

```python
from vps.core import VisualPositioningSystem

# Initialize the system with VGGT (default)
vps = VisualPositioningSystem(config_path='configs/default.yaml')

# Perform localization
pose = vps.localize(query_image='path/to/query.jpg')
```

### Using MASt3R for Pose Estimation

```python
from vps.core import VisualPositioningSystem

# Initialize the system with MASt3R
vps = VisualPositioningSystem(config_path='configs/mast3r.yaml')

# Perform localization
pose = vps.localize(query_image='path/to/query.jpg')
```

### Configuration

You can switch between pose estimation methods by modifying the configuration file:

```yaml
# For VGGT
pose:
  method: "vggt"
  vggt:
    model_path: "checkpoints/model.pt"
    image_size: 1024
    ref_dir: "data/ref"
    results_dir: "data/outputs/poses"

# For MASt3R
pose:
  method: "mast3r"
  mast3r:
    model_path: "third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    image_size: 512
    batch_size: 1
    ref_dir: "data/ref"
    results_dir: "data/outputs/poses_mast3r"
```

## Testing

Test MASt3R integration:
```bash
python scripts/test_mast3r.py
```

## Project Structure

```
VPS/
├── vps/                    # Main source code
│   ├── core/              # Core modules
│   │   ├── pose.py        # VGGT pose estimator
│   │   ├── pose_mast3r.py # MASt3R pose estimator
│   │   ├── vpr.py         # Visual place recognition
│   │   └── __init__.py    # Main VPS class
├── configs/                # Configuration files
│   ├── default.yaml       # Default config (VGGT)
│   └── mast3r.yaml        # MASt3R config
├── scripts/                # Training and evaluation scripts
│   └── test_mast3r.py     # MASt3R test script
├── data/                   # Dataset directory
├── third_party/           # Third-party models
│   └── mast3r/            # MASt3R model files
└── tests/                  # Test code
```

## Dependencies

- PyTorch >= 2.3.1
- Hloc (from [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization))
- VGGT (from [VGGT](https://github.com/facebookresearch/vggt))
- MASt3R (from [MASt3R](https://github.com/naver/MASt3R))
- OpenCV
- NumPy
- PyColmap

## License

MIT License 