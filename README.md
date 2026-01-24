# AquaSOT-3D

**End-to-End RGB-Sonar Single Object Tracking for Underwater Environments**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AquaSOT-3D is an end-to-end multi-modal single object tracking framework that fuses RGB camera and Forward-Looking Sonar (FLS) data without requiring extrinsic calibration. The system employs a dual-stream Vision Transformer architecture with Deformable Cross-Modal Attention (DCMA) for calibration-free correspondence learning.

## Architecture

```
RGB Input ──> ViT Backbone ──┐
                             ├──> DCMA Fusion ──> Prediction Heads ──> BBox (RGB, Sonar)
Sonar Input ──> ViT Backbone ─┘
```

**Key Components:**
- Dual ViT-Base backbones for RGB (3-ch) and Sonar (1-ch) feature extraction
- Deformable Cross-Modal Attention for spatial alignment without calibration
- Separate prediction heads for RGB and Sonar modalities

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AquaSOT-3D.git
cd AquaSOT-3D

# Install PyTorch with CUDA
pip3 install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install package
pip install -e .
```

## Dataset

The AquaSOT-3D dataset contains 12 sequences with paired RGB-Sonar frames:

```
data/
├── uuv1/                    # Training
│   ├── light/               # RGB: img/, groundtruth.txt
│   └── sonar/               # Sonar: img_rect/, groundtruth_rect.txt
├── uuv2/ ... uuv12/
```

**Split:** Training (uuv1,2,4,5,7,8,9,11,12) | Validation (uuv3,6,10)

**Preprocessing:** Convert sonar from polar to Cartesian coordinates:
```bash
python scripts/preprocess_sonar.py --all
```

## Training

```bash
# Standard training
python tracking/train.py --script aquasot3d --config vitb_256_dual

# With W&B logging
python tracking/train.py --script aquasot3d --config vitb_256_dual --wandb

# Ablation configurations
python tracking/train.py --script aquasot3d --config ablation_rgb_only
python tracking/train.py --script aquasot3d --config ablation_sonar_only
```

Configuration files are located in `experiments/aquasot3d/`.

## Evaluation

```bash
python tracking/test.py \
    --tracker aquasot3d \
    --checkpoint outputs/train/checkpoints/model.pth \
    --dataset aquasot3d \
    --dataset_path data/
```

**Metrics:** Success (AUC), Precision@20, Center Error (per modality)

## Project Structure

```
AquaSOT-3D/
├── lib/
│   ├── models/aquasot3d/    # Model: backbone, fusion, head
│   ├── train/               # Training: dataset, actor, trainer
│   ├── test/                # Evaluation: tracker, metrics
│   └── config/              # Default configuration
├── tracking/
│   ├── train.py             # Training entry point
│   └── test.py              # Evaluation entry point
├── experiments/aquasot3d/   # YAML configurations
├── scripts/                 # Utility scripts
└── data/                    # Dataset directory
```

## References

- **OSTrack** (ECCV 2022): [Joint Feature Learning and Relation Modeling for Tracking](https://arxiv.org/abs/2203.11991)
- **RGBS50** (2024): [RGB-Sonar Tracking Benchmark](https://arxiv.org/abs/2406.07189)
- **Deformable DETR** (ICLR 2021): [Deformable Transformers for Object Detection](https://arxiv.org/abs/2010.04159)

## Citation

```bibtex
@software{aquasot3d2025,
  title={AquaSOT-3D: End-to-End RGB-Sonar Single Object Tracking},
  author={AquaSOT-3D Team},
  year={2025},
  url={https://github.com/yourusername/AquaSOT-3D}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.