# AquaSOT-3D

**End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation**

A unified end-to-end multi-modal single-object tracking system that fuses RGB camera and Forward-Looking Sonar (FLS) data for underwater object tracking, with weak 3D center estimation (X, Y, Z) without requiring extrinsic calibration.

## Key Features

- **Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA)**: Novel fusion module with geometric regularization
- **Differentiable Log-Polar Preprocessing**: Handles sonar fan-shaped field of view
- **Vision-Mamba Temporal Modeling**: Linear-complexity temporal feature aggregation for long-sequence tracking
- **Weak 3D Center Estimation**: Calibration-free 3D position estimation with cyclic consistency
- **Quality-Aware Fusion**: Adaptive weighting based on RGB sharpness and Sonar SNR

## Architecture Overview

```
Template-Search (OSTrack-style) Framework
   ├─ RGB Branch: ViT/ConvNeXt
   ├─ Sonar Branch: Differentiable Log-Polar → Lightweight ViT
   ├─ Cross-Modal Fusion: G²-DCMA
   ├─ Temporal Head: Vision-Mamba (linear-complexity SSM)
   └─ Output Heads: RGB Box, Sonar Box, Fusion Box, Weak 3D Center (X, Y, Z)
```

## Development Progress

### Phase 1: Foundation & Infrastructure ✅ COMPLETED

- [x] Project structure setup
- [x] Package installation configuration (setup.py, requirements.txt)
- [x] Core module system (base classes, registry, configuration)
- [x] Unit tests for core modules (71 tests, 100% passing)
- [x] Directory structure optimization (resolved data/ naming conflict)

**Status:** All Phase 1 components verified and functional.

### Phase 2: Data Pipeline (Current - Weeks 3-4)

- [ ] RGB-Sonar dataset loader
- [ ] Template-search pair sampling
- [ ] Data transforms and augmentation
- [ ] Log-polar transform for sonar

### Upcoming Phases

- [ ] Phase 3: Backbone Networks (Weeks 5-6)
- [ ] Phase 4: Fusion Module - G²-DCMA (Weeks 7-11)
- [ ] Phase 5: Temporal Modeling (Weeks 12-13)
- [ ] Phase 6: Prediction Heads (Weeks 14-15)
- [ ] Phase 7-12: Training, Evaluation, and Finalization

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recommended for GPU training)

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/AquaSOT-3D.git
cd AquaSOT-3D

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies (for testing)
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Dataset

**Current Dataset:**
- UUV2: 2740 frames (training set)
- UUV3: 751 frames (validation/test set)

**Modalities:**
- RGB Camera: 1920×1080 pixels
- Forward-Looking Sonar: 682×376 pixels (fan-shaped field of view)

See [Dataset Specification](docs/Dataset_Specification.md) for details.

## Project Structure

```
AquaSOT-3D/
├── src/aquasot3d/           # Main package
│   ├── core/                # Base modules, registry, config
│   ├── dataloader/          # Dataset loaders and transforms
│   ├── models/              # Neural network components
│   │   ├── backbones/       # RGB and Sonar encoders
│   │   ├── fusion/          # G²-DCMA fusion module
│   │   ├── temporal/        # Temporal modeling heads
│   │   ├── heads/           # Prediction heads
│   │   └── projector/       # 3D-2D cyclic projector
│   ├── training/            # Training infrastructure
│   ├── evaluation/          # Metrics and evaluation
│   └── utils/               # Utilities
├── tests/                   # Unit and integration tests
├── configs/                 # Configuration files
├── scripts/                 # Training and evaluation scripts
├── docs/                    # Documentation
└── data/                    # Dataset storage
```

## Documentation

- [Project Overview](docs/AquaSOT-3D_E2E_RGBSonar_Tracking.md) - Complete technical specification
- [Development Planning](docs/Development_Planning.md) - Detailed development roadmap
- [Dataset Specification](docs/Dataset_Specification.md) - Dataset format and statistics

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=aquasot3d tests/

# Run specific test file
pytest tests/test_core_registry.py -v
```

**Current Test Coverage:**
- Core modules: 71 tests, 100% passing

## Technology Stack

**Deep Learning:**
- PyTorch 2.0+
- timm (Vision Transformers)
- einops (tensor operations)

**Computer Vision:**
- OpenCV
- scikit-image
- Pillow

**Scientific Computing:**
- NumPy
- SciPy

**Configuration & Logging:**
- PyYAML
- TensorBoard
- tqdm

## Base Framework

Built on top of **OSTrack** (ECCV 2022) one-stream template-search framework:
- GitHub: https://github.com/botaoye/OSTrack

## Key References

1. **RGBS50 Benchmark** (Task Definition)
   - Paper: https://arxiv.org/abs/2406.07189
   - Code: https://github.com/LiYunfengLYF/RGBS50

2. **OSTrack** (Framework Base)
   - Paper: ECCV 2022
   - Code: https://github.com/botaoye/OSTrack

3. **Vision-Mamba** (Temporal Modeling)
   - Paper: ICML 2024
   - Code: https://github.com/hustvl/Vim

4. **Deformable DETR** (Deformable Attention)
   - Code: https://github.com/fundamentalvision/Deformable-DETR

## Coordinate System

**Fusion Frame Convention:**
- X → Right
- Y → Down
- Z → Forward (depth)

RGB frame is aligned with Fusion frame. Sonar-to-Fusion transformation is learned without calibration.

## License

MIT License - See LICENSE file for details.

## Development Status

✅ **Phase 1 Complete** - Foundation & Infrastructure fully implemented and tested.

🚧 **Phase 2 Starting** - Data pipeline implementation in progress.

## Contributing

Please see [Development Planning](docs/Development_Planning.md) for contribution guidelines and development roadmap.

## Contact

For questions and issues, please open an issue on GitHub.

---

**Last Updated:** 2025-11-06
**Version:** 0.1.0
**Status:** Phase 1 Complete
