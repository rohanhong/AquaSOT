# AquaSOT-3D

**End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation**

A unified end-to-end multi-modal single-object tracking system that fuses RGB camera and Forward-Looking Sonar (FLS) data for underwater object tracking, with weak 3D center estimation (X, Y, Z) without requiring extrinsic calibration.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

- **Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA)**: Novel fusion module with geometric regularization
- **Differentiable Log-Polar Preprocessing**: Handles sonar fan-shaped field of view with learnable parameters
- **Vision-Mamba Temporal Modeling**: Linear-complexity temporal feature aggregation for long-sequence tracking
- **Weak 3D Center Estimation**: Calibration-free 3D position estimation with cyclic consistency
- **Quality-Aware Fusion**: Adaptive weighting based on RGB sharpness and Sonar SNR
- **OSTrack Compatible**: Template-search paradigm for robust tracking

## Architecture Overview

```
Template-Search (OSTrack-style) Framework
   ├─ RGB Branch: ViT/ConvNeXt (pretrained)
   ├─ Sonar Branch: Differentiable Log-Polar → Lightweight ViT
   ├─ Cross-Modal Fusion: G²-DCMA
   │    · Geometry-Gated (radial–scale priors)
   │    · Deformable (cross-modal sampling alignment)
   │    · Dual-Consistency (representation & component consistency)
   ├─ Temporal Head: Vision-Mamba (linear-complexity SSM)
   └─ Output Heads: RGB Box, Sonar Box, Fusion Box, Weak 3D Center (X, Y, Z)
```

## Project Status

**Current Phase:** Data Pipeline (Phase 2) - ✅ **COMPLETED**

**Next Phase:** Backbone Networks (Phase 3)

### Completed Milestones

#### ✅ Phase 1: Foundation & Infrastructure (Weeks 1-2)
- Project structure and configuration
- Core module system (base classes, registry, configuration)
- Unit testing framework (71 tests, 100% passing)
- Dataset specification and analysis

#### ✅ Phase 2: Data Pipeline (Weeks 3-4) - **9 Modules Implemented**
- Multi-modal dataset loader (RGB-Sonar, RGBS50 compatible)
- Temporal sequence samplers (4 strategies: uniform, random, occlusion-aware, adaptive)
- Template-search pair generation (OSTrack-style)
- Sonar geometry handling (fan-mask, radial weights)
- Differentiable log-polar transform (learnable α, β parameters)
- RGB and Sonar data augmentation (modality-specific)
- Unified augmentation pipelines (tracking, template-search modes)
- DataLoader utilities (batching, distributed, prefetching)
- Test coverage: 2 test files with 44+ test cases

**Code Statistics:**
- 9 core modules (~3,948 lines of code)
- 2 test files (44+ test cases)
- 2 example files (config + usage)
- 100% docstring coverage

See [Development Planning](docs/Development_Planning.md) for detailed module specifications.

### Upcoming Phases

- [ ] **Phase 3:** Backbone Networks (Weeks 5-6) - *NEXT*
- [ ] **Phase 4:** Fusion Module - G²-DCMA (Weeks 7-11)
- [ ] **Phase 5:** Temporal Modeling (Weeks 12-13)
- [ ] **Phase 6:** Prediction Heads (Weeks 14-15)
- [ ] **Phases 7-12:** Training, Evaluation, and Finalization

## Quick Start

### Installation

See [Installation Guide](docs/Development_Planning.md#installation-guide) in Development Planning for detailed instructions.

```bash
# Clone repository
git clone https://github.com/yourusername/AquaSOT-3D.git
cd AquaSOT-3D

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Or install core dependencies only
pip install -e .
```

### Verify Installation

```bash
# Check package installation
python -c "import aquasot3d; print('AquaSOT-3D installed successfully!')"

# Run Phase 2 verification script
python3 tests/test_phase2_verification.py

# Run tests
pytest tests/test_core_registry.py tests/test_rgb_sonar_dataset.py -v
```

### Usage Examples

```bash
# Run data loading example (requires dataset)
python examples/dataloader_usage_example.py

# View augmentation configuration
cat examples/augmentation_example.yaml
```

## Dataset

**Current Dataset:**
- **UUV2:** 2,740 frames (training set)
- **UUV3:** 751 frames (validation/test set)

**Modalities:**
- **RGB Camera:** 1920×1080 pixels
- **Forward-Looking Sonar:** 682×376 pixels (fan-shaped field of view)

**Annotation Format:**
- RGBS50-style groundtruth.txt (x,y,w,h per frame)
- Occlusion indicator: (0,0,0,0)

See [Dataset Specification](docs/Dataset_Specification.md) for details.

## Project Structure

```
AquaSOT-3D/
├── src/aquasot3d/              # Main package
│   ├── core/                   # Base modules, registry, config ✅
│   ├── dataloader/             # Dataset loaders and transforms ✅
│   │   ├── datasets/           # RGB-Sonar dataset, samplers, augmentation
│   │   ├── transforms/         # RGB/Sonar transforms, log-polar, geometry
│   │   └── loader.py           # DataLoader utilities
│   ├── models/                 # Neural network components (Phase 3+)
│   │   ├── backbones/          # RGB and Sonar encoders
│   │   ├── fusion/             # G²-DCMA fusion module
│   │   ├── temporal/           # Temporal modeling heads
│   │   ├── heads/              # Prediction heads
│   │   └── projector/          # 3D-2D cyclic projector
│   ├── training/               # Training infrastructure
│   ├── evaluation/             # Metrics and evaluation
│   └── utils/                  # Utilities
├── tests/                      # Unit and integration tests ✅
├── configs/                    # Configuration files
├── scripts/                    # Training and evaluation scripts
│   └── verify_phase2.py        # Phase 2 verification script ✅
├── examples/                   # Usage examples ✅
│   ├── augmentation_example.yaml
│   └── dataloader_usage_example.py
├── docs/                       # Documentation
│   ├── AquaSOT-3D_E2E_RGBSonar_Tracking.md
│   ├── Development_Planning.md
│   └── Dataset_Specification.md
└── data/                       # Dataset storage (not included in repo)
```

## Documentation

- [Project Overview](docs/AquaSOT-3D_E2E_RGBSonar_Tracking.md) - Complete technical specification
- [Development Planning](docs/Development_Planning.md) - Detailed 26-week roadmap with installation guide
- [Dataset Specification](docs/Dataset_Specification.md) - Dataset format and statistics

## Testing

```bash
# Run all tests
pytest tests/

# Run Phase 1 tests (core infrastructure)
pytest tests/test_core_registry.py tests/test_config_loader.py tests/test_base_module.py -v

# Run Phase 2 tests (data pipeline)
pytest tests/test_rgb_sonar_dataset.py tests/test_dataloader.py -v

# Run with coverage report
pytest --cov=aquasot3d tests/

# Run verification without full dependencies
python3 scripts/verify_phase2.py
```

**Current Test Coverage:**
- Phase 1 (Core): 71 tests, 100% passing ✅
- Phase 2 (Data): 44+ tests ✅

## Development Roadmap

Detailed development plan available in [docs/Development_Planning.md](docs/Development_Planning.md)

### Timeline (26 weeks)

| Phase | Weeks | Status | Modules | Description |
|-------|-------|--------|---------|-------------|
| Phase 1 | 1-2 | ✅ **COMPLETED** | 3 | Foundation & Infrastructure |
| Phase 2 | 3-4 | ✅ **COMPLETED** | 9 | Data Pipeline |
| Phase 2.5 | 4.5 | ✅ **COMPLETED** | - | Template-Search Framework |
| Phase 3 | 5-6 | 📋 Pending | 4 | Backbone Networks |
| Phase 4 | 7-11 | 📋 Pending | 4 | G²-DCMA Fusion Module |
| Phase 5 | 12-13 | 📋 Pending | 2 | Temporal Modeling |
| Phase 6 | 14-15 | 📋 Pending | 4 | Prediction Heads |
| Phase 7 | 16 | 📋 Pending | 1 | Complete Tracker Model |
| Phase 8 | 17-18 | 📋 Pending | 4 | Training Infrastructure |
| Phase 9 | 19-20 | 📋 Pending | 3 | Evaluation & Metrics |
| Phase 10 | 21 | 📋 Pending | - | Configuration & Scripts |
| Phase 11 | 22-23 | 📋 Pending | - | Integration Testing |
| Phase 12 | 24 | 📋 Pending | - | Documentation & Finalization |

**Total Progress: 2/12 phases complete (16.7%)**

## Technology Stack

**Deep Learning:**
- PyTorch 2.0+
- timm (Vision Transformers)
- einops (tensor operations)
- mamba-ssm (Vision-Mamba, optional)

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
- Paper: *Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework*
- Code: https://github.com/botaoye/OSTrack

## Key References

1. **RGBS50 Benchmark** (Task Definition)
   - Li et al., *RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker*, 2024
   - Paper: https://arxiv.org/abs/2406.07189
   - Code: https://github.com/LiYunfengLYF/RGBS50

2. **OSTrack** (Framework Base)
   - Ye et al., ECCV 2022
   - Code: https://github.com/botaoye/OSTrack

3. **Vision-Mamba** (Temporal Modeling)
   - Zhu et al., ICML 2024
   - Code: https://github.com/hustvl/Vim

4. **ARTrackV2** (Alternative Temporal Model)
   - Bai et al., CVPR 2024
   - Code: https://github.com/MIV-XJTU/ARTrack

5. **Deformable DETR** (Deformable Attention)
   - Zhu et al., ICLR 2021
   - Code: https://github.com/fundamentalvision/Deformable-DETR

## Coordinate System

**Fusion Frame Convention:**
- X → Right
- Y → Down
- Z → Forward (depth)

RGB frame is aligned with Fusion frame. Sonar-to-Fusion transformation is learned without calibration.

## Contributing

Please see [Development Planning](docs/Development_Planning.md) for contribution guidelines and development roadmap.

### Code Style

- Follow PEP 8 conventions
- Use Black for code formatting
- Type hints for all function signatures
- Docstrings in Google style (required)

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** feat, fix, docs, test, refactor, perf, chore

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{aquasot3d2025,
  title={AquaSOT-3D: End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation},
  author={AquaSOT-3D Team},
  year={2025},
  url={https://github.com/yourusername/AquaSOT-3D}
}
```

## Contact

For questions and issues, please open an issue on [GitHub Issues](https://github.com/yourusername/AquaSOT-3D/issues).

---

**Last Updated:** 2025-11-06
**Version:** 0.1.0
**Status:** Phase 2 Complete (Data Pipeline) ✅
**Next:** Phase 3 (Backbone Networks)
