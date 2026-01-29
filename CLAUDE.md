# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AquaSOT is an end-to-end dual-modal single object tracking framework for underwater environments. It fuses RGB camera and Forward-Looking Sonar (FLS) data using a dual-stream Vision Transformer architecture with Deformable Cross-Modal Attention (DCMA).

## Build and Development Commands

### Installation
```bash
pip3 install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

### Training
```bash
# Standard dual-modal training
python tracking/train.py --script aquasot --config vitb_256_dual_auto

# With W&B logging
python tracking/train.py --script aquasot --config vitb_256_dual_auto --wandb

# Ablation studies
python tracking/train.py --script aquasot --config ablation_rgb_only
python tracking/train.py --script aquasot --config ablation_sonar_only
```

### Evaluation
```bash
# Dual-modal evaluation
python tracking/test.py --tracker aquasot --checkpoint outputs/train/checkpoints/model.pth

# Single-modal evaluation (testing-time modality dropout)
python tracking/test.py --checkpoint model.pth --eval_mode rgb_only
python tracking/test.py --checkpoint model.pth --eval_mode sonar_only
```

### Testing
```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_model_ostrack_style.py -v  # Run single test file
```

### Data Preprocessing
```bash
python scripts/preprocess_sonar.py --all  # Convert sonar polar to Cartesian
```

## Architecture Overview

### Dual-Stream Pipeline
```
RGB [B, 3, H, W] -----> ViT-Base (RGB) ----+
                                           |---> DCMA Fusion ---> Modality-Specific Heads
Sonar [B, 1, H, W] ---> ViT-Base (Sonar) --+
```

### Key Components
- **Backbones**: `lib/models/aquasot/vit.py` - Dual ViT-Base encoders (RGB: 3-ch, Sonar: 1-ch)
- **Fusion**: `lib/models/aquasot/fusion.py` - DCMA cross-modal attention (2-layer bidirectional)
- **Heads**: `lib/models/layers/head.py` - Separate CenterPredictor heads per modality
- **Model**: `lib/models/aquasot/aquasot.py` - Main dual-stream model (~190 lines)

### Critical Design Decisions
1. **Separate prediction heads**: Prevents prediction drift due to asymmetric FOV (FLS 120-160 deg vs RGB 60-70 deg)
2. **Weak fusion weights (0.3)**: Preserves modality-specific information
3. **Template update**: MixFormer three-tier strategy (confidence-aware)
4. **Target loss detection**: Dual-modal independent state machine (STABLE/UNCERTAIN/LOST)

## Configuration System

- **Base config**: `lib/config/aquasot/config.py` - All default parameters
- **YAML overrides**: `experiments/aquasot/*.yaml` - Only override existing keys (validation enforced)
- YAML cannot add new parameters, only modify existing ones from config.py

## Dataset Structure

12 RGB-Sonar sequences (UUV1-12):
- Training: UUV1,2,4,5,6,7,8,9,10,11,12 (9,588 frames)
- Test/Validation: UUV3 (751 frames)

Default test set is configured in `cfg.DATA.TEST.SEQUENCES`.

Bounding box formats:
- Dataset/Evaluation: `(x, y, w, h)` top-left + size
- Model internal: `(cx, cy, w, h)` center + size

## Entry Points

| Task | Entry Point |
|------|-------------|
| Training | `tracking/train.py` |
| Evaluation | `tracking/test.py` |
| Model export | `scripts/export_checkpoint.py` |
| Sonar preprocessing | `scripts/preprocess_sonar.py` |

## Code Conventions

- Follows OSTrack design philosophy (compact, direct feature extraction)
- Loss functions: GIoU (weight 2.0), L1 (weight 5.0), Focal (weight 1.0)
- Visibility filtering: GOT-10k style (excludes absent frames with GT=[0,0,0,0])
- Pretrained weights: `models/pretrained/`
- Training outputs: `outputs/`
