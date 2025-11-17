# AquaSOT-3D Scripts

This directory contains executable scripts for training, evaluation, and utilities.

## Training Scripts

### train.py
Main training script for AquaSOT-3D.

**Usage:**
```bash
# Full training on UUV2
python scripts/train.py --config configs/train/train_multiseq.yaml

# RGB-only pretraining
python scripts/train.py --config configs/train/pretrain_rgb_only.yaml

# Training with overrides
python scripts/train.py --config configs/train/train_multiseq.yaml \
    --opts training.lr=0.001 training.batch_size=32

# Resume training
python scripts/train.py --config configs/train/train_multiseq.yaml \
    --resume outputs/train_multiseq/checkpoints/checkpoint_epoch_20.pth
```

**Features:**
- Configuration-based training
- Command-line overrides
- Checkpoint resuming
- Distributed training support
- Mixed precision (AMP)
- TensorBoard logging

## Evaluation Scripts

### eval.py
Evaluate trained model on test set.

**Usage:**
```bash
# Evaluate on UUV3 test set
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/best_model.pth

# Custom output directory
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint model.pth \
    --output custom_eval_output/
```

**Features:**
- RGBS50-style metrics computation
- Per-modality evaluation (RGB, Sonar, Fusion)
- Visualization generation
- Failure case analysis
- Detailed reports (txt, json, html)

## Demo Scripts

### demo.py
Run tracking demo on video or image sequence.

**Usage:**
```bash
# Video input
python scripts/demo.py \
    --checkpoint model.pth \
    --input video.mp4 \
    --output demo_result.mp4

# Image directory
python scripts/demo.py \
    --checkpoint model.pth \
    --input images/ \
    --output result.mp4
```

## Utility Scripts

### analyze_dataset.py
Analyze dataset statistics.

**Usage:**
```bash
python scripts/analyze_dataset.py \
    --data_root data/UUV2 \
    --output dataset_stats.json
```

**Statistics:**
- Sequence count and lengths
- Frame size distributions
- Attribute distributions
- Dataset quality metrics

## Configuration Files

All scripts use YAML configuration files from `configs/` directory:

```
configs/
├── base/
│   └── default.yaml              # Base configuration
├── train/
│   ├── train_multiseq.yaml      # Full multi-sequence training
│   ├── pretrain_rgb_only.yaml   # RGB pretraining
│   └── pretrain_sonar_only.yaml # Sonar pretraining
├── eval/
│   └── eval_multiseq.yaml       # Multi-sequence evaluation config
└── ablation/
    ├── ablation_no_mamba.yaml
    ├── ablation_no_gating.yaml
    └── ...                       # Other ablations
```

## Command-Line Overrides

All scripts support config overrides via `--opts`:

```bash
# Override single value
--opts training.lr=0.001

# Override multiple values
--opts training.lr=0.001 training.batch_size=32 model.temporal.enabled=false

# Override nested values
--opts model.fusion.use_geometry_gating=false
```

## Environment Setup

Ensure environment is properly setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import aquasot3d; print('AquaSOT-3D installed successfully')"
```

## Distributed Training

For multi-GPU training:

```bash
# Using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train.py \
    --config configs/train/train_multiseq.yaml

# Using torchrun (PyTorch >= 1.10)
torchrun --nproc_per_node=4 \
    scripts/train.py \
    --config configs/train/train_multiseq.yaml
```

## Common Issues

**ImportError: No module named 'aquasot3d'**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/AquaSOT-3D/src"
```

**CUDA out of memory**
```bash
# Reduce batch size or enable gradient accumulation
--opts training.batch_size=8 training.gradient_accumulation_steps=4
```

**Configuration file not found**
```bash
# Use absolute path or run from project root
python scripts/train.py --config $(pwd)/configs/train/train_multiseq.yaml
```

---

**Author:** AquaSOT-3D Team
**Date:** 2025-11-10
