# AquaSOT-3D Training Guide

**Version:** 1.0
**Date:** 2025-11-12

---

## Overview

AquaSOT-3D uses a **three-stage training strategy**:

1. **Stage 1:** RGB-only pretraining (40 epochs)
2. **Stage 2:** Sonar-only pretraining (40 epochs)
3. **Stage 3:** Multi-modal fusion training (60 epochs)

---

## Quick Start

### Stage 1: RGB Pretraining

```bash
python scripts/train.py --config configs/train/pretrain_rgb_only.yaml
```

### Stage 2: Sonar Pretraining

```bash
python scripts/train.py --config configs/train/pretrain_sonar_only.yaml
```

### Stage 3: Fusion Training

```bash
python scripts/train.py --config configs/train/train_uuv2.yaml
```

---

## Configuration Files

All configurations are in `configs/` directory:

```
configs/
├── base/default.yaml              # Base configuration
├── train/
│   ├── pretrain_rgb_only.yaml     # Stage 1: RGB pretraining
│   ├── pretrain_sonar_only.yaml   # Stage 2: Sonar pretraining
│   └── train_uuv2.yaml            # Stage 3: Fusion training
└── train_test/
    ├── test_pretrain_rgb.yaml     # Stage 1: Quick test (5 epochs)
    ├── test_pretrain_sonar.yaml   # Stage 2: Quick test (5 epochs)
    └── test_fusion.yaml           # Stage 3: Quick test (5 epochs)
```

Configuration files use YAML inheritance via `_base_` directive.

---

## Command-Line Options

### Basic Usage

```bash
python scripts/train.py --config <config_file>
```

### Override Parameters

```bash
python scripts/train.py \
    --config configs/train/train_uuv2.yaml \
    --opts training.batch_size=4 training.num_epochs=80
```

### Resume Training

```bash
python scripts/train.py \
    --config configs/train/train_uuv2.yaml \
    --resume outputs/train_uuv2/checkpoints/checkpoint_epoch_20.pth
```

---

## Testing Configuration

Quick validation tests (5 epochs each):

```bash
# Stage 1 test
python scripts/train.py --config configs/train_test/test_pretrain_rgb.yaml

# Stage 2 test
python scripts/train.py --config configs/train_test/test_pretrain_sonar.yaml

# Stage 3 test
python scripts/train.py --config configs/train_test/test_fusion.yaml
```

---

## Distributed Training

### Multi-GPU Training

```bash
# 2 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train.py \
    --config configs/train/train_uuv2.yaml

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/train/train_uuv2.yaml
```

---

## Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/<experiment>/logs --port 6006
```

### Log Files

- Training log: `outputs/<experiment>/logs/train.log`
- Config snapshot: `outputs/<experiment>/config.yaml`

---

## Checkpoints

### Automatic Saving

Checkpoints are saved based on `save_interval` in config:

- Regular: `checkpoint_epoch_N.pth` (every N epochs per config)
- Best model: `checkpoint_best.pth` (highest validation metric)
- Latest (auto): Managed by `max_to_keep` parameter

### Loading Checkpoints

```bash
# Resume from specific epoch
python scripts/train.py \
    --config configs/train/train_uuv2.yaml \
    --resume outputs/train_uuv2/checkpoints/checkpoint_epoch_20.pth

# Resume from best model
python scripts/train.py \
    --config configs/train/train_uuv2.yaml \
    --resume outputs/train_uuv2/checkpoints/checkpoint_best.pth
```

---

## Common Issues

### Out of Memory

Reduce batch size:
```bash
python scripts/train.py --config <config> --opts training.batch_size=2
```

Use gradient accumulation to maintain effective batch size:
```bash
python scripts/train.py --config <config> \
    --opts training.batch_size=2 training.gradient_accumulation_steps=8
```

### NaN Loss

Reduce learning rate:
```bash
python scripts/train.py --config <config> --opts training.optimizer.lr=0.0001
```

Enable gradient clipping (should be enabled by default):
```bash
python scripts/train.py --config <config> --opts training.grad_clip.max_norm=0.5
```

### Slow Training

Increase dataloader workers:
```bash
python scripts/train.py --config <config> --opts data.num_workers=8
```

---

## Best Practices

✓ Complete Stage 1 and Stage 2 before Stage 3
✓ Use test configs for quick validation before full training
✓ Monitor TensorBoard during training
✓ Keep `save_interval == val_interval` for optimal best model tracking

---

**Last Updated:** 2025-11-12
