# AquaSOT-3D Training Guide

**Version:** 3.0
**Date:** 2025-11-19
**Multi-Sequence Architecture with Preprocessed Sonar Data**

---

## Data Preprocessing (Required First Step)

**IMPORTANT:** Before training, convert all sonar images from polar to rectangular coordinates. This eliminates runtime coordinate transformations and unifies sonar resolution to 682×376.

### Quick Start

```bash
# Preprocess all 12 sequences (required before first training)
python scripts/preprocess_sonar.py --all
```

### Output Structure

```
data/uuvX/sonar/
├── img/                    # Original polar images (preserved)
├── img_rect/               # Rectangular images (TRAINING USES THIS)
├── groundtruth.txt         # Original polar bboxes
└── groundtruth_rect.txt    # Rectangular bboxes (TRAINING USES THIS)
```

All preprocessed images are unified to **682×376 pixels**.

**Data Augmentation Note:**
Sonar augmentation pipeline uses rectangular-coordinate-appropriate transforms:
- ✅ Speckle noise (intensity perturbation)
- ✅ Rectangular occlusion (similar to RGB)
- ✅ Intensity normalization
- ❌ Polar-specific transforms (radial jitter, sector occlusion, fan mask) are disabled

---

## Overview

AquaSOT-3D supports **multi-sequence training** across the full AquaSOT-3D dataset (12 sequences):

- **Training set**: `uuv1, uuv2, uuv4, uuv5, uuv7, uuv8, uuv9, uuv11, uuv12` (9 sequences)
- **Validation set**: `uuv3, uuv6, uuv10` (3 sequences)

### Training Strategy

AquaSOT-3D uses **single-stage end-to-end training** with RGB + Sonar dual-modality:

- **Duration**: 100 epochs (extended for end-to-end convergence)
- **Initialization**: ImageNet-pretrained ViT backbones (via timm library)
- **Modalities**: RGB + Sonar (both required, no single-modality support)
- **Temporal Modeling**: Multi-frame sequences (5 consecutive frames) for Vision-Mamba

**Key Features:**

- **No pretraining stages**: Direct end-to-end optimization from scratch
- **Dual-modality fusion**: G2-DCMA fusion module enabled from the start
- **Temporal tracking**: `sequence_length=5` enables learning across consecutive frames
- **Quality-adaptive gating**: RGB and Sonar quality estimators for dynamic modality weighting

This approach simplifies the training pipeline while enabling the model to jointly learn modality-specific features and cross-modal fusion strategies.

---

## Quick Start

### Full Training Command

Train the complete model end-to-end:

```bash
# Single-stage full training (100 epochs)
python scripts/train.py --config configs/train.yaml
```

---

## Configuration Files

All configurations use the **multi-sequence format**:

```yaml
data:
  data_root: data
  train_sequences: [uuv1, uuv2, uuv4, uuv5, uuv7, uuv8, uuv9, uuv11, uuv12]
  val_sequences: [uuv3, uuv6, uuv10]
```

Available configs in `configs/` directory:

```
configs/
├── default.yaml          # Base configuration (shared defaults)
├── train.yaml            # Main training configuration (100 epochs, dual-modality)
├── eval.yaml             # Evaluation configuration
└── ablation/             # Ablation study configurations
    ├── ablation_no_mamba.yaml
    ├── ablation_no_quality_gating.yaml
    └── ...
```

### Training Configuration

**Main Configuration** (`configs/train.yaml`):

```yaml
data:
  # Multi-frame temporal training (required for Vision-Mamba)
  search_sequence_length: 5  # 5 consecutive search frames
  use_sequence_grouping: true  # Prevent cross-sequence contamination in batches

model:
  temporal:
    type: mamba
    enabled: true  # Vision-Mamba temporal modeling
```

**Key Parameters:**

- `search_sequence_length`: Number of consecutive search frames (4-6 recommended)
  - Stage 1/2 (pretraining): 1 (single-frame pairs, default)
  - Stage 3 (fusion): 5 (multi-frame sequences, **required for temporal training**)

- `use_sequence_grouping`: Ensures batches contain samples from the same sequence
  - Prevents Vision-Mamba from seeing discontinuous frames within a batch
  - Critical for training temporal state management correctly

**Adjusting Sequence Length:**

```bash
# Use shorter sequences (faster, less memory)
python scripts/train.py --config configs/train/train.yaml \
    --opts data.search_sequence_length=3

# Use longer sequences (better temporal context, more memory)
python scripts/train.py --config configs/train/train.yaml \
    --opts data.search_sequence_length=7
```

### Per-Sequence Resolution Detection

The system automatically detects RGB and Sonar resolutions for each sequence:

```bash
[INFO] Multi-sequence mode: data_root=data
[INFO] Train sequences: ['uuv1', 'uuv2', ...]
[INFO]   Train sequence 'uuv1': 2518 valid frames, RGB (1920, 1080), Sonar (682, 376)
[INFO]   Train sequence 'uuv2': 2740 valid frames, RGB (1920, 1080), Sonar (682, 376)
...
```

---

## Command-Line Options

### Basic Usage

```bash
python scripts/train.py --config <config_file>
```

### Override Parameters

```bash
python scripts/train.py \
    --config configs/train/train.yaml \
    --opts training.batch_size=4 training.num_epochs=80
```

### Resume Training

```bash
python scripts/train.py \
    --config configs/train/train.yaml \
    --resume outputs/train/checkpoints/checkpoint_epoch_20.pth
```

---

## Distributed Training

### Multi-GPU Training

```bash
# 2 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train.py \
    --config configs/train/train.yaml

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/train/train.yaml
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
    --config configs/train/train.yaml \
    --resume outputs/train/checkpoints/checkpoint_epoch_20.pth

# Resume from best model
python scripts/train.py \
    --config configs/train/train.yaml \
    --resume outputs/train/checkpoints/checkpoint_best.pth
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

### NaN Loss / CUDA Assertion Failures

**Issue**: Training crashes with CUDA assertion error: `input_val >= zero && input_val <= one` or NaN loss appears early in training.

**Root Cause**: AMP (FP16) numerical instability causing BCE loss computation to receive out-of-range values.

**Solution 1 (Recommended)**: Disable AMP for numerical stability
```bash
python scripts/train.py --config <config> --opts system.use_amp=false
```

**Solution 2**: Try BFloat16 (better numerical range than FP16)
```bash
python scripts/train.py --config <config> --opts system.amp_dtype=bfloat16
```

**Solution 3**: Reduce learning rate if using AMP
```bash
python scripts/train.py --config <config> --opts training.optimizer.lr=0.0001
```

**Additional Debugging**:
- Enable anomaly detection: `--opts debug.detect_anomaly=true`
- Check specific log: `docs/audits/Stage3_BCE_CUDA_Error_Resolution.md`

**Note**: FP32 training (AMP disabled) is ~30% slower but guarantees numerical stability for Stage 3 fusion training.

### Slow Training

Increase dataloader workers:
```bash
python scripts/train.py --config <config> --opts data.num_workers=8
```

### Corrupt JPEG Data Warning

**Issue**: Training logs show warnings like:
```
Corrupt JPEG data: premature end of data segment
Corrupt JPEG data: bad Huffman code
```

**Root Cause**: Some RGB frames in the dataset contain damaged JPEG markers that PIL/OpenCV can still decode but trigger warnings. These do not crash training but pollute logs and may indicate subtle image quality degradation.

**Solution**: Scrub damaged frames using `jpegtran` (lossless JPEG repair):

```bash
# Repair single file (lossless, strips corrupted markers)
jpegtran -copy none -optimize -progressive input.jpg > output.jpg

# Batch repair all JPEGs in a sequence directory
find data/uuv2/rgb -name "*.jpg" -exec sh -c \
  'jpegtran -copy none -optimize -progressive "$1" > "$1.tmp" && mv "$1.tmp" "$1"' _ {} \;
```

**Note**:
- The `-copy none` flag strips EXIF/metadata that may contain corruption
- `-optimize` and `-progressive` rebuild the Huffman tables cleanly
- This operation is **lossless** (pixel data unchanged)
- Run this preprocessing step **before** training if warnings appear frequently

**Reference**: See Audit 2025111406 Finding 1.4 for detailed analysis.

---

## Best Practices

✓ Complete Stage 1 and Stage 2 before Stage 3
✓ Use test configs for quick validation before full training
✓ Monitor TensorBoard during training
✓ Keep `save_interval == val_interval` for optimal best model tracking

---

**Last Updated:** 2025-11-19
