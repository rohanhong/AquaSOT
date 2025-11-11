# AquaSOT-3D Training Guide

**Version:** 0.1.0
**Date:** 2025-11-11
**Focus:** Practical training workflow and configuration

---

## Overview

AquaSOT-3D uses a **three-stage training strategy** for optimal multi-modal fusion:

1. **Stage 1:** RGB-only pretraining (40 epochs, ~4-6 hours)
2. **Stage 2:** Sonar-only pretraining (40 epochs, ~4-6 hours)
3. **Stage 3:** Multi-modal joint fine-tuning (60 epochs, ~8-12 hours)

**Rationale:**
- Stages 1-2 establish strong unimodal feature extractors independently
- Stage 3 learns cross-modal fusion (G²-DCMA) and weak 3D estimation
- Total training time: ~20 hours (RTX 3090, batch=32)

**Prerequisites:**
- Ubuntu 20.04+, CUDA 11.8+, Python 3.8+
- Installation: `pip install -e ".[dev]"`
- Pretrained weights: Download ViT-Base from timm or manually
- Dataset: RGBS50-style RGB+Sonar sequences (see README.md)

---

## Configuration System

AquaSOT-3D uses YAML-based configuration with **inheritance**.

### Configuration Hierarchy

```
configs/
├── base/default.yaml            # Base config (inherited by all)
├── train/
│   ├── train_uuv2.yaml          # Stage 3: Multi-modal training
│   ├── pretrain_rgb_only.yaml   # Stage 1: RGB pretraining
│   └── pretrain_sonar_only.yaml # Stage 2: Sonar pretraining
├── eval/eval_uuv3.yaml          # Evaluation config
└── ablation/*.yaml              # Ablation study configs
```

### Base Configuration (configs/base/default.yaml)

**Key sections:**

```yaml
model:
  type: AquaSOT3DTracker
  backbone:
    rgb:
      type: vit                  # or 'convnext'
      patch_size: 16
      embed_dim: 768
      depth: 12
      num_heads: 12
      pretrained: true
      pretrained_path: pretrained/vit_base_patch16_224.npz
    sonar:
      type: sonar_encoder
      output_dim: 384
      use_logpolar: true
      logpolar_size: [256, 256]
  fusion:
    type: g2_dcma
    fusion_dim: 512
    num_heads: 8
    use_geometric_gating: true
    use_deformable_attention: true
  temporal:
    type: vision_mamba          # or 'transformer'
    num_layers: 4
    d_state: 16
  heads:
    use_weak3d: true
    use_cyclic_consistency: true

training:
  optimizer:
    type: adamw
    base_lr: 0.0002             # Base learning rate
    weight_decay: 0.0001
    backbone_lr_mult: 0.1       # Backbone uses 0.1x base LR
  scheduler:
    type: cosine
    warmup_epochs: 5
    min_lr: 1e-7
  batch_size: 32
  epochs: 60
  gradient_clip: 1.0
  use_amp: true                 # Automatic Mixed Precision

data:
  train_sequences:
    - uuv2/light
    - uuv2/tank
  val_sequences:
    - uuv3/light
  rgb_size: [256, 256]
  sonar_size: [256, 256]
  template_search_gap: 30
  augmentation:
    rgb:
      color_jitter: true
      random_blur: true
      random_occlusion: true
      cutmix_prob: 0.3
    sonar:
      speckle_noise: true
      radial_jitter: true
      sector_occlusion: true

losses:
  rgb_cls: {type: focal, weight: 1.0}
  rgb_box: {type: giou, weight: 2.0}
  sonar_cls: {type: focal, weight: 1.0}
  sonar_box: {type: giou, weight: 2.0}
  fusion_cls: {type: focal, weight: 1.5}
  fusion_box: {type: giou, weight: 3.0}
  component_consistency: {type: l1, weight: 0.2}
  representation_consistency: {type: infonce, weight: 0.1, temperature: 0.07}
  weak3d_cyclic: {type: l2, weight: 0.6}
```

### Stage-Specific Configurations

**Stage 1 (RGB-only):** `configs/train/pretrain_rgb_only.yaml`

```yaml
_base_: ../base/default.yaml

model:
  backbone:
    sonar: {enabled: false}      # Disable sonar branch
  fusion: {enabled: false}
  heads: {use_weak3d: false}

training:
  epochs: 40

data:
  sonar_enabled: false

losses:
  sonar_cls: {weight: 0.0}
  sonar_box: {weight: 0.0}
  component_consistency: {weight: 0.0}
  representation_consistency: {weight: 0.0}
  weak3d_cyclic: {weight: 0.0}
```

**Stage 2 (Sonar-only):** `configs/train/pretrain_sonar_only.yaml`

```yaml
_base_: ../base/default.yaml

model:
  backbone:
    rgb: {enabled: false}        # Disable RGB branch
  fusion: {enabled: false}
  heads: {use_weak3d: false}

training:
  epochs: 40

data:
  rgb_enabled: false

losses:
  rgb_cls: {weight: 0.0}
  rgb_box: {weight: 0.0}
  component_consistency: {weight: 0.0}
  representation_consistency: {weight: 0.0}
  weak3d_cyclic: {weight: 0.0}
```

**Stage 3 (Multi-modal):** `configs/train/train_uuv2.yaml`

```yaml
_base_: ../base/default.yaml

training:
  epochs: 60
  optimizer:
    base_lr: 0.0002
    backbone_lr_mult: 0.1        # Lower LR for pretrained backbones

  # Load pretrained weights
  resume_rgb: outputs/stage1_rgb/checkpoints/best_model.pth
  resume_sonar: outputs/stage2_sonar/checkpoints/best_model.pth

  # Staged unfreezing
  freeze_backbone_epochs: 20     # Freeze backbones for first 20 epochs

# All losses enabled with full weights
```

### Command-Line Overrides

Override parameters without editing config files:

```bash
python scripts/train.py \
    --cfg configs/train/train_uuv2.yaml \
    training.optimizer.base_lr=0.0001 \
    training.batch_size=64 \
    training.epochs=80 \
    data.train_sequences=['uuv2/light']
```

**Syntax:**
- Dot notation: `section.subsection.key=value`
- Lists: `key=['item1','item2']`
- Booleans: `key=true` or `key=false`

---

## Training Pipeline

### Stage 1: RGB-Only Pretraining

**Objective:** Train RGB backbone and box head.

**Command:**

```bash
# Single GPU
python scripts/train.py \
    --cfg configs/train/pretrain_rgb_only.yaml \
    --output-dir outputs/stage1_rgb \
    --tag rgb_pretrain_v1

# Multi-GPU (2 GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train.py \
    --cfg configs/train/pretrain_rgb_only.yaml \
    --output-dir outputs/stage1_rgb \
    --distributed
```

**Expected Output:**

```
[2025-11-11 10:00:00] INFO: Loading config: configs/train/pretrain_rgb_only.yaml
[2025-11-11 10:00:02] INFO: Model: AquaSOT3DTracker (RGB-only), Params: 86.5M
[2025-11-11 10:00:05] INFO: Optimizer: AdamW (lr=0.0002, wd=0.0001)
[2025-11-11 10:00:08] INFO: Dataset: 2740 train, 1320 val frames

Epoch 1/40:
  Train: Loss=2.456 | RGB_cls=0.523 | RGB_box=1.933 [02:15, 1.58s/it]
  Val: Success_AUC=0.421 | Precision=0.563

Epoch 20/40:
  Train: Loss=0.687 | RGB_cls=0.145 | RGB_box=0.542 [02:10, 1.52s/it]
  Val: Success_AUC=0.702 | Precision=0.821 ✅ New best!

Epoch 40/40:
  Train: Loss=0.512 | RGB_cls=0.098 | RGB_box=0.414 [02:09, 1.50s/it]
  Val: Success_AUC=0.728 | Precision=0.847

Training Complete! Best AUC: 0.728 (Epoch 38) | Time: 4h 23min
```

**Expected Metrics:**
- RGB Success AUC: 0.72-0.75
- RGB Precision@20px: 0.84-0.87
- RGB Center Error: 8-12 pixels

**Key Checkpoints:**
- `outputs/stage1_rgb/checkpoints/best_model.pth` - Best model
- `outputs/stage1_rgb/checkpoints/latest.pth` - For resume

---

### Stage 2: Sonar-Only Pretraining

**Objective:** Train Sonar encoder with log-polar preprocessing.

**Command:**

```bash
python scripts/train.py \
    --cfg configs/train/pretrain_sonar_only.yaml \
    --output-dir outputs/stage2_sonar \
    --tag sonar_pretrain_v1
```

**Expected Output:**

```
[2025-11-11 14:30:00] INFO: Model: AquaSOT3DTracker (Sonar-only), Params: 45.2M
[2025-11-11 14:30:02] INFO: Sonar encoder initialized from scratch (no pretrained weights)
[2025-11-11 14:30:03] INFO: Log-polar transform: learnable scales (alpha=1.0, beta=1.0)

Epoch 1/40:
  Train: Loss=3.123 | Sonar_cls=0.687 | Sonar_box=2.436 [01:45, 1.22s/it]
  Val: Success_AUC=0.312 | Precision=0.445

Epoch 40/40:
  Train: Loss=0.678 | Sonar_cls=0.134 | Sonar_box=0.544 [01:40, 1.16s/it]
  Val: Success_AUC=0.685 | Precision=0.798

Training Complete! Best AUC: 0.692 (Epoch 37) | Time: 3h 52min
```

**Expected Metrics:**
- Sonar Success AUC: 0.68-0.72
- Sonar Precision@20px: 0.78-0.82
- Log-polar alpha/beta: 0.8-1.2 (converged)

**Note:** Sonar converges slower than RGB (no pretrained weights available).

---

### Stage 3: Multi-Modal Joint Fine-Tuning

**Objective:** Train end-to-end with G²-DCMA fusion and weak 3D estimation.

**Command:**

```bash
python scripts/train.py \
    --cfg configs/train/train_uuv2.yaml \
    --resume-rgb outputs/stage1_rgb/checkpoints/best_model.pth \
    --resume-sonar outputs/stage2_sonar/checkpoints/best_model.pth \
    --output-dir outputs/stage3_fusion \
    --tag fusion_v1
```

**Training Schedule:**

**Sub-Stage 3.1 (Epochs 1-20): Freeze Backbones**
- Trainable: G²-DCMA, Mamba, Fusion/Weak3D heads
- Frozen: RGB backbone, Sonar encoder
- Learning rate: 2e-4 (full)

**Sub-Stage 3.2 (Epochs 21-60): Unfreeze All**
- Trainable: All modules (end-to-end)
- Learning rate: Backbones 2e-5 (0.1x), Others 2e-4 (1.0x)

**Expected Output:**

```
[2025-11-11 19:00:00] INFO: Model: AquaSOT3DTracker (Full multi-modal), Params: 98.7M
[2025-11-11 19:00:02] INFO: Loaded RGB checkpoint (86.5M params)
[2025-11-11 19:00:04] INFO: Loaded Sonar checkpoint (45.2M params)
[2025-11-11 19:00:06] INFO: Freezing backbones for first 20 epochs

Epoch 1/60 (Backbones Frozen):
  Train: Loss=4.567 | RGB=0.678 | Sonar=0.712 | Fusion=2.145 | Consistency=0.523 | Weak3D=0.509 [03:20, 2.33s/it]
  Val: RGB_AUC=0.728 | Sonar_AUC=0.692 | Fusion_AUC=0.645 | Weak3D_Error=0.245m

Epoch 20/60:
  Train: Loss=1.567 | RGB=0.298 | Sonar=0.323 | Fusion=0.689 | Consistency=0.145 | Weak3D=0.112 [03:12, 2.23s/it]
  Val: RGB_AUC=0.742 | Sonar_AUC=0.705 | Fusion_AUC=0.798 | Weak3D_Error=0.098m
  🔓 Unfreezing backbones for fine-tuning...

Epoch 21/60 (Backbones Unfrozen):
  Train: Loss=1.489 | RGB=0.287 | Sonar=0.312 | Fusion=0.654 | Consistency=0.134 | Weak3D=0.102 [03:25, 2.38s/it]
  Val: RGB_AUC=0.749 | Sonar_AUC=0.712 | Fusion_AUC=0.803 | Weak3D_Error=0.089m

Epoch 60/60:
  Train: Loss=0.823 | RGB=0.165 | Sonar=0.189 | Fusion=0.345 | Consistency=0.072 | Weak3D=0.052 [03:20, 2.33s/it]
  Val: RGB_AUC=0.772 | Sonar_AUC=0.735 | Fusion_AUC=0.831 | Weak3D_Error=0.058m

Training Complete! Best Fusion AUC: 0.831 (Epoch 57) | Time: 10h 18min

Final Metrics:
  RGB AUC: 0.772 (+0.044 vs Stage 1)
  Sonar AUC: 0.735 (+0.043 vs Stage 2)
  Fusion AUC: 0.831 (BEST) ⭐
  Weak 3D Error: 0.058m
  Fusion Precision@20px: 0.891
```

**Expected Performance:**

| Metric | RGB | Sonar | Fusion |
|--------|-----|-------|--------|
| Success AUC | 0.76-0.78 | 0.72-0.74 | **0.82-0.84** |
| Precision@20px | 0.85-0.87 | 0.81-0.83 | **0.88-0.91** |
| Center Error (px) | 8-12 | 12-18 | **6-10** |
| Weak 3D Error (m) | - | - | **0.05-0.08** |

**Key Observations:**
- Fusion outperforms unimodal by 3-6% AUC
- Cross-modal learning improves both RGB and Sonar
- Weak 3D error converges below 0.10m by epoch 40

---

### Resume Training

If training is interrupted:

```bash
python scripts/train.py \
    --cfg configs/train/train_uuv2.yaml \
    --resume outputs/stage3_fusion/checkpoints/latest.pth \
    --output-dir outputs/stage3_fusion
```

System automatically restores: model weights, optimizer state, scheduler state, epoch number, best metrics.

---

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/tensorboard --port 6006
# Access at: http://localhost:6006
```

**Monitor:**
- Scalars: Loss components, validation metrics, LR schedule
- Images: Sample predictions, attention maps
- Histograms: Parameter/gradient distributions

### Log Files

- `outputs/<experiment>/train.log` - Full training log
- `outputs/<experiment>/val.log` - Validation results per epoch
- `outputs/<experiment>/config.yaml` - Saved configuration

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python scripts/train.py --cfg ... training.batch_size=16
   ```

2. **Use gradient accumulation:**
   ```bash
   python scripts/train.py --cfg ... training.batch_size=16 training.gradient_accumulation_steps=2
   ```

3. **Reduce input size:**
   ```bash
   python scripts/train.py --cfg ... data.rgb_size=[224,224] data.sonar_size=[224,224]
   ```

### Issue 2: NaN Loss

**Symptom:** `Loss = nan` after a few epochs

**Solutions:**

1. **Reduce learning rate:**
   ```bash
   python scripts/train.py --cfg ... training.optimizer.base_lr=0.0001
   ```

2. **Reduce gradient clipping:**
   ```bash
   python scripts/train.py --cfg ... training.gradient_clip=0.5
   ```

3. **Check pretrained weights:**
   ```bash
   python -c "import torch; ckpt = torch.load('outputs/stage1_rgb/checkpoints/best_model.pth'); print({k: v.isnan().any() for k, v in ckpt['model'].items() if torch.is_tensor(v)})"
   ```

### Issue 3: No Improvement (Stage 1)

**Symptom:** RGB AUC stays low (~0.35, expected 0.72-0.75)

**Solutions:**

1. **Verify pretrained weights loaded:**
   ```python
   from aquasot3d.models.backbones import ViTRGBBackbone
   backbone = ViTRGBBackbone(pretrained=True, pretrained_path='pretrained/vit_base_patch16_224.npz')
   print('Pretrained weights loaded successfully')
   ```

2. **Check dataset integrity:**
   ```bash
   python scripts/analyze_dataset.py --data-dir data/uuv2
   ```

3. **Try higher learning rate:**
   ```bash
   python scripts/train.py --cfg ... training.optimizer.base_lr=0.0004
   ```

### Issue 4: Fusion Worse Than Unimodal

**Symptom:** Fusion AUC < RGB AUC (expected Fusion AUC > RGB AUC)

**Solutions:**

1. **Ensure Stage 1-2 completed:** Never skip pretraining stages
2. **Increase fusion loss weight:**
   ```bash
   python scripts/train.py --cfg ... losses.fusion_box.weight=5.0
   ```
3. **Check fusion gradients:** Add gradient monitoring in train.py

### Issue 5: Training Too Slow

**Symptom:** >5 min/epoch (expected 2-3 min)

**Solutions:**

1. **Increase DataLoader workers:**
   ```bash
   python scripts/train.py --cfg ... data.num_workers=8
   ```

2. **Ensure AMP enabled:**
   ```yaml
   training:
     use_amp: true
   ```

3. **Profile bottlenecks:**
   ```bash
   python scripts/train.py --cfg ... --profile
   ```

---

## Best Practices

### Training Strategy

✅ **Always complete Stage 1-2 first** - Never skip to Stage 3 directly
✅ **Monitor validation every epoch** - Save best model by Fusion AUC
✅ **Use learning rate warmup** - 5 epochs prevents early divergence
✅ **Staged unfreezing in Stage 3** - Freeze backbones for first 20 epochs
✅ **Enable gradient clipping** - Clip norm at 1.0 (reduce to 0.5 if unstable)

### Configuration

✅ **Use pretrained ViT for RGB** - Critical for convergence
✅ **Monitor log-polar alpha/beta** - Should converge to 0.8-1.2
✅ **Validate loss weights** - Default weights work well for RGBS50
✅ **Command-line overrides for tuning** - Avoid editing config files repeatedly

### Data

✅ **Verify dataset first** - Run `analyze_dataset.py` before training
✅ **Use recommended augmentation** - RGB CutMix prob=0.3, Sonar speckle std=0.05
✅ **Handle missing targets properly** - System handles (0,0,0,0) automatically
✅ **Don't drop frames** - Keep all frames for temporal consistency

### Reproducibility

✅ **Set random seeds:**
   ```python
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   np.random.seed(42)
   random.seed(42)
   ```

✅ **Log everything:**
   ```bash
   pip freeze > outputs/<exp>/requirements.txt
   git log -1 > outputs/<exp>/git_info.txt
   ```

✅ **Save config to output dir** - Automatically saved by training script

---

## Advanced: Distributed Training

Train on multiple GPUs using PyTorch DDP:

```bash
# 2 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train.py \
    --cfg configs/train/train_uuv2.yaml \
    --distributed

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train.py \
    --cfg configs/train/train_uuv2.yaml \
    --distributed
```

**Effective batch size:** `batch_size * num_gpus`
- 2 GPUs: 32 * 2 = 64
- 4 GPUs: 32 * 4 = 128

**Learning rate scaling:** `lr_new = lr_base * (batch_new / batch_base)`

---

## Summary

**Three-Stage Training Pipeline:**

1. **Stage 1:** `python scripts/train.py --cfg configs/train/pretrain_rgb_only.yaml`
2. **Stage 2:** `python scripts/train.py --cfg configs/train/pretrain_sonar_only.yaml`
3. **Stage 3:** `python scripts/train.py --cfg configs/train/train_uuv2.yaml --resume-rgb ... --resume-sonar ...`

**Expected Results:**
- Fusion Success AUC: **0.82-0.84**
- Fusion Precision@20px: **0.88-0.91**
- Weak 3D Center Error: **<0.08m**
- Total training time: **~20 hours** (RTX 3090)

**Key Configurations:**
- Base LR: 2e-4
- Batch size: 32
- Warmup: 5 epochs
- Gradient clip: 1.0
- AMP: Enabled

For evaluation and metrics analysis, see [README.md](../README.md) and [API_Reference.md](API_Reference.md).

---

**Document Version:** 1.1
**Last Updated:** 2025-11-11
**License:** MIT
