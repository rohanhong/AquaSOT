# AquaSOT-3D Training Guide

## Prerequisites

### 1. Data Preparation

AquaSOT-3D dataset structure:
```
data/
    uuv1/
        light/
            groundtruth.txt
            img/
                1000.jpg, 1001.jpg, ...
        sonar/
            groundtruth.txt       # original polar coordinates
            groundtruth_rect.txt  # rect-converted (used)
            img/                  # original polar images
            img_rect/             # rect-converted (used)
                1000.jpg, 1001.jpg, ...
    uuv2/
    ...
    uuv12/
```

### 2. Sonar Preprocessing (required once)

Convert polar sonar images to rectangular coordinates:

```bash
# Single sequence
python scripts/preprocess_sonar.py --sequence uuv1 --data-root data

# All sequences
python scripts/preprocess_sonar.py --all --data-root data
```

Output: `sonar/img_rect/` and `sonar/groundtruth_rect.txt`

### 3. Pretrained Weights

Download MAE pretrained ViT-Base weights:
```bash
mkdir -p pretrained
wget -O pretrained/mae_pretrain_vit_base.pth \
    https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```

## Hardware Auto-Configuration

Detect hardware and get recommended configuration:

```bash
# Show recommended configuration
python scripts/auto_config.py

# Generate config file
python scripts/auto_config.py --output experiments/aquasot3d/auto_hardware.yaml
```

Example output:
```
Hardware Detection: RTX 3070 (8GB), 16 CPU cores
Recommended: BATCH_SIZE=1, NUM_WORKER=12, AMP=True, GRADIENT_ACCUMULATION=8
```

## Training

### Basic Training

```bash
python tracking/train.py --script aquasot3d --config vitb_256_dual
```

### Specify Output Directory

```bash
python tracking/train.py --script aquasot3d --config vitb_256_dual \
    --save_dir outputs/experiment_001
```

### Multi-GPU Training (Distributed)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tracking/train.py \
    --script aquasot3d \
    --config vitb_256_dual
```

### Resume Training

Training automatically resumes from latest checkpoint if available.

## Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/train/aquasot3d/vitb_256_dual/logs --port 6006
```

### Weights & Biases (wandb)

**Command-line (recommended):**
```bash
python tracking/train.py --script aquasot3d --config vitb_256_dual \
    --wandb \
    --wandb_project aquasot3d \
    --wandb_name exp001_baseline \
    --wandb_entity your-username
```

**YAML config:**
```yaml
WANDB:
  ENABLED: true
  PROJECT: "aquasot3d"
  ENTITY: "your-username"  # optional

EXPERIMENT:
  NAME: vitb_256_dual_exp001
  GROUP: baseline
  TAGS: [vit_base, dual_modal]
```

First time setup:
```bash
wandb login
```

## Output Structure

```
outputs/train/aquasot3d/vitb_256_dual/
    checkpoints/
        AquaSOT3D_ep0001.pth.tar
        AquaSOT3D_ep0005.pth.tar
        ...
    logs/
        events.out.tfevents.*
```

## Configuration

### Configuration Hierarchy

1. **Default config**: `lib/config/aquasot3d/config.py` - All parameters defined here
2. **Experiment config**: `experiments/aquasot3d/*.yaml` - Overrides defaults

YAML configs can only override existing parameters from config.py.

### Available Configurations

| Config | Description |
|--------|-------------|
| `vitb_256_dual` | Main config: ViT-Base, 256x256 search, DCMA fusion |
| `ablation_rgb_only` | RGB-only ablation |
| `ablation_sonar_only` | Sonar-only ablation |
| `ablation_no_fusion` | No fusion ablation |

### Key Parameters

```yaml
TRAIN:
  LR: 0.0001              # Learning rate
  BATCH_SIZE: 2           # Batch size per GPU
  EPOCH: 100              # Total epochs
  GRADIENT_ACCUMULATION: 4 # Effective batch = BATCH_SIZE * ACCUMULATION

DATA:
  ROOT: data              # Dataset root
  TRAIN:
    SEQUENCES: [uuv1, uuv2, uuv4, uuv5, uuv7, uuv8, uuv9, uuv11, uuv12]
  VAL:
    SEQUENCES: [uuv3, uuv6, uuv10]
```

## Train/Val Split

- **Training**: uuv1, uuv2, uuv4, uuv5, uuv7, uuv8, uuv9, uuv11, uuv12 (9 sequences)
- **Validation**: uuv3, uuv6, uuv10 (3 sequences)
