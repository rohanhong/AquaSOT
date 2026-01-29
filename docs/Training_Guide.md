# AquaSOT Training Guide

## Prerequisites

### 1. Data Preparation

AquaSOT dataset structure:
```
data/
    uuv1/
        light/
            groundtruth.txt
            img/
                1.jpg, 2.jpg, 3.jpg, ...
        sonar/
            groundtruth.txt       # original polar coordinates
            groundtruth_rect.txt  # rect-converted (used)
            img/                  # original polar images
            img_rect/             # rect-converted (used)
                1.jpg, 2.jpg, 3.jpg, ...
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
wget -O models/pretrained/mae_pretrain_vit_base.pth \
    https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```

## Hardware Auto-Configuration

Detect hardware and get recommended configuration:

```bash
# Show recommended configuration
python scripts/auto_config.py

# Create new config based on existing YAML (creates vitb_256_dual_auto.yaml)
python scripts/auto_config.py --base experiments/aquasot/vitb_256_dual.yaml

# Create with custom suffix (creates vitb_256_dual_custom.yaml)
python scripts/auto_config.py --base experiments/aquasot/vitb_256_dual.yaml --suffix _custom
```

Example output:
```
Hardware Detection: RTX 3070 (8GB), 16 CPU cores
Recommended: BATCH_SIZE=1, NUM_WORKER=12, AMP=True, GRADIENT_ACCUMULATION=8
Configuration created: experiments/aquasot/vitb_256_dual_auto.yaml
```

The `--base` option creates a new configuration file with modified parameters:
- `TRAIN.BATCH_SIZE`
- `TRAIN.NUM_WORKER`
- `TRAIN.GRADIENT_ACCUMULATION`
- `TRAIN.AMP`

The original config file remains unchanged.

## Training

### Basic Training

```bash
python tracking/train.py --script aquasot --config vitb_256_dual
```

### Specify Output Directory

```bash
python tracking/train.py --script aquasot --config vitb_256_dual \
    --save_dir outputs/experiment_001
```

### Multi-GPU Training (Distributed)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tracking/train.py \
    --script aquasot \
    --config vitb_256_dual
```

### Resume Training

Training automatically resumes from latest checkpoint if available.

## Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/train/aquasot/vitb_256_dual/logs --port 6006
```

### Weights & Biases (wandb)

**Command-line (recommended):**
```bash
python tracking/train.py --script aquasot --config vitb_256_dual \
    --wandb \
    --wandb_project aquasot \
    --wandb_name exp001_baseline \
    --wandb_entity your-username
```

**YAML config:**
```yaml
WANDB:
  ENABLED: true
  PROJECT: "aquasot"
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
outputs/train/aquasot/vitb_256_dual/
    checkpoints/
        checkpoint_ep0005.pth     # Saved at checkpoint_interval
        checkpoint_ep0010.pth
        ...
        checkpoint_latest.pth     # Updated every epoch
    logs/
        events.out.tfevents.*
```

## Configuration

### Configuration Hierarchy

1. **Default config**: `lib/config/aquasot/config.py` - All parameters defined here
2. **Experiment config**: `experiments/aquasot/*.yaml` - Overrides defaults

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
    SEQUENCES: [uuv1, uuv2, uuv4, uuv5, uuv6, uuv7, uuv8, uuv9, uuv10, uuv11, uuv12]
  VAL:
    SEQUENCES: [uuv3]
  TEMPLATE:
    ONLINE_NUMBER: 1      # Online template sampling (MixFormer style)
```

### Multi-Template Training

The training pipeline supports MixFormer-style multi-template learning:

- `DATA.TEMPLATE.ONLINE_NUMBER`: Number of online templates sampled per iteration (default: 1)
- Online templates are sampled from frames between the first-frame template and search frame
- This enables the model to learn adaptive template update during tracking

To disable online template sampling:
```yaml
DATA:
  TEMPLATE:
    ONLINE_NUMBER: 0
```

## Train/Val Split

- **Training**: uuv1, uuv2, uuv4, uuv5, uuv6, uuv7, uuv8, uuv9, uuv10, uuv11, uuv12 (11 sequences, 10,940 frames)
- **Validation**: uuv3 (1 sequence, 751 frames)
