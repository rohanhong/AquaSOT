# AquaSOT Training

## Prerequisites

```bash
# Sonar preprocessing (required once)
python scripts/preprocess_sonar.py --all --data-root data

# Download MAE pretrained weights
wget -O models/pretrained/mae_pretrain_vit_base.pth \
    https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```

## Hardware Auto-Configuration

```bash
# Show recommended configuration
python scripts/auto_config.py

# Create auto-tuned config (creates vitb_256_dual_auto.yaml)
python scripts/auto_config.py --base experiments/aquasot/vitb_256_dual.yaml
```

## Training

```bash
# Basic training
python tracking/train.py --script aquasot --config vitb_256_dual_auto

# With custom output directory
python tracking/train.py --script aquasot --config vitb_256_dual_auto \
    --save_dir outputs/experiment_001

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 \
    tracking/train.py --script aquasot --config vitb_256_dual_auto

# With W&B logging
python tracking/train.py --script aquasot --config vitb_256_dual_auto \
    --wandb --wandb_project aquasot --wandb_name exp001 --wandb_entity your-username
```

## Monitoring

```bash
# TensorBoard
tensorboard --logdir outputs/train/aquasot/vitb_256_dual_auto/logs --port 6006

# W&B setup (first time)
wandb login
```

## Output Structure

```
outputs/train/aquasot/vitb_256_dual_auto/
    checkpoints/
        checkpoint_ep0005.pth
        checkpoint_ep0010.pth
        checkpoint_latest.pth
    logs/
        events.out.tfevents.*
```