# AquaSOT-3D Training Guide

## Prerequisites

Preprocess sonar images (polar to rectangular, required once):

```bash
python scripts/preprocess_sonar.py --all
```

## Training

```bash
python scripts/train.py --config configs/train.yaml
```

### Resume Training

```bash
python scripts/train.py --config configs/train.yaml \
    --resume outputs/train/checkpoints/checkpoint_epoch_20.pth
```

### Override Parameters

```bash
python scripts/train.py --config configs/train.yaml \
    --opts training.batch_size=4 training.num_epochs=80
```

## Monitoring

```bash
tensorboard --logdir outputs/train/logs --port 6006
```

## Output Structure

```
outputs/train/
├── checkpoints/
│   ├── checkpoint_best.pth
│   ├── checkpoint_last.pth
│   └── checkpoint_epoch_N.pth
├── logs/
│   └── events.out.tfevents.*
└── config.yaml
```

## Configuration Reference

- `configs/default.yaml` - Base configuration with detailed comments
- `configs/train.yaml` - Training configuration