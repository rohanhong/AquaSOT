# AquaSOT-3D Evaluation Guide

## Evaluation

```bash
python scripts/eval.py --config configs/eval.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth
```

### Override Parameters

```bash
python scripts/eval.py --config configs/eval.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth \
    --opts 'data.test_sequences=[uuv3]'
```

## Visualization

Visualization is enabled by default. Control via CLI flags:

```bash
# Disable visualization (faster)
python scripts/eval.py --config configs/eval.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth --no-visualize

# Force enable visualization
python scripts/eval.py --config configs/eval.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth --visualize
```

## Output Structure

```
outputs/eval/
├── overall_metrics.json
├── evaluation_report.txt
├── eval_config.yaml
├── sequences/
│   └── <sequence_name>/
│       ├── predictions.npz
│       ├── metrics.json
│       └── metadata.json
├── visualization/
└── logs/
    └── eval.log
```

## Configuration Reference

- `configs/default.yaml` - Base configuration with detailed comments
- `configs/eval.yaml` - Evaluation configuration