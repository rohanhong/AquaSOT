# AquaSOT-3D Evaluation Guide

**Version:** 1.0
**Date:** 2025-11-12

---

## Overview

AquaSOT-3D evaluation follows **RGBS50 benchmark protocol** with three evaluation modes:

1. **RGB Tracking**: RGB-only performance assessment
2. **Sonar Tracking**: Sonar-only performance assessment
3. **Fusion Tracking**: Multi-modal fusion performance assessment

**Metrics**: Success (AUC), Precision@20, Average Center Error, Mean IoU

---

## Quick Start

### Full Model Evaluation

```bash
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/train_uuv2/checkpoints/checkpoint_best.pth
```

### Single-Modality Evaluation

```bash
# RGB-only model
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/pretrain_rgb/checkpoints/checkpoint_best.pth

# Sonar-only model
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/pretrain_sonar/checkpoints/checkpoint_best.pth
```

---

## Configuration Files

Evaluation configurations in `configs/eval/`:

```
configs/eval/
└── eval_uuv3.yaml              # Standard evaluation on UUV3 test set
```

Configuration inherits from `configs/base/default.yaml` via `_base_` directive.

---

## Command-Line Options

### Basic Usage

```bash
python scripts/eval.py --config <config_file> --checkpoint <checkpoint_path>
```

### Override Parameters

```bash
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/train_uuv2/checkpoints/checkpoint_best.pth \
    --opts evaluation.batch_size=1 evaluation.visualization.enabled=true
```

### Custom Output Directory

```bash
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/train_uuv2/checkpoints/checkpoint_best.pth \
    --output outputs/custom_eval
```

---

## Evaluation Metrics

### Primary Metrics (RGBS50 Protocol)

- **Success (AUC)**: Area Under Curve of success plot (overlap threshold 0-1)
- **Precision@20**: Precision at 20-pixel center location error threshold
- **Average Center Error**: Mean Euclidean distance between predicted and ground truth centers
- **Mean IoU**: Average Intersection-over-Union across all frames

### Per-Modality Reporting

Metrics reported separately for:
- **RGB**: RGB branch predictions
- **Sonar**: Sonar branch predictions
- **Fusion**: Multi-modal fusion predictions

---

## Output Structure

Evaluation results saved to `outputs/<experiment>/`:

```
outputs/eval_uuv3/
├── overall_metrics.json        # Aggregated metrics across all sequences
├── evaluation_report.txt       # Human-readable summary
├── eval_config.yaml           # Config snapshot
├── sequences/                 # Per-sequence results
│   ├── uuv3_001/
│   │   ├── predictions.npz    # Per-frame predictions
│   │   └── metrics.json       # Sequence metrics
│   └── ...
├── visualization/             # Visualizations (if enabled)
│   ├── trajectories/
│   ├── curves/
│   └── videos/
└── logs/
    └── eval.log
```

---

## Visualization

### Enable Visualization

Set in config or via command line:

```bash
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint <checkpoint> \
    --opts evaluation.visualization.enabled=true
```

### Visualization Options

**Trajectory Plots**: Predicted vs ground truth trajectories
**Success Curves**: Success plot (overlap threshold vs success rate)
**Precision Curves**: Precision plot (location error threshold vs precision)
**Tracking Videos**: Frame-by-frame tracking visualization with bounding boxes

**Output**: `outputs/<experiment>/visualization/`

---

## Ablation Studies

Evaluate ablation models using corresponding checkpoints:

```bash
# Without Vision-Mamba temporal module
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/ablation_no_mamba/checkpoints/checkpoint_best.pth

# Without geometric gating
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/ablation_no_gating/checkpoints/checkpoint_best.pth

# Without weak 3D cycle
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/ablation_no_cycle3d/checkpoints/checkpoint_best.pth
```

---

## Fast Evaluation

For quick validation on subset:

```bash
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint <checkpoint> \
    --opts debug.max_sequences=5
```

---

## Metric Interpretation

### Success AUC

- **Range**: [0, 1]
- **Higher is better**
- **Interpretation**:
  - > 0.6: Good tracking performance
  - > 0.7: Excellent tracking performance
  - < 0.4: Poor tracking, investigate failure modes

### Precision@20

- **Range**: [0, 1]
- **Higher is better**
- **Interpretation**: Fraction of frames with center error < 20 pixels

### Average Center Error

- **Range**: [0, ∞) pixels
- **Lower is better**
- **Interpretation**:
  - < 10 px: High precision
  - < 20 px: Acceptable precision
  - > 50 px: Poor localization

### Mean IoU

- **Range**: [0, 1]
- **Higher is better**
- **Interpretation**:
  - > 0.5: Good bounding box alignment
  - > 0.7: Excellent bounding box alignment

---

## Troubleshooting

### Out of Memory

Reduce batch size (should be 1 for tracking):

```bash
python scripts/eval.py --config <config> --opts evaluation.batch_size=1
```

### Missing Checkpoint

Ensure checkpoint path is correct:

```bash
ls outputs/train_uuv2/checkpoints/
```

### Slow Evaluation

Increase dataloader workers:

```bash
python scripts/eval.py --config <config> --opts data.num_workers=8
```

Disable visualization:

```bash
python scripts/eval.py --config <config> --opts evaluation.visualization.enabled=false
```

---

## Best Practices

✓ Use `checkpoint_best.pth` for final evaluation
✓ Disable augmentation for evaluation (default in eval configs)
✓ Run full evaluation on complete test set before reporting results
✓ Save visualizations for failure analysis
✓ Compare ablation results to validate component contributions

---

**Last Updated:** 2025-11-12