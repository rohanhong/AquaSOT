# AquaSOT-3D Evaluation Guide

**Version:** 2.0
**Date:** 2025-11-14
**Multi-Sequence Architecture**

---

## Overview

AquaSOT-3D supports **multi-sequence evaluation** across the full AquaSOT-3D dataset (12 sequences):

- **Test set**: `uuv3, uuv6, uuv10` (3 sequences, official benchmark)
- **Single-sequence evaluation**: `uuv3` (for UUV3-specific workflows)

### Evaluation Modes

AquaSOT-3D evaluation follows **RGBS50 benchmark protocol** with three evaluation modes:

1. **RGB Tracking**: RGB-only performance assessment
2. **Sonar Tracking**: Sonar-only performance assessment
3. **Fusion Tracking**: Multi-modal fusion performance assessment

**Metrics**: Success (AUC), Precision@20, Average Center Error, Mean IoU

---

## Quick Start

### Multi-Sequence Evaluation (Recommended)

Evaluate on all 3 test sequences:

```bash
# Full multi-modal model
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/checkpoint_best.pth
```

### Single-Sequence Evaluation (For Debugging)

Evaluate on UUV3 only:

```bash
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/train_uuv2/checkpoints/checkpoint_best.pth
```

### Single-Modality Evaluation

```bash
# RGB-only model (multi-sequence)
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/pretrain_rgb/checkpoints/checkpoint_best.pth

# Sonar-only model (multi-sequence)
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/pretrain_sonar/checkpoints/checkpoint_best.pth
```

---

## Configuration Files

All configurations use the **multi-sequence format**:

```yaml
data:
  data_root: data
  test_sequences: [uuv3, uuv6, uuv10]
```

Available configs in `configs/eval/` directory:

```
configs/eval/
├── eval_multiseq.yaml          # Multi-sequence: All 3 test sequences
└── eval_uuv3.yaml              # Single sequence: UUV3 only (debugging)
```

### Per-Sequence Resolution Detection

The system automatically detects RGB and Sonar resolutions for each test sequence:

```bash
[INFO] Multi-sequence mode: data_root=data
[INFO] Test sequences: ['uuv3', 'uuv6', 'uuv10']
[INFO]   Test sequence 'uuv3': 1893 valid frames, RGB (1920, 1080), Sonar (682, 376)
[INFO]   Test sequence 'uuv6': 2156 valid frames, RGB (1920, 1080), Sonar (682, 376)
[INFO]   Test sequence 'uuv10': 1987 valid frames, RGB (1920, 1080), Sonar (682, 376)
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
# Multi-sequence evaluation with custom settings
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/checkpoint_best.pth \
    --opts evaluation.batch_size=1 evaluation.visualization.enabled=true
```

### Custom Output Directory

```bash
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/checkpoint_best.pth \
    --output outputs/custom_eval
```

### Evaluate Specific Sequences

```bash
# Override to evaluate only specific test sequences
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/checkpoint_best.pth \
    --opts data.test_sequences=[uuv3,uuv6]
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

### Multi-Sequence Evaluation

Evaluation results saved to `outputs/<experiment>/`:

```
outputs/eval_multiseq/
├── overall_metrics.json        # Aggregated metrics across all test sequences
├── evaluation_report.txt       # Human-readable summary
├── eval_config.yaml           # Config snapshot
├── sequences/                 # Per-sequence results
│   ├── uuv3/
│   │   ├── predictions.npz    # Per-frame predictions
│   │   ├── metrics.json       # Sequence metrics (Success, Precision, etc.)
│   │   └── metadata.json      # Sequence metadata (frame count, resolutions)
│   ├── uuv6/
│   │   ├── predictions.npz
│   │   ├── metrics.json
│   │   └── metadata.json
│   └── uuv10/
│       ├── predictions.npz
│       ├── metrics.json
│       └── metadata.json
├── visualization/             # Visualizations (if enabled)
│   ├── trajectories/
│   │   ├── uuv3_trajectory.png
│   │   ├── uuv6_trajectory.png
│   │   └── uuv10_trajectory.png
│   ├── curves/
│   │   ├── success_plot.png
│   │   └── precision_plot.png
│   └── videos/
│       ├── uuv3_tracking.mp4
│       ├── uuv6_tracking.mp4
│       └── uuv10_tracking.mp4
└── logs/
    └── eval.log
```

### Single-Sequence Evaluation

```
outputs/eval_uuv3/
├── overall_metrics.json        # Metrics for single sequence
├── evaluation_report.txt
├── eval_config.yaml
├── sequences/
│   └── uuv3/
│       ├── predictions.npz
│       ├── metrics.json
│       └── metadata.json
├── visualization/
└── logs/
    └── eval.log
```

---

## Visualization

### Default Behavior

**Visualization is enabled by default** in all evaluation configs (`eval_multiseq.yaml`, `eval_uuv3.yaml`).

The evaluation script will automatically generate visualizations and save them to the output directory unless explicitly disabled.

### Control Visualization

Use CLI flags to override config settings:

```bash
# Use default (visualization enabled)
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint>

# Explicitly disable for faster evaluation
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint> \
    --no-visualize

# Force enable (override config if disabled)
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint> \
    --visualize

# Alternative: via config override
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint> \
    --opts evaluation.visualization.enabled=false
```

### Visualization Outputs

When enabled, the following visualizations are generated:

**Trajectory Plots**: Predicted vs ground truth trajectories (per-sequence)
**Success Curves**: Success plot (overlap threshold vs success rate) (aggregated)
**Precision Curves**: Precision plot (location error threshold vs precision) (aggregated)
**Tracking Videos**: Frame-by-frame tracking visualization with bounding boxes (per-sequence, if configured)
**Failure Analysis**: Best/worst sequences for debugging

**Output Location**: `outputs/<experiment>/visualization/`

For multi-sequence evaluation, visualizations are generated for each test sequence independently, plus aggregated metric curves.

### Runtime Impact

- **With visualization**: ~20-30% slower (depending on visualization settings)
- **Without visualization**: Faster, metrics-only evaluation
- Use `--no-visualize` for quick metric computation or CI/CD pipelines

---

## Ablation Studies

### Multi-Sequence Ablation (Recommended)

Evaluate ablation models on all test sequences:

```bash
# Without Vision-Mamba temporal module
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/ablation_no_mamba/checkpoints/checkpoint_best.pth

# Without geometric gating
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/ablation_no_gating/checkpoints/checkpoint_best.pth

# Without weak 3D cycle
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/ablation_no_cycle3d/checkpoints/checkpoint_best.pth
```

### Single-Sequence Ablation (For Debugging)

```bash
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/ablation_no_mamba/checkpoints/checkpoint_best.pth
```

---

## Fast Evaluation

### Quick Multi-Sequence Validation

For quick validation on subset of frames per sequence:

```bash
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint> \
    --opts debug.max_frames_per_sequence=100
```

### Evaluate Subset of Sequences

```bash
# Evaluate only first test sequence
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint> \
    --opts data.test_sequences=[uuv3]
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

Evaluate sequences one at a time:

```bash
# Evaluate only UUV3
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint> \
    --opts data.test_sequences=[uuv3]

# Evaluate only UUV6
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint> \
    --opts data.test_sequences=[uuv6]
```

### Missing Checkpoint

Ensure checkpoint path is correct:

```bash
# For multi-sequence training
ls outputs/train_multiseq/checkpoints/

# For single-sequence training
ls outputs/train_uuv2/checkpoints/
```

### Slow Evaluation

Increase dataloader workers:

```bash
python scripts/eval.py --config <config> --opts data.num_workers=8
```

Disable visualization (recommended for speed):

```bash
# Preferred method: Use CLI flag
python scripts/eval.py --config <config> --checkpoint <checkpoint> --no-visualize

# Alternative: Config override
python scripts/eval.py --config <config> --checkpoint <checkpoint> --opts evaluation.visualization.enabled=false
```

Use frame subset for quick validation:

```bash
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint <checkpoint> \
    --opts debug.max_frames_per_sequence=100
```

---

## Best Practices

✓ Use multi-sequence evaluation (`eval_multiseq.yaml`) for final benchmark results
✓ Use `checkpoint_best.pth` for final evaluation
✓ Disable augmentation for evaluation (default in eval configs)
✓ Run full evaluation on complete test set (all 3 sequences) before reporting results
✓ Save visualizations for failure analysis
✓ Compare ablation results to validate component contributions
✓ Report per-sequence metrics in addition to aggregated metrics

---

**Last Updated:** 2025-11-14