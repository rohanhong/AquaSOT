# AquaSOT-3D Evaluation Guide

## Basic Evaluation

```bash
python tracking/test.py --tracker aquasot3d \
    --checkpoint outputs/train/aquasot3d/vitb_256_dual/checkpoints/AquaSOT3D_ep0100.pth.tar
```

## Command Line Options

### Dataset Options

```bash
# Specify dataset path
python tracking/test.py --tracker aquasot3d \
    --checkpoint path/to/checkpoint.pth \
    --dataset_path /path/to/data

# Evaluate specific sequences
python tracking/test.py --tracker aquasot3d \
    --checkpoint path/to/checkpoint.pth \
    --sequences uuv3 uuv6 uuv10
```

### Output Options

```bash
# Custom result directory
python tracking/test.py --tracker aquasot3d \
    --checkpoint path/to/checkpoint.pth \
    --result_dir results/experiment_001 \
    --report_dir reports/experiment_001
```

### Visualization

```bash
# Enable visualization
python tracking/test.py --tracker aquasot3d \
    --checkpoint path/to/checkpoint.pth \
    --visualize

# Plot success/precision curves
python tracking/test.py --tracker aquasot3d \
    --checkpoint path/to/checkpoint.pth \
    --plot_curves
```

### Additional Options

```bash
# Overwrite existing results
python tracking/test.py --tracker aquasot3d \
    --checkpoint path/to/checkpoint.pth \
    --overwrite

# Generate report from existing results only
python tracking/test.py --tracker aquasot3d \
    --report_only

# Custom model parameters
python tracking/test.py --tracker aquasot3d \
    --checkpoint path/to/checkpoint.pth \
    --search_size 256 \
    --template_size 128 \
    --search_factor 4.0 \
    --template_factor 2.0
```

## Output Structure

```
results/
    AquaSOT3D/
        uuv3.txt              # Tracking predictions [x,y,w,h]
        uuv3_time.txt         # Per-frame timing
        uuv6.txt
        uuv6_time.txt
        ...

reports/
    performance.json          # Detailed metrics
    curves.png                # Success/Precision curves (if --plot_curves)
```

## Metrics

| Metric | Description |
|--------|-------------|
| Success AUC | Area under success curve (IoU thresholds 0-1) |
| Precision@20 | Precision at 20-pixel center error threshold |
| Average IoU | Mean IoU across all frames |
| Center Error | Mean Euclidean distance between predicted and GT centers |

## Expected Console Output

```
[Evaluation] Dataset: aquasot3d
[Evaluation] Dataset path: /path/to/data
[Evaluation] Found 12 sequences
[Evaluation] Tracker: AquaSOT3D

======================================================================
Evaluation Summary
======================================================================
Tracker              Success AUC    Precision@20      Avg IoU
----------------------------------------------------------------------
AquaSOT3D                  0.652           0.823        0.587
======================================================================

[Results] AquaSOT3D:
  Success AUC:  0.652
  Precision@20: 0.823
  Average IoU:  0.587
  Center Error: 15.3 px
```

## Dataset Structure

Evaluation uses rect-converted sonar data:
- RGB: `light/img/` + `light/groundtruth.txt`
- Sonar: `sonar/img_rect/` + `sonar/groundtruth_rect.txt`

Run sonar preprocessing first if `img_rect/` does not exist:
```bash
python scripts/preprocess_sonar.py --all --data-root data
```
