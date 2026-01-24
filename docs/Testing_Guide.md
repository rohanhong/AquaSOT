# AquaSOT Testing Guide

## Basic Testing

```bash
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/AquaSOT_ep0100.pth.tar
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tracker` | Tracker name | `aquasot` |
| `--checkpoint` | Model checkpoint path | Required |
| `--dataset_path` | Dataset root directory | `data/` |
| `--sequences` | Specific sequences to evaluate | All |
| `--result_dir` | Tracking results output directory | `outputs/test/results` |
| `--report_dir` | Evaluation reports directory | `outputs/test/reports` |
| `--visualize` | Enable real-time visualization | Off |
| `--plot_curves` | Generate success/precision curves | Off |
| `--overwrite` | Overwrite existing results | Off |
| `--report_only` | Generate report from existing results | Off |
| `--search_size` | Search region size | 256 |
| `--template_size` | Template size | 128 |
| `--search_factor` | Search region scale factor | 4.0 |
| `--template_factor` | Template scale factor | 2.0 |

**Full evaluation with visualization:**
```bash
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/AquaSOT_ep0100.pth.tar \
    --visualize --plot_curves
```

**Evaluate specific sequences only:**
```bash
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/AquaSOT_ep0100.pth.tar \
    --sequences uuv3 uuv6 uuv10
```

## Output Structure

```
outputs/
    test/
        results/
            AquaSOT/
                uuv3.txt              # Tracking predictions [x,y,w,h]
                uuv3_time.txt         # Per-frame timing
                uuv6.txt
                ...
        reports/
            performance.json          # Detailed metrics
            curves.png                # Success/Precision curves (if --plot_curves)
    visualization/
        uuv3.mp4                      # Combined RGB + Sonar video
        uuv6.mp4
        ...
```

## Metrics

| Metric | Description |
|--------|-------------|
| Success AUC | Area under success curve (IoU thresholds 0-1) |
| Precision@20 | Precision at 20-pixel center error threshold |
| Average IoU | Mean IoU across all frames |
| Center Error | Mean Euclidean distance between predicted and GT centers |

## Visualize Results

Generate side-by-side RGB and Sonar visualization with GT (green) and Pred (red) boxes:

```bash
# Generate videos
python scripts/visualize_results.py \
    --result_dir outputs/test/results/AquaSOT \
    --data_root data \
    --output_dir outputs/visualization

# Generate images for specific sequences
python scripts/visualize_results.py \
    --result_dir outputs/test/results/AquaSOT \
    --data_root data \
    --format images \
    --sequences uuv3 uuv6
```

| Option | Description | Default |
|--------|-------------|---------|
| `--result_dir` | Tracking results directory | Required |
| `--data_root` | Dataset root directory | `data/` |
| `--output_dir` | Output directory | `outputs/visualization/` |
| `--sequences` | Specific sequences | All |
| `--format` | Output format: `video` or `images` | `video` |
| `--fps` | Video frame rate | 10 |
| `--height` | Output frame height | 480 |

## Dataset Structure

Evaluation uses rect-converted sonar data:
- RGB: `light/img/` + `light/groundtruth.txt`
- Sonar: `sonar/img_rect/` + `sonar/groundtruth_rect.txt`

Run sonar preprocessing first if `img_rect/` does not exist:
```bash
python scripts/preprocess_sonar.py --all --data-root data
```
