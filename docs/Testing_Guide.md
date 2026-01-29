# AquaSOT Testing Guide

## Evaluation

```bash
# Basic evaluation (test set UUV3, with visualization and curves)
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth

# Evaluate specific sequences
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --sequences uuv3 uuv6 uuv10

# Disable visualization
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --no_visualize --no_plot_curves
```

## Command Line Options

| Option | Default |
|--------|---------|
| `--checkpoint` | Required |
| `--sequences` | Test set (uuv3) |
| `--result_dir` | `outputs/test/baseline/results` |
| `--report_dir` | `outputs/test/baseline/reports` |
| `--visualize` | On |
| `--plot_curves` | On |
| `--eval_mode` | `dual` (`rgb_only`, `sonar_only`) |
| `--overwrite` | Off |

## Visualization

```bash
python scripts/visualize_results.py \
    --result_dir outputs/test/baseline/results/AquaSOT \
    --output_dir outputs/visualization/baseline
```

## Export Checkpoint

```bash
# Export lightweight inference checkpoint
python scripts/export_checkpoint.py \
    --input outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --output models/aquasot_vitb_256.pth
```

## Output Structure

```
outputs/test/baseline/
    results/
        AquaSOT/
            uuv3_rgb.txt
            uuv3_rgb_scores.txt
            uuv3_sonar.txt
            uuv3_sonar_scores.txt
            uuv3_time.txt
    reports/
        performance.json
        curves.png
```