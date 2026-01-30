# AquaSOT Testing

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

## Ablation Studies

```bash
# RGB-only evaluation (requires --bn_adapt for best results)
python tracking/test.py --checkpoint outputs/train/checkpoints/model.pth \
    --eval_mode rgb_only --bn_adapt

# Sonar-only evaluation
python tracking/test.py --checkpoint outputs/train/checkpoints/model.pth \
    --eval_mode sonar_only --bn_adapt

# No-fusion evaluation (both modalities, fusion weights = 0)
python tracking/test.py --checkpoint outputs/train/checkpoints/model.pth \
    --eval_mode no_fusion --bn_adapt
```

**Note**: BN adaptation (`--bn_adapt`) is recommended for ablation studies because training uses fusion weights of 1.0 while inference uses 0.3. This causes feature distribution shift that BN adaptation helps compensate.

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--sequences` | Test set (uuv3) | Sequences to evaluate |
| `--result_dir` | `outputs/test/baseline/results` | Results output directory |
| `--report_dir` | `outputs/test/baseline/reports` | Reports output directory |
| `--visualize` | On | Generate visualization |
| `--plot_curves` | On | Plot performance curves |
| `--eval_mode` | `dual` | Evaluation mode: `dual`, `rgb_only`, `sonar_only`, `no_fusion` |
| `--bn_adapt` | Off | Enable BN adaptation for ablation studies |
| `--overwrite` | Off | Overwrite existing results |

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