# AquaSOT Ablation Guide

## Running Ablation Experiments

```bash
# RGB-only
python tracking/test.py --tracker aquasot \
    --config experiments/aquasot/ablation_rgb_only.yaml \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --result_dir outputs/test/ablation_rgb_only/results \
    --report_dir outputs/test/ablation_rgb_only/reports

# Sonar-only
python tracking/test.py --tracker aquasot \
    --config experiments/aquasot/ablation_sonar_only.yaml \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --result_dir outputs/test/ablation_sonar_only/results \
    --report_dir outputs/test/ablation_sonar_only/reports

# No fusion
python tracking/test.py --tracker aquasot \
    --config experiments/aquasot/ablation_no_fusion.yaml \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --result_dir outputs/test/ablation_no_fusion/results \
    --report_dir outputs/test/ablation_no_fusion/reports
```

## Visualization

```bash
python scripts/visualize_results.py \
    --result_dir outputs/test/ablation_rgb_only/results/AquaSOT \
    --output_dir outputs/visualization/ablation_rgb_only

python scripts/visualize_results.py \
    --result_dir outputs/test/ablation_sonar_only/results/AquaSOT \
    --output_dir outputs/visualization/ablation_sonar_only

python scripts/visualize_results.py \
    --result_dir outputs/test/ablation_no_fusion/results/AquaSOT \
    --output_dir outputs/visualization/ablation_no_fusion
```

## Output Directory Structure

```
outputs/test/
    baseline/
        results/
        reports/
    ablation_rgb_only/
        results/
        reports/
    ablation_sonar_only/
        results/
        reports/
    ablation_no_fusion/
        results/
        reports/
```

## Paper Table Format

```
| Method     | AUC   | Precision | FPS  |
|------------|-------|-----------|------|
| RGB-only   | 0.xxx | 0.xxx     | xx.x |
| Sonar-only | 0.xxx | 0.xxx     | xx.x |
| w/o Fusion | 0.xxx | 0.xxx     | xx.x |
| Full Model | 0.xxx | 0.xxx     | xx.x |
```