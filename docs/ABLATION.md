# AquaSOT Ablation

## Running Ablation Experiments

```bash
# RGB-only
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --eval_mode rgb_only --bn_adapt \
    --result_dir outputs/test/ablation_rgb_only/results \
    --report_dir outputs/test/ablation_rgb_only/reports

# Sonar-only
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --eval_mode sonar_only --bn_adapt \
    --result_dir outputs/test/ablation_sonar_only/results \
    --report_dir outputs/test/ablation_sonar_only/reports

# No fusion
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth \
    --eval_mode no_fusion --bn_adapt \
    --result_dir outputs/test/ablation_no_fusion/results \
    --report_dir outputs/test/ablation_no_fusion/reports
```

## BatchNorm Adaptation

When running ablation experiments, BatchNorm statistics learned during dual-modal training
may not match the ablated feature distribution. The `--bn_adapt` flag recalculates BN
running_mean/running_var before evaluation.

| Mode | BN Adaptation | Notes |
|------|---------------|-------|
| Dual-modal | Not needed | Baseline configuration |
| RGB-only | Required | Ensures fair comparison |
| Sonar-only | Required | Ensures fair comparison |
| No-fusion | Required | Ensures fair comparison |

## Expected Results

| Method     | RGB AUC | Sonar AUC | Overall AUC |
|------------|---------|-----------|-------------|
| Dual-modal | -       | -         | -           |
| RGB-only   | -       | N/A       | -           |
| Sonar-only | N/A     | -         | -           |
| No-fusion  | -       | -         | -           |

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
| Method     | AUC | Precision | Notes                             |
|------------|-----|-----------|-----------------------------------|
| RGB-only   | -   | -         | --eval_mode rgb_only --bn_adapt   |
| Sonar-only | -   | -         | --eval_mode sonar_only --bn_adapt |
| w/o Fusion | -   | -         | --eval_mode no_fusion --bn_adapt  |
| Full Model | -   | -         | Baseline                          |
```
