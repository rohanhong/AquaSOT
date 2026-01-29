# AquaSOT Ablation Guide

## Overview

AquaSOT uses **testing-time ablation** with a single checkpoint trained on both modalities. This follows academic best practices (OSTrack, MixFormer, ViPT) for fair comparison.

## Available Configurations

| Config | Description | Effect |
|--------|-------------|--------|
| `ablation_rgb_only` | RGB-only tracking | Sonar input zeroed |
| `ablation_sonar_only` | Sonar-only tracking | RGB input zeroed |
| `ablation_no_fusion` | No cross-modal fusion | IdentityFusion bypass |

## Running Ablation Experiments

```bash
# Full model (baseline)
python tracking/test.py aquasot vitb_256_dual \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/checkpoint_ep0100.pth

# RGB-only
python tracking/test.py aquasot ablation_rgb_only \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/checkpoint_ep0100.pth

# Sonar-only
python tracking/test.py aquasot ablation_sonar_only \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/checkpoint_ep0100.pth

# No fusion
python tracking/test.py aquasot ablation_no_fusion \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/checkpoint_ep0100.pth
```

## Interpreting Results

| Comparison | Insight |
|------------|---------|
| Full vs RGB-only | Sonar contribution |
| Full vs Sonar-only | RGB contribution |
| Full vs No-fusion | DCMA fusion contribution |
| No-fusion vs Single-modal | Dual-modal redundancy value |

## Paper Table Format

```
| Method          | AUC   | Precision | FPS  |
|-----------------|-------|-----------|------|
| RGB-only        | 0.xxx | 0.xxx     | xx.x |
| Sonar-only      | 0.xxx | 0.xxx     | xx.x |
| w/o Fusion      | 0.xxx | 0.xxx     | xx.x |
| Full Model      | 0.xxx | 0.xxx     | xx.x |
```

## Technical Notes

- Modality dropout applied in `AquaSOTTracker.init()` and `update()`
- Dropped modality produces low confidence score (~0)
- Target Loss Handler detects dropped modality as LOST
- Primary output excludes dropped modality automatically
- Default config has `ENABLED: false` for backward compatibility
