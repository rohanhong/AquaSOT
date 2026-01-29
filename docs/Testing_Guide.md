# AquaSOT Testing Guide

## Basic Testing

```bash
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/checkpoint_ep0100.pth
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
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/checkpoint_ep0100.pth \
    --visualize --plot_curves
```

**Evaluate validation set (uuv3):**
```bash
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/aquasot/vitb_256_dual/checkpoints/checkpoint_ep0100.pth \
    --sequences uuv3
```

## Output Structure

```
outputs/
    test/
        results/
            AquaSOT/
                uuv3_rgb.txt          # RGB predictions [x,y,w,h]
                uuv3_sonar.txt        # Sonar predictions [x,y,w,h]
                uuv3_rgb_scores.txt   # RGB confidence scores
                uuv3_sonar_scores.txt # Sonar confidence scores
                uuv3_time.txt         # Per-frame timing
                uuv6_rgb.txt
                ...
        reports/
            performance.json          # Detailed metrics
            curves.png                # Success/Precision curves (if --plot_curves)
    visualization/
        uuv3.mp4                      # Combined RGB + Sonar video
        uuv6.mp4
        ...
```

The tracker outputs dual-modal predictions:
- `{seq}_rgb.txt`: RGB predictions
- `{seq}_sonar.txt`: Sonar predictions

## Metrics

| Metric | Description |
|--------|-------------|
| Success AUC | Area under success curve (IoU thresholds 0-1) |
| Precision@20 | Precision at 20-pixel center error threshold |
| Average IoU | Mean IoU across all frames |
| Center Error | Mean Euclidean distance between predicted and GT centers |

## Visualize Results

Generate side-by-side RGB and Sonar visualization:
- Ground truth boxes shown in **green**
- RGB predictions shown in **red** on RGB image
- Sonar predictions shown in **red** on Sonar image

```bash
# Generate videos
python scripts/visualize_results.py \
    --result_dir outputs/test/results/AquaSOT \
    --data_root data \
    --output_dir outputs/visualization

# Generate images for validation sequence
python scripts/visualize_results.py \
    --result_dir outputs/test/results/AquaSOT \
    --data_root data \
    --format images \
    --sequences uuv3
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

## Template Update (MixFormer Strategy)

The tracker uses MixFormer-style three-tier template management for robust tracking:

1. **First-frame template**: Fixed, never updated
2. **Best-frame cache**: Highest confidence frame in current interval
3. **Online template queue**: FIFO collection of recent best frames

### Configuration

Template update is enabled by default. Key parameters in config:

```yaml
TEST:
  TEMPLATE_UPDATE:
    ENABLED: true           # Enable/disable template update
    STRATEGY: mixformer     # Update strategy
    ONLINE_SIZE: 2          # Number of online templates in queue
    UPDATE_INTERVAL: 200    # Frames between queue updates
    SCORE_THRESHOLD: 0.5    # Minimum confidence for update
    MAX_SCORE_DECAY: 1.0    # Score decay rate (1.0 = no decay)
```

To disable template update (use first-frame only):
```yaml
TEST:
  TEMPLATE_UPDATE:
    ENABLED: false
```

## Checkpoint Formats

### Training Checkpoint (Full)

Training produces full checkpoints containing:

| Key | Content | Purpose |
|-----|---------|---------|
| `net` | Model weights | Required for inference |
| `optimizer` | Optimizer state (momentum, Adam buffers) | Resume training |
| `lr_scheduler` | Learning rate scheduler state | Resume training |
| `epoch` | Training epoch number | Resume training |
| `stats` | Training statistics | Logging |

File naming: `checkpoint_ep{epoch:04d}.pth`, `checkpoint_latest.pth`

### Inference Checkpoint (Lightweight)

For deployment and evaluation, export inference-only checkpoint to `models/`:

```bash
# Export to models/ directory
python scripts/export_checkpoint.py \
    --input outputs/train/aquasot/vitb_256_dual/checkpoints/checkpoint_ep0100.pth \
    --output models/aquasot_vitb_256.pth
```

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input training checkpoint path | - |
| `--output` | Output inference checkpoint path | - |
| `--input_dir` | Input directory for batch export | - |
| `--output_dir` | Output directory for batch export | - |
| `--format` | Output format: `dict` or `raw` | `dict` |
| `--latest_only` | Only export checkpoint_latest.pth | Off |
| `--strip_ddp` | Remove DDP "module." prefix | On |

**Size comparison** (estimated):
- Training checkpoint: ~200-250MB (includes optimizer state)
- Inference checkpoint: ~100MB (model weights only)

### Supported Loading Formats

The tracker automatically detects and loads these formats:

```python
# Format 1: dict with 'net' key (recommended)
{'net': state_dict}

# Format 2: dict with 'state_dict' key
{'state_dict': state_dict}

# Format 3: raw state_dict
state_dict
```

Both training checkpoints and exported inference checkpoints work with `tracking/test.py`.

## Dataset Structure

Evaluation uses rect-converted sonar data:
- RGB: `light/img/` + `light/groundtruth.txt`
- Sonar: `sonar/img_rect/` + `sonar/groundtruth_rect.txt`

Run sonar preprocessing first if `img_rect/` does not exist:
```bash
python scripts/preprocess_sonar.py --all --data-root data
```
