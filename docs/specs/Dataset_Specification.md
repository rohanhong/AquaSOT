# AquaSOT-3D Dataset Specification

## Overview

The AquaSOT-3D dataset is a multi-modal underwater object tracking benchmark featuring synchronized RGB camera and forward-looking sonar (FLS) imagery. The dataset captures underwater unmanned vehicles (UUVs) in challenging aquatic environments for multi-object tracking research.

**Dataset Split:**
- **Training set**: `uuv1`, `uuv2`, `uuv4`, `uuv5`, `uuv7`, `uuv8`, `uuv9`, `uuv11`, `uuv12` (9 sequences, 9,588 frames)
- **Validation/Evaluation set**: `uuv3`, `uuv6`, `uuv10` (3 sequences, 2,103 frames)
- **Total**: 12 sequences, 11,691 frames

## Directory Structure

```
data/
├── uuv1/                    # Training set (1,757 frames)
│   ├── light/               # RGB camera modality
│   │   ├── img/             # Image sequences (1.jpg - N.jpg)
│   │   └── groundtruth.txt  # Bounding box annotations
│   └── sonar/               # FLS sonar modality
│       ├── img/             # Image sequences (1.jpg - N.jpg)
│       └── groundtruth.txt  # Bounding box annotations
├── uuv2/                    # Training set (2,740 frames)
│   └── ...
├── uuv3/                    # Validation/Evaluation set (751 frames)
│   └── ...
├── uuv4/                    # Training set (691 frames)
│   └── ...
├── uuv5/                    # Training set (881 frames)
│   └── ...
├── uuv6/                    # Validation/Evaluation set (751 frames)
│   └── ...
├── uuv7/                    # Training set (601 frames)
│   └── ...
├── uuv8/                    # Training set (455 frames)
│   └── ...
├── uuv9/                    # Training set (901 frames)
│   └── ...
├── uuv10/                   # Validation/Evaluation set (601 frames)
│   └── ...
├── uuv11/                   # Training set (661 frames)
│   └── ...
└── uuv12/                   # Training set (901 frames)
    └── ...
```

## Modality Characteristics

### RGB Camera (Light)
- **Resolution:** 1920 × 1080 pixels (Full HD)
- **Format:** JPEG images, RGB color space
- **Characteristics:**
  - Underwater optical imagery with natural lighting
  - Subject to water turbidity and lighting variations
  - Standard rectangular field of view

### Forward-Looking Sonar (FLS)
- **Resolution:** Variable across sequences (~680×375 pixels typical)
  - 682×376 pixels: `uuv2`, `uuv3`, `uuv4`, `uuv6`, `uuv7`, `uuv8`, `uuv9` (7 sequences)
  - 684×377 pixels: `uuv5`, `uuv12` (2 sequences)
  - 680×375 pixels: `uuv10` (1 sequence)
  - 909×501 pixels: `uuv11` (1 sequence)
  - 912×503 pixels: `uuv1` (1 sequence)
- **Format:** JPEG images, RGB color space (grayscale visualized)
- **Characteristics:**
  - **Fan-shaped field of view** (sector scan pattern)
  - Non-uniform spatial resolution (radial vs. angular)
  - Low signal-to-noise ratio with acoustic artifacts
  - Black background regions outside effective scanning area
  - Targets appear as bright reflective regions

**Important:** Sonar images exhibit a characteristic fan/sector shape due to the acoustic beam pattern. The spatial resolution is not uniform - angular resolution decreases with range while radial resolution remains relatively constant. Resolution variations across sequences are minor (within ±2% of base ~680×376) and are handled adaptively by the model architecture.

## Annotation Format

Annotations are stored in `groundtruth.txt` files with the following format:

```
x,y,w,h
x,y,w,h
...
```

**Format Specification:**
- Each line corresponds to one frame (line number = frame ID)
- Four comma-separated integers per line:
  - `x`: Top-left x-coordinate of bounding box
  - `y`: Top-left y-coordinate of bounding box
  - `w`: Width of bounding box
  - `h`: Height of bounding box

**Special Case - Occlusion/Invisibility:**
- `0,0,0,0`: Indicates target is not visible or occluded in the current frame
- Used when target temporarily leaves field of view or is obscured

**Example:**
```
122,359,1356,254    # Frame 1: Target at (122,359) with size 1356×254
114,357,1354,257    # Frame 2: Target at (114,357) with size 1354×257
0,0,0,0             # Frame 3: Target not visible
```

## Dataset Statistics

### Complete Sequence Overview

| Sequence | Split        | Light Frames | Sonar Frames | Light Resolution | Sonar Resolution | Frame Sync |
|----------|--------------|--------------|--------------|------------------|------------------|------------|
| uuv1     | Train        | 1,757        | 1,757        | 1920×1080        | 912×503          | ✓          |
| uuv2     | Train        | 2,740        | 2,740        | 1920×1080        | 682×376          | ✓          |
| uuv3     | Val/Eval     | 751          | 751          | 1920×1080        | 682×376          | ✓          |
| uuv4     | Train        | 691          | 691          | 1920×1080        | 682×376          | ✓          |
| uuv5     | Train        | 881          | 881          | 1920×1080        | 684×377          | ✓          |
| uuv6     | Val/Eval     | 751          | 751          | 1920×1080        | 682×376          | ✓          |
| uuv7     | Train        | 601          | 601          | 1920×1080        | 682×376          | ✓          |
| uuv8     | Train        | 455          | 455          | 1920×1080        | 682×376          | ✓          |
| uuv9     | Train        | 901          | 901          | 1920×1080        | 682×376          | ✓          |
| uuv10    | Val/Eval     | 601          | 601          | 1920×1080        | 680×375          | ✓          |
| uuv11    | Train        | 661          | 661          | 1920×1080        | 909×501          | ✓          |
| uuv12    | Train        | 901          | 901          | 1920×1080        | 684×377          | ✓          |
| **Total**| **-**        | **11,691**   | **11,691**   | **-**            | **-**            | **✓**      |

**Split Summary:**
- **Training**: 9 sequences, 9,588 frames total
- **Validation/Evaluation**: 3 sequences, 2,103 frames total

### Representative Bounding Box Statistics

The following statistics are from `uuv2` (largest training set) and `uuv3` (representative validation set). Bounding box statistics for other sequences can be generated using the `tests/test_dataset_statistics.py` script.

#### UUV2 (Training Set - 2,740 frames)

**RGB Bounding Box Statistics:**
- Width: mean=335.1, std=498.4, range=[0, 1920]
- Height: mean=109.5, std=149.8, range=[0, 636]
- Center X: mean=341.5, std=584.9
- Center Y: mean=191.6, std=215.4

**Sonar Bounding Box Statistics:**
- Width: mean=49.0, std=4.6, range=[34, 71]
- Height: mean=24.5, std=6.3, range=[10, 44]
- Center X: mean=289.9, std=100.1
- Center Y: mean=183.2, std=82.8

#### UUV3 (Validation Set - 751 frames)

**RGB Bounding Box Statistics:**
- Width: mean=280.8, std=273.3, range=[0, 767]
- Height: mean=90.0, std=90.5, range=[0, 316]
- Center X: mean=421.1, std=575.5
- Center Y: mean=260.6, std=226.1

**Sonar Bounding Box Statistics:**
- Width: mean=47.3, std=5.2, range=[31, 59]
- Height: mean=30.0, std=8.6, range=[10, 51]
- Center X: mean=309.7, std=91.6
- Center Y: mean=165.0, std=37.6

**Note:** High variance in RGB statistics is due to occlusion annotations (0,0,0,0). Sonar annotations show significantly more stable bounding box dimensions.

## Key Observations

1. **Temporal Synchronization:** RGB and sonar sequences are frame-synchronized (same frame count per sequence)

2. **Modality Complementarity:**
   - RGB provides high-resolution visual details but suffers from water turbidity
   - Sonar provides robust range-bearing information independent of lighting conditions
   - Sonar bounding boxes show significantly lower variance (more stable tracking)

3. **Annotation Characteristics:**
   - RGB annotations show high variance due to frequent occlusions (0,0,0,0)
   - Sonar annotations are more consistent with tighter bounding boxes
   - Both modalities track the same physical target with temporal alignment

4. **Sonar Spatial Properties:**
   - Fan-shaped coverage area (sector scan)
   - Resolution degrades with increasing range (distance from sensor)
   - Angular resolution varies across the field of view

5. **Challenge Factors:**
   - Underwater turbidity and lighting variations in RGB
   - Low SNR and acoustic noise in sonar
   - Frequent target occlusions in both modalities
   - Scale variation due to target motion

## Usage Notes

### Configuration in Training/Evaluation

To use the dataset split in configuration files:

```yaml
dataset:
  data_root: "data"
  train_sequences:
    - "uuv1"
    - "uuv2"
    - "uuv4"
    - "uuv5"
    - "uuv7"
    - "uuv8"
    - "uuv9"
    - "uuv11"
    - "uuv12"

  val_sequences:
    - "uuv3"
    - "uuv6"
    - "uuv10"
```

### General Usage Notes

- Frame indexing starts at 1 (not 0)
- Both modalities must be processed jointly for optimal tracking performance
- Handle 0,0,0,0 annotations appropriately during training/evaluation
- Account for sonar's non-uniform spatial resolution in feature extraction
- Consider modality-specific preprocessing (e.g., denoising for sonar)
- Model architecture handles variable sonar resolutions adaptively

## Data Integrity

All sequences have been validated for:
- ✓ Consistent frame counts between RGB and sonar modalities (no mismatches detected)
- ✓ Matching annotation lengths with frame counts (groundtruth lines = frame count)
- ✓ Valid image formats and readability (all images load successfully)
- ✓ Bounding box coordinates within image boundaries (excluding occlusion cases)
- ✓ No trailing frame discrepancies between modalities

### Statistics Generation

To regenerate dataset statistics:

```bash
python3 tests/test_dataset_statistics.py
```

This script generates:
- Console report with per-sequence frame counts and resolutions
- Detection of frame count mismatches or inconsistencies
- JSON output file: `dataset_statistics.json`

## Update History

- **2025-11-13**: Dataset expanded from 2 to 12 sequences; defined train/val/eval splits (uuv3/6/10 as validation); documented sonar resolution variations; verified complete frame synchronization across all sequences
