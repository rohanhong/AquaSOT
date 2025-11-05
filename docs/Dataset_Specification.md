# AquaSOT-3D Dataset Specification

## Overview

The AquaSOT-3D dataset is a multi-modal underwater object tracking benchmark featuring synchronized RGB camera and forward-looking sonar (FLS) imagery. The dataset captures underwater unmanned vehicles (UUVs) in challenging aquatic environments for multi-object tracking research.

**Dataset Split:**
- `uuv2`: Training set (2,740 frames)
- `uuv3`: Validation/Test set (751 frames)

## Directory Structure

```
data/
├── uuv2/                    # Training set
│   ├── light/               # RGB camera modality
│   │   ├── img/             # Image sequences (1.jpg - 2740.jpg)
│   │   └── groundtruth.txt  # Bounding box annotations
│   └── sonar/               # FLS sonar modality
│       ├── img/             # Image sequences (1.jpg - 2740.jpg)
│       └── groundtruth.txt  # Bounding box annotations
└── uuv3/                    # Validation/Test set
    ├── light/               # RGB camera modality
    │   ├── img/             # Image sequences (1.jpg - 751.jpg)
    │   └── groundtruth.txt  # Bounding box annotations
    └── sonar/               # FLS sonar modality
        ├── img/             # Image sequences (1.jpg - 751.jpg)
        └── groundtruth.txt  # Bounding box annotations
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
- **Resolution:** 682 × 376 pixels
- **Format:** JPEG images, RGB color space (grayscale visualized)
- **Characteristics:**
  - **Fan-shaped field of view** (sector scan pattern)
  - Non-uniform spatial resolution (radial vs. angular)
  - Low signal-to-noise ratio with acoustic artifacts
  - Black background regions outside effective scanning area
  - Targets appear as bright reflective regions

**Important:** Sonar images exhibit a characteristic fan/sector shape due to the acoustic beam pattern. The spatial resolution is not uniform - angular resolution decreases with range while radial resolution remains relatively constant.

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

### UUV2 (Training Set)

| Modality | Frames | Resolution | Bbox Width | Bbox Height | Notes |
|----------|--------|------------|------------|-------------|-------|
| RGB      | 2,740  | 1920×1080  | 335±498 px | 110±150 px  | High variance due to occlusions |
| Sonar    | 2,740  | 682×376    | 49±5 px    | 24±6 px     | Stable target size |

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

### UUV3 (Validation/Test Set)

| Modality | Frames | Resolution | Bbox Width | Bbox Height | Notes |
|----------|--------|------------|------------|-------------|-------|
| RGB      | 751    | 1920×1080  | 281±273 px | 90±91 px    | High variance due to occlusions |
| Sonar    | 751    | 682×376    | 47±5 px    | 30±9 px     | Stable target size |

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

- Frame indexing starts at 1 (not 0)
- Both modalities must be processed jointly for optimal tracking performance
- Handle 0,0,0,0 annotations appropriately during training/evaluation
- Account for sonar's non-uniform spatial resolution in feature extraction
- Consider modality-specific preprocessing (e.g., denoising for sonar)

## Data Integrity

All sequences have been validated for:
- Consistent frame counts between RGB and sonar modalities
- Matching annotation lengths with frame counts
- Valid image formats and readability
- Bounding box coordinates within image boundaries (excluding occlusion cases)
