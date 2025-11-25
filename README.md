# AquaSOT-3D

**End-to-End RGB+Sonar Single-Object Tracking**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AquaSOT-3D is a unified end-to-end multi-modal single-object tracking system that fuses RGB camera and Forward-Looking Sonar (FLS) data for robust underwater object tracking. The system outputs 2D tracking bounding boxes for each modality using advanced cross-modal attention fusion without requiring extrinsic calibration between sensors.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Algorithm Overview](#algorithm-overview)
  - [1. Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA)](#1-geometry-gated-deformable-cross-modal-attention-g-dcma)
  
  - [2. Confidence-Aware Adaptive Fusion](#2-confidence-aware-adaptive-fusion)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Model Export and Deployment](#model-export-and-deployment)
- [Coordinate System](#coordinate-system)
- [Technology Stack](#technology-stack)
- [References](#references)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- **Calibration-Free Multi-Modal Fusion**: Fuses RGB and Sonar data without requiring extrinsic calibration parameters
- **Novel G²-DCMA Module**: Geometry-Gated Deformable Cross-Modal Attention with geometric regularization for spatially misaligned modalities
- **Template-Search Paradigm**: OSTrack-inspired one-stream architecture with per-frame template updates
- **Confidence-Aware Weighting**: Learnable confidence estimation for adaptive multi-modal fusion
- **Multi-Scale Prediction**: Separate prediction heads for RGB, Sonar, and fused modalities
- **OSTrack Compatible**: Built on the proven one-stream template-search framework

---

## System Architecture

AquaSOT-3D employs a template-search paradigm inspired by OSTrack, processing paired RGB and Sonar frames through separate feature extraction branches before fusing them with geometric priors:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AquaSOT-3D Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  RGB Input (1920×1080)              Sonar Input (682×376)           │
│         │                                   │                        │
│         ↓                                   ↓                        │
│  ┌─────────────┐                 ┌──────────────────┐              │
│  │ ViT/ConvNeXt│                 │   ViT Backbone   │             │
│  │  Backbone   │                 │  (Sonar Encoder) │             │
│  └──────┬──────┘                 └──────────┬─────────┘             │
│         │                                   │                        │
│         │    RGB Features                   │  Sonar Features        │
│         └───────────────┬───────────────────┘                        │
│                         ↓                                            │
│              ┌─────────────────────┐                                 │
│              │  G²-DCMA Fusion     │                                 │
│              │  ┌───────────────┐  │                                 │
│              │  │ Geometry Gate │  │                                 │
│              │  │ (Radial+Scale)│  │                                 │
│              │  └───────┬───────┘  │                                 │
│              │          ↓           │                                 │
│              │  ┌───────────────┐  │                                 │
│              │  │ Deformable    │  │                                 │
│              │  │ Cross-Attn    │  │                                 │
│              │  └───────┬───────┘  │                                 │
│              │          ↓           │                                 │
│              │  ┌───────────────┐  │                                 │
│              │  │ Dual          │  │                                 │
│              │  │ Consistency   │  │                                 │
│              │  └───────────────┘  │                                 │
│              └─────────┬───────────┘                                 │
│                        ↓                                              │
│         ┌──────────────┼────────────────┐                           │
│         ↓              ↓                ↓                            │
│   ┌─────────┐   ┌──────────┐   ┌────────────┐                      │
│   │RGB Box  │   │Sonar Box │   │Fusion Box  │                      │
│   │Head     │   │Head      │   │Head        │                      │
│   │(CXCYWH) │   │(CXCYWH)  │   │(CXCYWH)    │                      │
│   └─────────┘   └──────────┘   └────────────┘                      │
│                                                                       │
│  Output: 2D Bounding Boxes (RGB, Sonar, Fusion)                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **Dual Backbones**: ViT for RGB (3-channel), ViT with 1-to-3 channel adapter for Sonar
- **G²-DCMA Fusion**: Geometry-aware cross-modal attention with deformable sampling
- **Dual Consistency**: Geometric consistency (scale vs radial) and representation consistency (InfoNCE)
- **Multi-Head Prediction**: Separate 2D box prediction heads for RGB, Sonar, and Fusion modalities

---

## Algorithm Overview

### 1. Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA)

The core innovation of AquaSOT-3D is the G²-DCMA module, designed to handle the spatial misalignment between RGB and Sonar modalities without calibration.

**Problem**: RGB cameras and forward-looking sonar sensors capture the same scene from different perspectives with unknown spatial relationships. Direct feature concatenation or global attention leads to incorrect feature pairing.

**Solution**: G²-DCMA combines three complementary mechanisms:

#### (a) Geometric Gating
Constructs geometric priors from:
- **Radial activations (r)** from Sonar feature maps
- **Scale pyramids (s)** from RGB feature maps

Attention weights are modulated by a learned gating function:
```
attention_weight = σ(g(r, s)) * attention_score
```

When geometric priors align (e.g., near/far distances or similar scales), cross-modal attention is amplified; otherwise, it is suppressed.

#### (b) Deformable Cross-Attention
Inspired by Deformable DETR, adds learnable spatial offsets to query-key-value attention:
```
Attention(Q, K, V) = ∑ w_i · V(p + Δp_i)
```

This allows fine-grained spatial alignment correction between misaligned modalities.

#### (c) Dual Consistency
Enforces feature consistency at two levels:

1. **Component Consistency**: RGB and Sonar branches independently estimate the same physical quantity (e.g., X coordinate):
   ```
   L_comp = ||X_rgb - X_sonar||²
   ```

2. **Representation Consistency**: InfoNCE contrastive loss ensures features from the same frame/target are similar while features from different frames are dissimilar:
   ```
   L_rep = -log[exp(sim(f_rgb, f_sonar)/τ) / ∑ exp(sim(f_rgb, f_neg)/τ)]
   ```

**Result**: The model learns "who talks to whom" across modalities through geometry-aware gating and deformable alignment, achieving calibration-free correspondence.

### 2. Confidence-Aware Adaptive Fusion

Not all frames provide equally reliable information. AquaSOT-3D estimates per-frame quality and adapts fusion weights:

**Quality Metrics**:
- **RGB Sharpness**: Laplacian variance measures focus and turbidity
- **Sonar SNR**: Signal-to-noise ratio from intensity distribution

**Adaptive Weighting**:
```
w_rgb = softmax(quality_rgb)
w_sonar = softmax(quality_sonar)
output = w_rgb · prediction_rgb + w_sonar · prediction_sonar + w_fusion · prediction_fusion
```

Low-quality modalities are automatically downweighted, improving robustness.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8 or higher (for GPU acceleration)
- GCC 7.5+ (for compiling Mamba SSM)

### Install from Source

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/AquaSOT-3D.git
cd AquaSOT-3D
```

2. **Create virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch with CUDA 11.8**:
```bash
pip3 install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

4. **Install AquaSOT-3D**:
```bash
# Development mode (editable installation)
pip install -e ".[dev]"

# Or production mode (core dependencies only)
pip install -e .
```

**Note**: Installing `mamba-ssm` requires CUDA toolkit and may take 5-10 minutes to compile.

### Verify Installation

```bash
# Check package installation
python -c "import aquasot3d; print('AquaSOT-3D installed successfully!')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Prepare Dataset

AquaSOT-3D supports **multi-sequence training** across 12 sequences. Each sequence follows the same structure:

```
data/
├── uuv1/                    # Training sequence 1
│   ├── light/               # RGB modality
│   │   ├── img/             # Frame images (1.jpg, 2.jpg, ...)
│   │   └── groundtruth.txt  # Bounding box annotations
│   └── sonar/               # Sonar modality
│       ├── img/             # Frame images (1.jpg, 2.jpg, ...)
│       └── groundtruth.txt
├── uuv2/                    # Training sequence 2 (2,740 frames)
│   ├── light/
│   └── sonar/
├── uuv3/                    # Validation sequence 1 (751 frames)
│   ├── light/
│   └── sonar/
... (uuv4-12 follow the same pattern)
```

**Dataset Split**:
- **Training**: `uuv1, uuv2, uuv4, uuv5, uuv7, uuv8, uuv9, uuv11, uuv12` (9 sequences)
- **Validation/Test**: `uuv3, uuv6, uuv10` (3 sequences)

**Groundtruth Format**: Each line contains `x,y,w,h` (top-left coordinates and size). Use `0,0,0,0` for occluded frames.

### Preprocess Sonar Data

**IMPORTANT**: Before training or evaluation, you **must** preprocess sonar data to convert polar coordinates to Cartesian (rectified) format. The template-search architecture requires geometrically consistent bounding boxes, which are only valid in Cartesian coordinates.

**Preprocess All Sequences** (Recommended):

```bash
python scripts/preprocess_sonar.py --all
```

This will generate rectified sonar data for all sequences:
- `sonar/img_rect/` - Rectified sonar images (Cartesian coordinates)
- `sonar/groundtruth_rect.txt` - Rectified bounding box annotations

**Preprocess Single Sequence**:

```bash
# Preprocess specific sequence
python scripts/preprocess_sonar.py --sequence uuv1
```

**Preprocess Multiple Sequences**:

```bash
# Process sequences one by one
python scripts/preprocess_sonar.py --sequence uuv1
python scripts/preprocess_sonar.py --sequence uuv2
python scripts/preprocess_sonar.py --sequence uuv3

# Or use a shell loop
for seq in uuv1 uuv2 uuv3; do
    python scripts/preprocess_sonar.py --sequence $seq
done
```

**Why Preprocessing is Required**:
- Raw sonar data is in **polar coordinates** (range-bearing format)
- Training requires **Cartesian coordinates** for letterbox transforms and bbox operations
- Using polar data directly will produce geometrically incorrect training

**Verification**: After preprocessing, your dataset structure should include:
```
data/uuv1/sonar/
├── img/                    # Original polar images (preserved)
├── img_rect/               # Rectified Cartesian images (generated)
├── groundtruth.txt         # Original polar annotations (preserved)
└── groundtruth_rect.txt    # Rectified Cartesian annotations (generated)
```

### Run Training

**Multi-Sequence Training** (Recommended):

Train on all 9 training sequences:

```bash
python scripts/train.py --config configs/train.yaml
```

**Single-Sequence Training** (Legacy):

Train on UUV2 only (for compatibility or debugging):

```bash
python scripts/train.py --config configs/train.yaml
```

**Training Options**:
```bash
# RGB-only pretraining
python scripts/train.py --config configs/train/pretrain_rgb_only.yaml

# Sonar-only pretraining
python scripts/train.py --config configs/train/pretrain_sonar_only.yaml

# Resume training from checkpoint
python scripts/train.py --config configs/train/train.yaml \
    --resume outputs/train/checkpoints/checkpoint_epoch_20.pth

# Override sequences dynamically
python scripts/train.py --config configs/base/default.yaml \
    --opts data.data_root=data \
           data.train_sequences="[uuv1,uuv2,uuv4]" \
           data.val_sequences="[uuv3]" \
           training.batch_size=4

# Override configuration parameters
python scripts/train.py --config configs/train/train.yaml \
    --opts training.lr=0.001 training.batch_size=8
```

### Run Evaluation

**Multi-Sequence Evaluation** (Recommended):

Evaluate on all 3 validation sequences:

```bash
python scripts/eval.py \
    --config configs/eval.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth
```

**Single-Sequence Evaluation** (Legacy):

Evaluate on UUV3 only:

```bash
python scripts/eval.py \
    --config configs/eval.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth
```

**Custom Evaluation**:
```bash
# Evaluate specific sequences
python scripts/eval.py \
    --config configs/base/default.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth \
    --opts data.data_root=data \
           data.test_sequences="[uuv6,uuv10]" \
           evaluation.batch_size=1
```

**Visualization Control**:

Visualization is **enabled by default** in evaluation configs. To control it:

```bash
# Disable visualization for faster evaluation
python scripts/eval.py \
    --config configs/eval/eval.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth \
    --no-visualize

# Force enable visualization (override config)
python scripts/eval.py \
    --config configs/eval/eval.yaml \
    --checkpoint outputs/train/checkpoints/checkpoint_best.pth \
    --visualize
```

**Evaluation Output**:
- **Metrics**: Success curves (AUC), Precision curves, Average Center Error, Per-modality metrics (RGB, Sonar, Fusion)
- **Visualization** (when enabled): Trajectory plots, success/precision curves, failure case analysis
  - Saved to: `outputs/<eval_name>/visualization/`
  - Includes best/worst sequence visualizations

### Run Demo

Track objects in custom video or image sequences:

```bash
# Video input
python scripts/demo.py \
    --checkpoint model.pth \
    --input video.mp4 \
    --output demo_result.mp4

# Image directory
python scripts/demo.py \
    --checkpoint model.pth \
    --input images/ \
    --output result.mp4
```

---

## Dataset

### Supported Datasets

AquaSOT-3D is designed for **RGBS50-style** paired RGB+Sonar datasets. The provided example uses:

- **UUV2**: 2,740 frames (training set)
- **UUV3**: 751 frames (validation/test set)

### Modality Characteristics

| Modality | Resolution | Field of View | Characteristics |
|----------|-----------|---------------|-----------------|
| **RGB Camera** | 1920×1080 | Rectangular | High-resolution visual details, affected by turbidity and lighting |
| **Forward-Looking Sonar** | 682×376 | Fan-shaped | Range-bearing information, low SNR, lighting-independent |

### Annotation Format

**Bounding Box Annotations** (`groundtruth.txt`):
```
x,y,w,h   # Frame 1
x,y,w,h   # Frame 2
0,0,0,0   # Frame 3 (target occluded)
...
```

- `x, y`: Top-left corner coordinates
- `w, h`: Width and height
- `0,0,0,0`: Indicates occlusion or target not visible

### Dataset Statistics

**UUV2 (Training)**:
- RGB: Mean bbox size 335×110 px (high variance due to occlusions)
- Sonar: Mean bbox size 49×25 px (stable)

**UUV3 (Validation/Test)**:
- RGB: Mean bbox size 281×90 px
- Sonar: Mean bbox size 47×30 px

**Key Observations**:
- Temporal synchronization between RGB and Sonar
- Sonar provides more stable tracking (lower variance)
- Frequent occlusions in both modalities
- Complementary information: RGB for visual details, Sonar for robust range-bearing

---

## Training

### Training Pipeline

AquaSOT-3D follows a two-stage training strategy:

#### Stage 1: Single-Modality Pretraining

Train RGB and Sonar branches independently to initialize backbone networks:

```bash
# RGB-only pretraining (40 epochs)
python scripts/train.py --config configs/train/pretrain_rgb_only.yaml

# Sonar-only pretraining (40 epochs)
python scripts/train.py --config configs/train/pretrain_sonar_only.yaml
```

**Quick Testing** (5 epochs for validation):
```bash
python scripts/train.py --config configs/train/pretrain_rgb_only.yaml
python scripts/train.py --config configs/train/pretrain_sonar_only.yaml
```

#### Stage 2: Multi-Modal Fusion Training

Train the complete system with G²-DCMA fusion and temporal modules (60 epochs):

```bash
python scripts/train.py --config configs/train/train.yaml
```

**Quick Testing** (5 epochs for validation):
```bash
python scripts/train.py --config configs/train/train.yaml
```

**Key Training Settings**:
- Small batch size (2-8) with gradient accumulation for memory efficiency
- Lower learning rate for pretrained backbones (0.1x multiplier)
- All loss components enabled (RGB, Sonar, Fusion, Cross-Modal Consistency)

### Loss Functions

AquaSOT-3D employs a multi-task loss with initial weights:

| Loss Component | Weight | Description |
|----------------|--------|-------------|
| RGB Box Loss | 1.0 | GIoU + L1 + Quality Focal |
| RGB Cls Loss | 0.5 | Classification confidence |
| Sonar Box Loss | 1.0 | GIoU + L1 + Quality Focal |
| Sonar Cls Loss | 0.5 | Classification confidence |
| Fusion Box Loss | 1.5 | GIoU + L1 + Quality Focal for fused predictions |
| Fusion Cls Loss | 0.8 | Classification confidence for fusion |
| Geometric Consistency | 0.5 | Fusion layer: scale vs radial encoding alignment |
| Representation Consistency | 0.3 | InfoNCE contrastive loss for feature alignment |

**Static Weighting**: Loss weights are fixed during training to maintain stable 2D tracking performance.

### Optimization

- **Optimizer**: AdamW with decoupled weight decay
- **Learning Rate**: Base lr = 2e-4 with cosine decay
- **Batch Size**: 32 (adjust based on GPU memory)
- **Gradient Clipping**: Max norm = 1.0
- **Mixed Precision**: FP16 automatic mixed precision (AMP)

### Data Augmentation

AquaSOT-3D implements **domain-specific augmentation pipelines** tailored to optical and acoustic imaging characteristics:

**RGB Augmentation** (optical domain):
- **ColorJitter**: Brightness, contrast, saturation, hue variations (prob=0.5)
- **RandomBlur**: Gaussian blur simulation for turbidity effects (prob=0.3)
- **RandomOcclusion**: Synthetic rectangular occlusions (prob=0.2, size 5-30%)

**Sonar Augmentation** (acoustic domain):
- **SpeckleNoise**: Multiplicative speckle noise characteristic of sonar imagery (prob=0.5, level=0.1)
- **SectorOcclusion**: Fan-shaped occlusion masks matching sonar geometry (prob=0.2, 10-40°)
- **RadialScalingJitter**: DISABLED due to bbox synchronization bug (geometric transform does not update bbox coordinates, causing ~19% IoU loss)

**Implementation Details**:
- Transforms are modality-specific and applied independently
- No shared geometric transforms (flip/rotation) to preserve sensor-specific geometry
- Augmentation enabled only during training (disabled for validation)
- Implemented in `src/aquasot3d/dataloader/transforms/`

### Distributed Training

For multi-GPU training:

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 \
    scripts/train.py \
    --config configs/train/train.yaml

# Using torch.distributed.launch (older PyTorch)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train.py \
    --config configs/train/train.yaml
```

**Distributed Features**:
- DistributedDataParallel (DDP)
- Gradient synchronization across GPUs
- Automatic batch splitting

---

## Evaluation

### Evaluation Metrics

AquaSOT-3D uses **RGBS50-style** evaluation metrics:

1. **Success Rate (AUC)**: Area under the success curve (IoU thresholds 0-1)
2. **Precision**: Percentage of frames with center error < 20 pixels
3. **Average Center Error**: Mean Euclidean distance between predicted and ground truth centers

Metrics are reported separately for:
- **RGB**: RGB-only predictions
- **Sonar**: Sonar-only predictions
- **Fusion**: Multi-modal fused predictions (primary metric)

### Run Evaluation

```bash
python scripts/eval.py \
    --config configs/eval/eval.yaml \
    --checkpoint outputs/train/checkpoints/best_model.pth \
    --output eval_results/
```

### Evaluation Output

The evaluation script generates:

**Text Reports** (`results.txt`):
```
=== Evaluation Results ===
Dataset: UUV3
Sequence: uuv3 (751 frames)

RGB Metrics:
  Success (AUC): 0.652
  Precision: 0.741
  Center Error: 12.3 px

Sonar Metrics:
  Success (AUC): 0.698
  Precision: 0.812
  Center Error: 8.7 px

Fusion Metrics:
  Success (AUC): 0.743
  Precision: 0.856
  Center Error: 6.9 px
```

**Visualization**:
- Success curves (IoU threshold vs. success rate)
- Precision curves (distance threshold vs. precision)
- Per-frame tracking videos with overlaid boxes
- Failure case analysis

**JSON Report** (`results.json`):
Detailed per-frame results for further analysis.

### Ablation Studies

Evaluate model variants to analyze component contributions:

```bash
# No Geometric Gating
python scripts/eval.py \
    --config configs/ablation/ablation_no_gating.yaml \
    --checkpoint model_no_gating.pth

# No Deformable Attention
python scripts/eval.py \
    --config configs/ablation/ablation_no_deformable.yaml \
    --checkpoint model_no_deformable.pth
```

---

## Configuration

### Configuration System

AquaSOT-3D uses hierarchical YAML configuration files:

```
configs/
├── base/
│   └── default.yaml              # Base configuration with all defaults
├── train/
│   ├── train.yaml      # Full multi-sequence training (60 epochs)
│   ├── pretrain_rgb_only.yaml   # RGB-only pretraining (40 epochs)
│   └── pretrain_sonar_only.yaml # Sonar-only pretraining (40 epochs)
├── eval/
│   └── eval.yaml       # Multi-sequence evaluation configuration
└── ablation/
    ├── ablation_no_mamba.yaml
    ├── ablation_no_gating.yaml
    ├── ablation_no_3d.yaml
    └── ablation_transformer_vs_mamba.yaml
```

### Configuration Structure

Configuration files use YAML format with hierarchical inheritance via `_base_` directive. Key configuration sections include:

- **data**: Dataset paths, augmentation settings, batch size
- **model**: Backbone, fusion, temporal, and head configurations
- **training**: Optimizer, scheduler, loss weights, epochs
- **system**: Device, mixed precision, distributed training

Refer to actual configuration files in `configs/` directory for detailed parameters.

### Command-Line Overrides

Override any configuration parameter at runtime:

```bash
python scripts/train.py --config configs/train/train.yaml \
    --opts \
    training.lr=0.001 \
    training.batch_size=16 \
    model.fusion.use_geometry_gating=false \
    model.temporal.type=transformer
```

---

## Model Export and Deployment

### Export to ONNX

Export trained model for deployment:

```bash
python scripts/export.py \
    --checkpoint outputs/train/checkpoints/best_model.pth \
    --output model.onnx \
    --format onnx \
    --input_shape 1,3,224,224
```

**Supported Formats**:
- ONNX (Open Neural Network Exchange)
- TorchScript
- TensorRT (for NVIDIA GPUs)

### Model Inference

Use exported model for inference:

```python
import torch
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Prepare input
rgb_input = preprocess_rgb(rgb_image)  # [1, 3, H, W]
sonar_input = preprocess_sonar(sonar_image)  # [1, 3, H, W]

# Run inference
outputs = session.run(
    None,
    {
        "rgb": rgb_input.numpy(),
        "sonar": sonar_input.numpy()
    }
)

# Parse outputs
rgb_box, sonar_box, fusion_box, xyz = outputs
```

---

## Coordinate System

AquaSOT-3D uses a unified **Fusion Frame** coordinate system:

```
     Y (Down)
     │
     │
     └────── X (Right)
    ╱
   ╱
  Z (Forward / Depth)
```

**Coordinate Definitions**:
- **X**: Horizontal axis (right is positive)
- **Y**: Vertical axis (down is positive, following image convention)
- **Z**: Depth axis (forward/away from camera is positive)

**Modality Alignment**:
- RGB frame is aligned with Fusion frame
- Sonar-to-Fusion transformation is learned implicitly (no calibration required)
- All 2D bounding boxes are expressed in normalized [0,1] CXCYWH format

**Coordinate Format**:
- All bounding boxes use normalized center-width-height format: `(cx, cy, w, h)` in [0, 1]
- `cx, cy`: Normalized center coordinates
- `w, h`: Normalized width and height

---

## Technology Stack

### Deep Learning Frameworks

- **PyTorch 2.3.0+**: Core deep learning framework with CUDA 11.8
- **torchvision 0.18.0+**: Image transformations and pretrained models
- **timm (PyTorch Image Models)**: Vision Transformer backbones
- **einops**: Elegant tensor operations for multi-dimensional arrays

### Computer Vision

- **OpenCV 4.8+**: Image I/O and preprocessing
- **scikit-image 0.20+**: Advanced image processing algorithms
- **Pillow 9.5+**: Image manipulation

### Scientific Computing

- **NumPy 1.24+**: Numerical computations
- **SciPy 1.10+**: Scientific algorithms

### Configuration and Logging

- **PyYAML 6.0+**: Configuration file parsing
- **TensorBoard 2.13+**: Training visualization and logging
- **tqdm 4.65+**: Progress bars

### Visualization

- **Matplotlib 3.7+**: Plotting and visualization

### Data Processing

- **pandas 2.0+**: Data manipulation and analysis
- **scikit-learn 1.3+**: Machine learning utilities

---

## References

### Base Framework

- **OSTrack** (ECCV 2022): *Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework*
  - Paper: https://arxiv.org/abs/2203.11991
  - Code: https://github.com/botaoye/OSTrack

### Task and Benchmark

- **RGBS50** (2024): *RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker*
  - Paper: https://arxiv.org/abs/2406.07189
  - Code: https://github.com/LiYunfengLYF/RGBS50

### Attention Mechanisms

- **Deformable DETR** (ICLR 2021): *Deformable DETR: Deformable Transformers for End-to-End Object Detection*
  - Paper: https://arxiv.org/abs/2010.04159
  - Code: https://github.com/fundamentalvision/Deformable-DETR

### Robustness

- **NA-OSTrack**: *Noise-Aware OSTrack for Robust Object Tracking*
  - Code: https://github.com/fbasatemur/NA-OSTrack

---

## Citation

If you use AquaSOT-3D in your research, please cite:

```bibtex
@software{aquasot3d2025,
  title={AquaSOT-3D: End-to-End RGB+Sonar Single-Object Tracking},
  author={AquaSOT-3D Team},
  year={2025},
  url={https://github.com/yourusername/AquaSOT-3D},
  note={Calibration-free multi-modal underwater tracking with geometry-gated deformable cross-modal attention}
}
```

**Key Publications**:

```bibtex
@inproceedings{ostrack2022,
  title={Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework},
  author={Ye, Botao and Chang, Hong and Ma, Bingpeng and Shan, Shiguang},
  booktitle={ECCV},
  year={2022}
}

@article{rgbs50_2024,
  title={RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker},
  author={Li, Yunfeng and others},
  journal={arXiv preprint arXiv:2406.07189},
  year={2024}
}
```

---

## License

AquaSOT-3D is released under the **MIT License**. See [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 AquaSOT-3D Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

For questions, issues, or collaboration inquiries:

- **GitHub Issues**: https://github.com/yourusername/AquaSOT-3D/issues
- **Email**: aquasot3d@example.com

---

**Acknowledgments**: This project builds upon the excellent work of OSTrack, RGBS50, Deformable DETR, and the broader computer vision research community. We thank the authors for open-sourcing their implementations.
