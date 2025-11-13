# AquaSOT-3D

**End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AquaSOT-3D is a unified end-to-end multi-modal single-object tracking system that fuses RGB camera and Forward-Looking Sonar (FLS) data for robust underwater object tracking. The system outputs 2D tracking bounding boxes for each modality and provides weak 3D center estimation (X, Y, Z) without requiring extrinsic calibration between sensors.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Algorithm Overview](#algorithm-overview)
  - [1. Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA)](#1-geometry-gated-deformable-cross-modal-attention-g-dcma)
  - [2. Differentiable Log-Polar Preprocessing](#2-differentiable-log-polar-preprocessing)
  - [3. Vision-Mamba Temporal Modeling](#3-vision-mamba-temporal-modeling)
  - [4. Weak 3D Center Estimation](#4-weak-3d-center-estimation)
  - [5. Quality-Aware Adaptive Fusion](#5-quality-aware-adaptive-fusion)
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
- **Differentiable Log-Polar Transform**: Handles sonar fan-shaped field of view with learnable parameters for end-to-end optimization
- **Linear-Complexity Temporal Modeling**: Vision-Mamba based temporal aggregation for efficient long-sequence tracking
- **Weak 3D Estimation**: Outputs 3D center position (X, Y, Z) with cyclic consistency constraints
- **Quality-Aware Weighting**: Adaptive fusion based on RGB sharpness and Sonar SNR
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
│  │ ViT/ConvNeXt│                 │ Log-Polar Transform│             │
│  │  Backbone   │                 │ (Learnable α, β)  │             │
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
│              ┌─────────────────────┐                                 │
│              │ Temporal Head       │                                 │
│              │ (Vision-Mamba SSM)  │                                 │
│              └─────────┬───────────┘                                 │
│                        ↓                                              │
│         ┌──────────────┴────────────────┐                           │
│         ↓              ↓                ↓                            │
│   ┌─────────┐   ┌──────────┐   ┌────────────┐                      │
│   │RGB Box  │   │Sonar Box │   │Fusion Box  │                      │
│   │Head     │   │Head      │   │Head        │                      │
│   └─────────┘   └──────────┘   └─────┬──────┘                      │
│                                       ↓                               │
│                            ┌──────────────────┐                      │
│                            │ Weak 3D Center   │                      │
│                            │ Head (X, Y, Z)   │                      │
│                            └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **Dual Backbones**: ViT/ConvNeXt for RGB, Lightweight ViT with log-polar preprocessing for Sonar
- **G²-DCMA Fusion**: Geometry-aware cross-modal attention with deformable sampling
- **Vision-Mamba Temporal**: Linear-complexity state-space model for temporal aggregation
- **Multi-Head Prediction**: Separate heads for RGB, Sonar, Fusion boxes and weak 3D center

---

## Algorithm Overview

### 1. Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA)

The core innovation of AquaSOT-3D is the G²-DCMA module, designed to handle the spatial misalignment between RGB and Sonar modalities without calibration.

**Problem**: RGB cameras and forward-looking sonar sensors capture the same scene from different perspectives with unknown spatial relationships. Direct feature concatenation or global attention leads to incorrect feature pairing.

**Solution**: G²-DCMA combines three complementary mechanisms:

#### (a) Geometric Gating
Constructs geometric priors from:
- **Radial activations (r)** from Sonar log-polar representation
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

### 2. Differentiable Log-Polar Preprocessing

Forward-looking sonar images exhibit a fan-shaped field of view with non-uniform spatial resolution. Traditional Cartesian processing is suboptimal.

**Log-Polar Transform**:
```
ρ = α · log(r + 1)  # Radial coordinate (log-scaled)
θ = β · angle       # Angular coordinate (scaled)
```

Where:
- `r` is the Euclidean distance from the sonar origin
- `angle` is the angular position within the sonar fan
- `α, β` are **learnable parameters** optimized end-to-end

**Advantages**:
- Encodes radial distance and angular information naturally
- Learnable scales adapt to specific sensor characteristics
- Differentiable for end-to-end gradient flow
- No manual calibration required

The log-polar representation is processed by a lightweight ViT encoder before fusion.

### 3. Vision-Mamba Temporal Modeling

Long-term temporal modeling is crucial for tracking under occlusions and rapid motion. Traditional transformers have quadratic complexity O(T²) in sequence length T.

**Vision-Mamba** employs State-Space Models (SSMs) for linear complexity O(T):

```
h_t = A · h_{t-1} + B · x_t    # State update
y_t = C · h_t + D · x_t        # Output projection
```

Where A, B, C, D are learnable matrices. The model maintains a compressed state vector h_t that summarizes historical information.

**Benefits**:
- **Linear complexity**: Efficient for long sequences (hundreds of frames)
- **Long-term dependencies**: State vector aggregates information over time
- **Occlusion recovery**: Predicts target position when temporarily invisible

Alternative: ARTrackV2 autoregressive head for ablation studies.

### 4. Weak 3D Center Estimation

The system estimates 3D center position (X, Y, Z) without extrinsic calibration through **cyclic consistency**:

#### Estimation Strategy
- **X (Horizontal)**: Regressed by both RGB and Sonar branches independently, enforced with component consistency loss
- **Y (Vertical)**: Primarily from RGB branch (better vertical resolution)
- **Z (Depth)**: Extracted from Sonar log-polar radial activation distribution via learnable weighted integral

#### Cyclic Consistency
A learnable projection head maps 3D center back to 2D bounding boxes:
```
(X, Y, Z) → Project → (bbox_rgb, bbox_sonar)
```

The projected boxes are compared with ground truth 2D annotations:
```
L_cycle = L(bbox_rgb_proj, bbox_rgb_gt) + L(bbox_sonar_proj, bbox_sonar_gt)
```

This ensures 3D estimates are geometrically consistent with 2D observations across modalities.

**Key Insight**: No calibration matrix is required. The model learns implicit sensor relationships through 3D→2D→3D cyclic constraints.

### 5. Quality-Aware Adaptive Fusion

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

### Run Training

**Multi-Sequence Training** (Recommended):

Train on all 9 training sequences:

```bash
python scripts/train.py --config configs/train/train_multiseq.yaml
```

**Single-Sequence Training** (Legacy):

Train on UUV2 only (for compatibility or debugging):

```bash
python scripts/train.py --config configs/train/train_uuv2.yaml
```

**Training Options**:
```bash
# RGB-only pretraining
python scripts/train.py --config configs/train/pretrain_rgb_only.yaml

# Sonar-only pretraining
python scripts/train.py --config configs/train/pretrain_sonar_only.yaml

# Resume training from checkpoint
python scripts/train.py --config configs/train/train_multiseq.yaml \
    --resume outputs/train_multiseq/checkpoints/checkpoint_epoch_20.pth

# Override sequences dynamically
python scripts/train.py --config configs/base/default.yaml \
    --opts data.data_root=data \
           data.train_sequences="[uuv1,uuv2,uuv4]" \
           data.val_sequences="[uuv3]" \
           training.batch_size=4

# Override configuration parameters
python scripts/train.py --config configs/train/train_multiseq.yaml \
    --opts training.lr=0.001 training.batch_size=8
```

### Run Evaluation

**Multi-Sequence Evaluation** (Recommended):

Evaluate on all 3 validation sequences:

```bash
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/checkpoint_best.pth
```

**Single-Sequence Evaluation** (Legacy):

Evaluate on UUV3 only:

```bash
python scripts/eval.py \
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/train_uuv2/checkpoints/checkpoint_best.pth
```

**Custom Evaluation**:
```bash
# Evaluate specific sequences
python scripts/eval.py \
    --config configs/base/default.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/checkpoint_best.pth \
    --opts data.data_root=data \
           data.test_sequences="[uuv6,uuv10]" \
           evaluation.batch_size=1
```

**Visualization Control**:

Visualization is **enabled by default** in evaluation configs. To control it:

```bash
# Disable visualization for faster evaluation
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/checkpoint_best.pth \
    --no-visualize

# Force enable visualization (override config)
python scripts/eval.py \
    --config configs/eval/eval_multiseq.yaml \
    --checkpoint outputs/train_multiseq/checkpoints/checkpoint_best.pth \
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

# Sonar-only pretraining (40 epochs, with log-polar preprocessing)
python scripts/train.py --config configs/train/pretrain_sonar_only.yaml
```

**Quick Testing** (5 epochs for validation):
```bash
python scripts/train.py --config configs/train_test/test_pretrain_rgb.yaml
python scripts/train.py --config configs/train_test/test_pretrain_sonar.yaml
```

#### Stage 2: Multi-Modal Fusion Training

Train the complete system with G²-DCMA fusion and temporal modules (60 epochs):

```bash
python scripts/train.py --config configs/train/train_uuv2.yaml
```

**Quick Testing** (5 epochs for validation):
```bash
python scripts/train.py --config configs/train_test/test_fusion.yaml
```

**Key Training Settings**:
- Small batch size (2-8) with gradient accumulation for memory efficiency
- Lower learning rate for pretrained backbones (0.1x multiplier)
- All loss components enabled (RGB, Sonar, Fusion, Weak3D, Consistency)

### Loss Functions

AquaSOT-3D employs a multi-task loss with initial weights:

| Loss Component | Weight | Description |
|----------------|--------|-------------|
| RGB Box Loss | 1.0 | GIoU + L1 + Quality Focal |
| Sonar Box Loss | 1.0 | GIoU + L1 + Quality Focal (log-polar domain) |
| Component Consistency | 0.2 | RGB-Sonar X coordinate consistency |
| Representation Consistency | 0.1 | InfoNCE contrastive loss |
| Weak 3D Cyclic | 0.6 | 3D→2D projection consistency |
| Temporal Smoothness | 0.1 | Short-term EMA regularization |

**Dynamic Weighting**: Loss weights are automatically adjusted during training based on task convergence.

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
    --config configs/train/train_uuv2.yaml

# Using torch.distributed.launch (older PyTorch)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train.py \
    --config configs/train/train_uuv2.yaml
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
    --config configs/eval/eval_uuv3.yaml \
    --checkpoint outputs/train_uuv2/checkpoints/best_model.pth \
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
# No Vision-Mamba (Transformer baseline)
python scripts/eval.py \
    --config configs/ablation/ablation_no_mamba.yaml \
    --checkpoint model_no_mamba.pth

# No Geometric Gating
python scripts/eval.py \
    --config configs/ablation/ablation_no_gating.yaml \
    --checkpoint model_no_gating.pth

# No Weak 3D Estimation
python scripts/eval.py \
    --config configs/ablation/ablation_no_3d.yaml \
    --checkpoint model_no_3d.pth
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
│   ├── train_uuv2.yaml          # Full training (60 epochs)
│   ├── pretrain_rgb_only.yaml   # RGB-only pretraining (40 epochs)
│   └── pretrain_sonar_only.yaml # Sonar-only pretraining (40 epochs)
├── train_test/
│   ├── test_pretrain_rgb.yaml   # Quick RGB test (5 epochs)
│   ├── test_pretrain_sonar.yaml # Quick Sonar test (5 epochs)
│   └── test_fusion.yaml         # Quick fusion test (5 epochs)
├── eval/
│   └── eval_uuv3.yaml           # Evaluation configuration
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
python scripts/train.py --config configs/train/train_uuv2.yaml \
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
    --checkpoint outputs/train_uuv2/checkpoints/best_model.pth \
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
- All 2D bounding boxes and 3D centers are expressed in Fusion frame

**Weak 3D Center**:
- Output: `(X, Y, Z)` in Fusion frame coordinates
- X: Horizontal displacement from image center
- Y: Vertical displacement from image center
- Z: Estimated depth/range (not metrically accurate, relative scale)

---

## Technology Stack

### Deep Learning Frameworks

- **PyTorch 2.3.0+**: Core deep learning framework with CUDA 11.8
- **torchvision 0.18.0+**: Image transformations and pretrained models
- **timm (PyTorch Image Models)**: Vision Transformer and ConvNeXt backbones
- **mamba-ssm 1.2+**: Vision-Mamba state-space models for temporal modeling
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

### Temporal Modeling

- **Vision-Mamba** (ICML 2024): *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model*
  - Code: https://github.com/hustvl/Vim

- **VMamba** (2024): *VMamba: Visual State Space Models*
  - Code: https://github.com/MzeroMiko/VMamba

- **ARTrackV2** (CVPR 2024): *ARTrackV2: Prompting Autoregressive Tracker Where to Look and How to Describe*
  - Paper: https://arxiv.org/abs/2312.17133
  - Code: https://github.com/MIV-XJTU/ARTrack

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
  title={AquaSOT-3D: End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation},
  author={AquaSOT-3D Team},
  year={2025},
  url={https://github.com/yourusername/AquaSOT-3D},
  note={Calibration-free multi-modal underwater tracking with geometry-aware fusion}
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

@inproceedings{visionmamba2024,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  booktitle={ICML},
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

**Acknowledgments**: This project builds upon the excellent work of OSTrack, RGBS50, Vision-Mamba, and the broader computer vision research community. We thank the authors for open-sourcing their implementations.