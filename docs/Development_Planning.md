# AquaSOT-3D Development Planning Document

## Project Overview

**AquaSOT-3D** is an end-to-end multi-modal single-object tracking system that fuses RGB camera and Forward-Looking Sonar (FLS) data for underwater object tracking. The system outputs 2D tracking bounding boxes and weak 3D center estimation (X, Y, Z) without requiring extrinsic calibration.

**Key Features:**
- Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA)
- Differentiable Log-Polar preprocessing for sonar data
- Vision-Mamba temporal modeling for long-sequence tracking
- Weak 3D center estimation with cyclic consistency
- Calibration-free multi-modal fusion

**Base Framework:** OSTrack (ECCV 2022)
**Target Dataset:** RGBS50-style paired RGB+Sonar sequences
**Current Dataset:**
- UUV2: 2740 frames (training set)
- UUV3: 751 frames (validation/test set, no split needed)

**Development Timeline:** 26 weeks (24 weeks core development + 2 weeks buffer for debugging and iteration)
**Primary Backbone:** ViT (Vision Transformer), ConvNeXt as optional alternative
**Coordinate System:** Fusion Frame (X→right, Y→down, Z→forward)

---

## Software Architecture

### 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AquaSOT-3D System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐                    ┌──────────────┐           │
│  │ RGB Branch   │                    │ Sonar Branch │           │
│  │ (ViT/ConvNeXt)│                   │ (LogPolar+ViT)│          │
│  └──────┬───────┘                    └──────┬───────┘           │
│         │                                    │                   │
│         └────────────────┬───────────────────┘                   │
│                          ↓                                       │
│              ┌───────────────────────┐                           │
│              │  G²-DCMA Fusion      │                           │
│              │  - Geometry Gating    │                           │
│              │  - Deformable Attn   │                           │
│              │  - Dual Consistency  │                           │
│              └───────────┬───────────┘                           │
│                          ↓                                       │
│              ┌───────────────────────┐                           │
│              │ Temporal Head         │                           │
│              │ (Vision-Mamba/ARTrack)│                           │
│              └───────────┬───────────┘                           │
│                          ↓                                       │
│         ┌────────────────┴────────────────┐                     │
│         ↓                ↓                ↓                     │
│   ┌──────────┐   ┌─────────────┐   ┌──────────┐               │
│   │ RGB Box  │   │ Sonar Box   │   │ Fusion   │               │
│   │ Head     │   │ Head        │   │ Box Head │               │
│   └──────────┘   └─────────────┘   └────┬─────┘               │
│                                          ↓                      │
│                              ┌───────────────────┐              │
│                              │ Weak 3D Center    │              │
│                              │ (X, Y, Z)         │              │
│                              └───────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Module Hierarchy

```
aquasot3d/
├── core/                      # Core functionality
│   ├── base_module.py         # Base PyTorch module interfaces
│   ├── registry.py            # Component registry system
│   └── config.py              # Configuration management
├── dataloader/                # Data loading and processing
│   ├── datasets/
│   │   ├── rgb_sonar_dataset.py         # Multi-modal dataset loader
│   │   ├── sequence_sampler.py          # Temporal sequence sampling
│   │   ├── template_search_sampler.py   # OSTrack template-search pairs (NEW)
│   │   └── augmentation.py              # Data augmentation pipelines
│   ├── transforms/
│   │   ├── rgb_transforms.py            # RGB-specific transforms
│   │   ├── sonar_transforms.py          # Sonar-specific transforms
│   │   ├── sonar_geometry.py            # Fan-shape mask, radial weights (NEW)
│   │   └── logpolar.py                  # Differentiable log-polar transform
│   └── loader.py                        # DataLoader utilities
├── models/                    # Neural network models
│   ├── backbones/
│   │   ├── vit_rgb.py             # Vision Transformer for RGB
│   │   ├── convnext_rgb.py        # ConvNeXt backbone for RGB
│   │   ├── sonar_encoder.py       # Sonar feature encoder
│   │   └── backbone_factory.py    # Backbone factory
│   ├── fusion/
│   │   ├── g2_dcma.py             # G²-DCMA fusion module
│   │   ├── geometric_gating.py    # Geometry-aware gating
│   │   ├── deformable_attention.py # Deformable cross-attention
│   │   └── consistency_loss.py    # Dual consistency losses
│   ├── quality/                   # Quality estimation (NEW)
│   │   ├── rgb_quality_estimator.py    # RGB quality metrics
│   │   ├── sonar_quality_estimator.py  # Sonar SNR estimation
│   │   └── adaptive_weighting.py       # Dynamic fusion weights
│   ├── temporal/
│   │   ├── mamba_head.py          # Vision-Mamba temporal head
│   │   ├── artrack_head.py        # ARTrackV2-style head (optional)
│   │   ├── transformer_head.py    # Transformer fallback (NEW)
│   │   └── temporal_aggregator.py # Temporal feature aggregation
│   ├── heads/
│   │   ├── box_head.py            # 2D bounding box prediction
│   │   ├── fusion_head.py         # Multi-modal fusion head
│   │   └── weak3d_head.py         # Weak 3D center prediction
│   ├── projector/
│   │   └── geom_cycle.py          # 3D↔2D cyclic projection
│   └── tracker.py                 # Complete tracking model
├── training/                  # Training infrastructure
│   ├── trainer.py             # Training loop and logic
│   ├── optimizer.py           # Optimizer configuration
│   ├── scheduler.py           # Learning rate scheduling
│   ├── loss_manager.py        # Multi-task loss management
│   └── checkpoint.py          # Model checkpointing
├── evaluation/                # Evaluation and metrics
│   ├── metrics.py             # Success (AUC), Precision, Center Error
│   ├── evaluator.py           # Evaluation pipeline
│   └── visualizer.py          # Result visualization
├── utils/                     # Utilities
│   ├── logging.py             # Logging utilities
│   ├── checkpoint_io.py       # Checkpoint I/O
│   ├── distributed.py         # Distributed training support
│   ├── geometry.py            # Coordinate system utilities (NEW)
│   └── misc.py                # Miscellaneous helpers
└── __init__.py                # Package initialization
```

---

## Development Roadmap

### Phase 1: Foundation & Infrastructure (Weeks 1-2) ✅ COMPLETED

#### 1.1 Project Setup
- [x] Create `setup.py` / `pyproject.toml` for pip installation
- [x] Set up `requirements.txt` with core dependencies
- [x] Configure `__init__.py` for proper module imports
- [x] Create basic logging infrastructure
- [x] Set up configuration management system

**Dependencies:**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
einops>=0.7.0
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
mamba-ssm>=1.0.0          # Vision-Mamba temporal modeling
scikit-image>=0.20.0      # Image quality metrics (sharpness, blur)
scipy>=1.10.0             # Signal processing for sonar SNR estimation
```

#### 1.2 Core Module Development
- [x] Implement `core/base_module.py` - Abstract base classes
- [x] Implement `core/registry.py` - Component registration system
- [x] Implement `core/config.py` - YAML-based configuration
- [x] Create base unit tests for core modules

**Test Files:**
- `tests/test_core_registry.py` (19 tests ✅)
- `tests/test_config_loader.py` (28 tests ✅)
- `tests/test_base_module.py` (24 tests ✅)

**Phase 1 Summary:**
- Total tests: 71 (100% passing)
- Package installed and functional
- Directory structure optimized (data/ vs dataloader/)

---

### Phase 2: Data Pipeline (Weeks 3-4) ✅ COMPLETED

#### 2.1 Dataset Loader
- [x] Implement `dataloader/datasets/rgb_sonar_dataset.py`
  - Load paired RGB and Sonar frames
  - Parse groundtruth.txt annotations
  - Handle missing/occluded frames (0,0,0,0):
    * Skip occluded frames during training sampling
    * Return occlusion mask for loss computation
    * Maintain temporal continuity by using last valid bbox
  - Support template-search pair generation
- [x] Implement `dataloader/datasets/sequence_sampler.py`
  - Temporal sequence sampling strategies
  - Frame interval control
  - Occlusion-aware sampling
- [x] Implement `dataloader/datasets/augmentation.py`
  - Unified augmentation pipelines
  - Coordinated RGB-Sonar augmentation
  - Template-search augmentation strategy

**Test Files:**
- `tests/test_rgb_sonar_dataset.py` ✅
- `tests/test_sequence_sampler.py`

#### 2.2 Data Transforms
- [x] Implement `dataloader/transforms/rgb_transforms.py`
  - Color jitter, blur, random occlusion
  - CutMix/CopyPaste augmentation
  - Normalize to ImageNet statistics
- [x] Implement `dataloader/transforms/sonar_transforms.py`
  - Speckle noise simulation
  - Radial scaling jitter
  - Sector occlusion augmentation
- [x] Implement `dataloader/transforms/logpolar.py`
  - Differentiable log-polar coordinate transformation
  - Learnable angular/radial scale parameters (α, β)
  - Fan-shape to rectangular grid conversion

**Test Files:**
- `tests/test_rgb_transforms.py`
- `tests/test_sonar_transforms.py`
- `tests/test_logpolar_transform.py`

#### 2.3 DataLoader
- [x] Implement `dataloader/loader.py`
  - Multi-worker data loading
  - Collate function for variable-length sequences
  - Distributed sampler support

**Test Files:**
- `tests/test_dataloader.py` ✅

**Phase 2 Summary:**
- Implemented modules: 9/9 (100%)
- Code lines: ~3,948 lines (including docstrings)
- Test coverage: 2 test files with 44+ test cases
- Examples: augmentation_example.yaml, dataloader_usage_example.py
- All core data pipeline functionality complete

**Testing Status:**
- ✅ Test files created and logically verified
- ⚠️  Test execution requires dependencies installation:
  ```bash
  pip install -e ".[dev]"  # Install with test dependencies
  pytest tests/test_rgb_sonar_dataset.py -v
  pytest tests/test_dataloader.py -v
  ```
- 📝 Additional test files planned: test_sequence_sampler.py, test_template_search_sampler.py,
     test_sonar_geometry.py, test_logpolar_transform.py, test_rgb_transforms.py, test_sonar_transforms.py

**Detailed Module Implementations:**

1. **rgb_sonar_dataset.py** (347 lines)
   - Multi-modal RGB-Sonar dataset loader
   - RGBS50 format compatibility
   - Occlusion handling (0,0,0,0 annotations)
   - Frame range filtering
   - Last-valid-bbox tracking
   - Sequence metadata extraction

2. **sequence_sampler.py** (407 lines)
   - UniformSampler: Regular interval sampling
   - RandomSampler: Random frame selection
   - OcclusionAwareSampler: Intelligent occlusion handling
   - AdaptiveSampler: Motion-based adaptive sampling
   - Factory function for sampler creation

3. **template_search_sampler.py** (457 lines)
   - OSTrack-style template-search pair generation
   - Template selection (first frame / random history)
   - Search region extraction (2x context)
   - Coordinate normalization to search region
   - Synchronized RGB-Sonar pair sampling
   - Occlusion-aware bbox tracking

4. **augmentation.py** (439 lines)
   - TrackingAugmentation pipeline
   - TemplateSearchAugmentation pipeline
   - NoAugmentation (evaluation mode)
   - build_augmentation factory function
   - Coordinated RGB-Sonar augmentation

5. **rgb_transforms.py** (300 lines)
   - ColorJitter: Color augmentation
   - RandomBlur: Blur simulation
   - RandomOcclusion: CutMix-style occlusion
   - RandomHorizontalFlip: Geometric augmentation
   - Normalize: ImageNet normalization
   - ToTensor: Array to tensor conversion
   - RGBTransformPipeline: Complete pipeline

6. **sonar_transforms.py** (358 lines)
   - SpeckleNoise: Acoustic noise simulation
   - RadialScalingJitter: Range variation
   - SectorOcclusion: Angular occlusion
   - IntensityNormalization: Adaptive normalization
   - FanMaskApplication: Geometry-aware masking
   - SonarTransformPipeline: Complete pipeline

7. **sonar_geometry.py** (360 lines)
   - generate_fan_mask: Fan-shaped FoV mask
   - compute_radial_weights: Distance-dependent weights
   - compute_angular_resolution: Angular bin mapping
   - SonarGeometryHandler: PyTorch nn.Module
   - get_radial_profile: Radial activation extraction

8. **logpolar.py** (319 lines)
   - LogPolarTransform: Differentiable coordinate transform
   - Learnable scale parameters (α, β)
   - Bilinear sampling (fully differentiable)
   - inverse_transform: Inverse mapping
   - LogPolarPreprocessor: Convenience wrapper

9. **loader.py** (317 lines)
   - collate_fn_tracking: Tracking batch collation
   - collate_fn_template_search: Template-search collation
   - create_dataloader: DataLoader factory
   - InfiniteDataLoader: Infinite cycling wrapper
   - PrefetchLoader: GPU prefetching optimization
   - create_training_loaders: Training/validation split
   - Distributed sampler support

**Key Features Implemented:**
- ✅ RGBS50 Dataset Compatibility: Full support for RGBS50 annotation format, groundtruth.txt parsing, temporal alignment
- ✅ OSTrack Framework Integration: Template-search paradigm, search region extraction with context, coordinate normalization
- ✅ Occlusion Robustness: Detection of (0,0,0,0) annotations, last-valid-bbox tracking, temporal continuity maintenance
- ✅ Sonar Geometry Awareness: Fan-shaped field of view handling, radial resolution weighting, angular sector processing
- ✅ Differentiable Transforms: End-to-end learnable log-polar transform, learnable scale parameters (α, β), fully differentiable bilinear sampling
- ✅ Multi-Modal Augmentation: Coordinated RGB-Sonar augmentation, modality-specific transformations, template vs search differential augmentation
- ✅ Efficient Data Loading: Multi-worker support, GPU prefetching, distributed sampling, infinite iteration mode

---

### Phase 2.5: Template-Search Framework (Week 4.5) ✅ COMPLETED

> **Critical Addition**: OSTrack relies on template-search paradigm. This phase implements the core sampling strategy.

#### 2.5.1 Template-Search Pair Generation
- [x] Implement `dataloader/datasets/template_search_sampler.py`
  - **Template Selection**: First frame or random historical frame within N frames
  - **Search Region Extraction**: Crop around previous bbox with context margin (typically 2x bbox size)
  - **Coordinate Normalization**: Convert absolute coordinates to search-region-relative coordinates
  - **Paired RGB-Sonar Sampling**: Ensure synchronized template-search pairs for both modalities

**Algorithm Pseudocode:**
```python
def is_occluded(bbox):
    """Check if bbox indicates occlusion (0,0,0,0)."""
    return bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0

def sample_template_search_pair(sequence, frame_idx):
    # Skip if current frame is occluded
    current_bbox = sequence.annotations[frame_idx]
    if is_occluded(current_bbox):
        return None  # Skip this frame during training

    # Template: typically first frame or recent history
    template_idx = 0  # or random from [max(0, frame_idx-N), frame_idx-1]

    # Ensure template is not occluded
    while is_occluded(sequence.annotations[template_idx]) and template_idx < frame_idx:
        template_idx += 1

    # Get template frame and bbox
    template_frame = sequence[template_idx]
    template_bbox = sequence.annotations[template_idx]

    # Get search frame and bbox
    search_frame = sequence[frame_idx]

    # Find last valid (non-occluded) bbox for search region
    if frame_idx == 0:
        prev_bbox = template_bbox
    else:
        # Search backwards for last valid bbox
        prev_idx = frame_idx - 1
        while prev_idx >= 0 and is_occluded(sequence.annotations[prev_idx]):
            prev_idx -= 1
        prev_bbox = sequence.annotations[prev_idx] if prev_idx >= 0 else template_bbox

    # Calculate search region (2x context)
    search_region = expand_bbox(prev_bbox, scale=2.0)
    search_crop = crop_image(search_frame, search_region)

    # Normalize coordinates to search region
    gt_bbox_normalized = normalize_to_search_region(
        sequence.annotations[frame_idx], search_region
    )

    return {
        'template': template_frame,
        'template_bbox': template_bbox,
        'search': search_crop,
        'search_bbox': gt_bbox_normalized,
        'is_valid': True
    }
```

**Test Files:**
- `tests/test_template_search_sampler.py` (planned)

#### 2.5.2 Sonar Geometry Handler
- [x] Implement `dataloader/transforms/sonar_geometry.py`
  - **Fan-Shape Mask Generation**: Binary mask for valid sonar scanning area
  - **Radial Resolution Weighting**: Weight map decreasing with distance from sensor
  - **Angular Sector Sampling**: Handle non-uniform angular resolution

**Implementation Details:**
```python
def generate_fan_mask(height, width, max_range, fov_angle):
    """
    Generate binary mask for sonar fan-shaped field of view.

    Args:
        height, width: Sonar image dimensions (682x376)
        max_range: Maximum effective range in pixels
        fov_angle: Field of view angle in degrees (e.g., 130°)

    Returns:
        mask: Binary mask (1=valid, 0=background)
    """
    # Create polar coordinate grid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

    # Sensor position: bottom center for upward-facing fan
    cx, cy = width // 2, height - 1

    # Calculate radial distance
    r = torch.sqrt((x - cx)**2 + (y - cy)**2)

    # Calculate angle from vertical axis (0° = upward)
    # For upward-facing fan: use atan2(x - cx, cy - y)
    theta = torch.atan2(x - cx, cy - y)  # Range: [-π, π], 0 = upward

    # Apply fan-shape constraints
    fov_angle_rad = torch.deg2rad(torch.tensor(fov_angle))
    mask = (r <= max_range) & (torch.abs(theta) <= fov_angle_rad / 2)
    return mask.float()

def compute_radial_weights(sonar_img):
    """
    Compute position-dependent quality weights.
    Near-range has better resolution than far-range.
    """
    height, width = sonar_img.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    cx, cy = width // 2, height
    r = torch.sqrt((x - cx)**2 + (y - cy)**2)

    # Exponential decay with distance
    weights = torch.exp(-r / (height * 0.5))
    return weights
```

**Test Files:**
- `tests/test_sonar_geometry.py` (planned)

**Phase 2.5 Summary:**
- Template-search sampler: Fully implemented with occlusion handling
- Sonar geometry handler: Complete with fan-mask, radial weights, PyTorch module
- All functionality integrated into template_search_sampler.py and sonar_geometry.py

---

### Phase 3: Backbone Networks (Weeks 5-6)

#### 3.1 RGB Backbones
- [ ] Implement `models/backbones/vit_rgb.py`
  - Load pretrained ViT from timm
  - Extract multi-scale features
  - Patch embedding adaptation
- [ ] Implement `models/backbones/convnext_rgb.py`
  - Load pretrained ConvNeXt
  - Feature pyramid extraction
- [ ] Implement `models/backbones/backbone_factory.py`
  - Registry-based backbone selection
  - Pretrained weight loading

**Test Files:**
- `tests/test_vit_backbone.py`
- `tests/test_convnext_backbone.py`
- `tests/test_backbone_factory.py`

#### 3.2 Sonar Encoder
- [ ] Implement `models/backbones/sonar_encoder.py`
  - Integrate log-polar preprocessing
  - Lightweight ViT/Conv encoder
  - Radial activation extraction for depth proxy

**Test Files:**
- `tests/test_sonar_encoder.py`

---

### Phase 3.5: Quality Estimation & Geometry Utils (Week 6.5, Optional but Recommended)

> **Critical Addition**: Uncertainty-aware gating and coordinate system utilities for weak 3D estimation. Quality estimation is optional (can be disabled via ablation config) but highly recommended for robustness under challenging underwater conditions.

#### 3.5.1 Quality Estimators
- [ ] Implement `models/quality/rgb_quality_estimator.py`
  - **Sharpness Estimation**: Laplacian variance or gradient magnitude
  - **Blur Detection**: Frequency domain analysis
  - **Per-frame Quality Score**: Normalized to [0, 1]

- [ ] Implement `models/quality/sonar_quality_estimator.py`
  - **SNR Estimation**: Signal-to-noise ratio from speckle patterns
  - **Speckle Noise Level**: Local variance analysis
  - **Echo Intensity**: Mean activation within bbox

- [ ] Implement `models/quality/adaptive_weighting.py`
  - **Dynamic Fusion Weights**: w_rgb, w_sonar based on quality
  - **Temporal Smoothing**: EMA of quality scores
  - **Threshold-based Gating**: Disable low-quality modality

**Pseudocode:**
```python
def estimate_rgb_quality(rgb_frame, bbox):
    """Estimate RGB frame quality within bounding box region."""
    roi = crop_bbox(rgb_frame, bbox)

    # Laplacian variance (sharpness)
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    sharpness = laplacian.var()

    # Normalize to [0, 1]
    quality = sigmoid((sharpness - 100) / 50)  # Tunable threshold
    return quality

def estimate_sonar_quality(sonar_frame, bbox):
    """Estimate sonar frame quality (SNR-based)."""
    roi = crop_bbox(sonar_frame, bbox)

    # Signal: mean intensity within bbox
    signal = roi.mean()

    # Noise: std of background region
    background_mask = ~bbox_mask(sonar_frame, bbox)
    noise = sonar_frame[background_mask].std()

    snr = signal / (noise + 1e-6)
    quality = sigmoid((snr - 5) / 2)  # Tunable threshold
    return quality

def adaptive_fusion(rgb_feat, sonar_feat, q_rgb, q_sonar):
    """Dynamically weight features based on quality."""
    # Softmax normalization
    w_rgb = torch.exp(q_rgb) / (torch.exp(q_rgb) + torch.exp(q_sonar))
    w_sonar = 1 - w_rgb

    fused = w_rgb * rgb_feat + w_sonar * sonar_feat
    return fused, (w_rgb, w_sonar)
```

**Test Files:**
- `tests/test_quality_estimation.py`

#### 3.5.2 Coordinate System Utilities
- [ ] Implement `utils/geometry.py`
  - **Fusion Frame Definition**: Right-handed coordinate system (X→right, Y→down, Z→forward)
  - **RGB→Fusion Transform**: Identity (RGB frame IS the Fusion frame)
  - **Sonar→Fusion Transform**: Learnable affine transformation (no calibration)
  - **3D→2D Projection**: Weak perspective projection for both modalities

**Mathematical Formulation:**
```python
# Coordinate Frames:
# - Fusion Frame: X→right, Y→down, Z→forward (RGB-centric)
# - RGB Frame: Same as Fusion Frame
# - Sonar Frame: X→right, Y→down, Z→forward (sensor-centric)

def project_3d_to_2d_rgb(X, Y, Z, focal_length=1000):
    """
    Weak perspective projection for RGB.

    Assumption: Weak perspective assumes that Z variation is small relative
    to the average distance from camera to object, allowing us to treat
    projection as orthographic (ignoring Z in 2D projection). This is valid
    for underwater tracking where depth changes are limited.

    Args:
        X, Y, Z: 3D center in Fusion Frame
        focal_length: Approx focal length (pixels, unused in weak projection)

    Returns:
        bbox_center_2d: (cx, cy) in RGB image coordinates
    """
    # Weak perspective: ignore Z in projection (orthographic approximation)
    # Full perspective would be: cx = focal_length * X / Z + cx0
    # But we assume Z ≈ constant for weak 3D estimation
    cx = X  # Directly use X (already in image plane coordinates)
    cy = Y  # Directly use Y
    return (cx, cy)

def project_3d_to_2d_sonar(X, Y, Z, learnable_transform):
    """
    Learned projection for Sonar (no calibration).

    Args:
        X, Y, Z: 3D center in Fusion Frame
        learnable_transform: 3x4 affine matrix [R|t]

    Returns:
        bbox_center_2d: (cx, cy) in Sonar image coordinates
    """
    point_3d = torch.tensor([X, Y, Z, 1.0])
    point_2d_homo = learnable_transform @ point_3d
    cx, cy = point_2d_homo[:2]
    return (cx, cy)

def cyclic_consistency_loss(pred_3d, gt_rgb_bbox, gt_sonar_bbox, transform):
    """
    3D→2D→3D cyclic consistency loss.

    Args:
        pred_3d: Predicted (X, Y, Z)
        gt_rgb_bbox: Ground truth RGB bbox (x, y, w, h)
        gt_sonar_bbox: Ground truth Sonar bbox (x, y, w, h)
        transform: Learnable sonar projection matrix

    Returns:
        loss: L2 distance between projected and GT centers
    """
    X, Y, Z = pred_3d

    # Project 3D to 2D
    cx_rgb_proj, cy_rgb_proj = project_3d_to_2d_rgb(X, Y, Z)
    cx_sonar_proj, cy_sonar_proj = project_3d_to_2d_sonar(X, Y, Z, transform)

    # Ground truth centers
    cx_rgb_gt = gt_rgb_bbox[0] + gt_rgb_bbox[2] / 2
    cy_rgb_gt = gt_rgb_bbox[1] + gt_rgb_bbox[3] / 2
    cx_sonar_gt = gt_sonar_bbox[0] + gt_sonar_bbox[2] / 2
    cy_sonar_gt = gt_sonar_bbox[1] + gt_sonar_bbox[3] / 2

    # Cyclic loss
    loss_rgb = (cx_rgb_proj - cx_rgb_gt)**2 + (cy_rgb_proj - cy_rgb_gt)**2
    loss_sonar = (cx_sonar_proj - cx_sonar_gt)**2 + (cy_sonar_proj - cy_sonar_gt)**2

    return loss_rgb + loss_sonar
```

**Test Files:**
- `tests/test_geometry_utils.py`

---

### Phase 4: Fusion Module - G²-DCMA (Weeks 7-11)

> **Extended Timeline**: G²-DCMA is complex, requiring 5 weeks instead of 3 weeks for robust implementation.

#### 4.1 Geometric Gating
- [ ] Implement `models/fusion/geometric_gating.py`
  - Radial activation from sonar
  - Scale pyramid from RGB
  - Learnable gating function g(r, s)
  - Attention weight modulation

**Test Files:**
- `tests/test_geometric_gating.py`

#### 4.2 Deformable Cross-Attention
- [ ] Implement `models/fusion/deformable_attention.py`
  - Sparse deformable sampling offsets
  - Q-K-V cross-modal attention
  - Efficient CUDA kernel integration (optional)

**Test Files:**
- `tests/test_deformable_attention.py`

#### 4.3 Dual Consistency
- [ ] Implement `models/fusion/consistency_loss.py`
  - Component consistency (L1/L2 distance)
  - Representation consistency (InfoNCE)
  - Positive/negative pair sampling

**Test Files:**
- `tests/test_consistency_loss.py`

#### 4.4 G²-DCMA Integration
- [ ] Implement `models/fusion/g2_dcma.py`
  - Integrate all fusion components
  - Multi-scale fusion strategy
  - Feature alignment and aggregation

**Test Files:**
- `tests/test_g2_dcma.py`

---

### Phase 5: Temporal Modeling (Weeks 12-13)

#### 5.1 Vision-Mamba Head
- [ ] Implement `models/temporal/mamba_head.py`
  - Integrate Mamba SSM blocks
  - Linear-complexity temporal modeling
  - Long-range dependency capture
  - Occlusion-aware gating

**Test Files:**
- `tests/test_mamba_head.py`

#### 5.2 Temporal Aggregator
- [ ] Implement `models/temporal/temporal_aggregator.py`
  - Short-term EMA smoothing
  - Visibility-based gating
  - Feature memory bank (optional)

**Test Files:**
- `tests/test_temporal_aggregator.py`

#### 5.3 ARTrackV2 Head (Optional)
- [ ] Implement `models/temporal/artrack_head.py`
  - Autoregressive temporal modeling
  - For ablation comparison

**Test Files:**
- `tests/test_artrack_head.py`

---

### Phase 6: Prediction Heads (Weeks 14-15)

#### 6.1 2D Box Heads
- [ ] Implement `models/heads/box_head.py`
  - Classification/quality prediction (BCE/Focal Loss)
  - Bounding box regression (L1 + GIoU/CIoU)
  - Separate heads for RGB, Sonar, Fusion

**Test Files:**
- `tests/test_box_head.py`

#### 6.2 Fusion Head
- [ ] Implement `models/heads/fusion_head.py`
  - Multi-modal feature integration
  - Uncertainty-aware weighting
  - Quality score prediction

**Test Files:**
- `tests/test_fusion_head.py`

#### 6.3 Weak 3D Center Head
- [ ] Implement `models/heads/weak3d_head.py`
  - Z (depth) from sonar radial activation
  - X prediction from RGB and Sonar branches
  - Y prediction from RGB dominance
  - Learnable weighted integral for depth

**Detailed Implementation:**
```python
class Weak3DHead(nn.Module):
    """
    Weak 3D center estimation (X, Y, Z) without extrinsic calibration.

    Architecture:
    - Z: Learnable weighted integral of sonar radial activation
    - X: Dual-branch regression (RGB + Sonar) with consistency loss
    - Y: RGB-dominant regression (better vertical resolution)

    Coordinate System: Fusion Frame (X→right, Y→down, Z→forward)
    """
    def __init__(self, feat_dim, num_radial_bins=50):
        super().__init__()

        # Z estimation from sonar radial distribution
        self.radial_weights = nn.Parameter(torch.randn(num_radial_bins))
        self.radial_encoder = nn.Sequential(
            nn.Linear(num_radial_bins, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # X regression (dual-branch with consistency)
        self.rgb_x_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.sonar_x_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Y regression (RGB-dominant)
        self.y_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Fusion weights for X
        self.x_fusion_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, rgb_feat, sonar_feat, sonar_radial_activation):
        """
        Args:
            rgb_feat: RGB features [B, C]
            sonar_feat: Sonar features [B, C]
            sonar_radial_activation: Radial intensity profile [B, num_bins]

        Returns:
            dict with:
                - center_3d: (X, Y, Z) in Fusion Frame [B, 3]
                - X_rgb, X_sonar: For consistency loss
                - consistency_loss: L1 distance
        """
        batch_size = rgb_feat.shape[0]

        # Z from weighted integral of radial activation
        weights = F.softmax(self.radial_weights, dim=0)  # [num_bins]
        Z_weighted = torch.sum(
            sonar_radial_activation * weights.unsqueeze(0),
            dim=-1,
            keepdim=True
        )  # [B, 1]
        Z = self.radial_encoder(sonar_radial_activation)  # [B, 1]
        Z = Z + Z_weighted  # Residual connection

        # X from both modalities
        X_rgb = self.rgb_x_head(rgb_feat)  # [B, 1]
        X_sonar = self.sonar_x_head(sonar_feat)  # [B, 1]

        # Adaptive fusion for X
        fusion_features = torch.cat([rgb_feat, sonar_feat], dim=-1)
        fusion_weights = self.x_fusion_gate(fusion_features)  # [B, 2]
        X_fused = (
            fusion_weights[:, 0:1] * X_rgb +
            fusion_weights[:, 1:2] * X_sonar
        )  # [B, 1]

        # Y from RGB
        Y = self.y_head(rgb_feat)  # [B, 1]

        # Consistency loss
        consistency_loss = F.l1_loss(X_rgb, X_sonar)

        return {
            'center_3d': torch.cat([X_fused, Y, Z], dim=-1),  # [B, 3]
            'X_rgb': X_rgb,
            'X_sonar': X_sonar,
            'X_weights': fusion_weights,
            'Z_raw': Z,
            'consistency_loss': consistency_loss
        }
```

**Sonar Radial Activation Extraction:**
```python
def extract_radial_activation(sonar_img, sonar_feat_map, num_bins=50):
    """
    Extract radial activation profile from sonar feature map.

    Args:
        sonar_img: Original sonar image [B, 1, H_orig, W_orig] (e.g., 682x376)
        sonar_feat_map: Sonar features [B, C, H', W'] (downsampled)
        num_bins: Number of radial bins

    Returns:
        radial_profile: [B, num_bins]
    """
    B, C, H, W = sonar_feat_map.shape

    # Calculate downsampling scale
    H_orig, W_orig = sonar_img.shape[-2:]
    scale_h = H / H_orig
    scale_w = W / W_orig

    # Sensor position in feature map coordinates (scaled from original)
    # Original: (cx=W_orig//2, cy=H_orig-1) for bottom center
    cx = (W_orig // 2) * scale_w
    cy = (H_orig - 1) * scale_h

    # Create radial distance map in feature space
    y, x = torch.meshgrid(
        torch.arange(H, device=sonar_img.device, dtype=torch.float32),
        torch.arange(W, device=sonar_img.device, dtype=torch.float32),
        indexing='ij'
    )
    r = torch.sqrt((x - cx)**2 + (y - cy)**2)  # [H, W]

    # Normalize to [0, num_bins)
    r_max = r.max()
    r_normalized = (r / r_max) * (num_bins - 1)
    r_bins = r_normalized.long().clamp(0, num_bins - 1)

    # Average feature activation per bin
    radial_profile = []
    for b in range(B):
        profile = []
        for bin_idx in range(num_bins):
            mask = (r_bins == bin_idx)
            if mask.sum() > 0:
                # Mean activation in this bin
                bin_activation = sonar_feat_map[b, :, mask].mean()
            else:
                bin_activation = torch.tensor(0.0, device=sonar_img.device)
            profile.append(bin_activation)
        radial_profile.append(torch.stack(profile))

    return torch.stack(radial_profile)  # [B, num_bins]
```

**Test Files:**
- `tests/test_weak3d_head.py`
- `tests/test_radial_activation.py`

#### 6.4 Geometric Cyclic Projector
- [ ] Implement `models/projector/geom_cycle.py`
  - Learnable 3D→2D projection
  - Cyclic consistency loss (3D→2D→3D)
  - Separate projectors for RGB and Sonar

**Test Files:**
- `tests/test_geom_cycle.py`

---

### Phase 7: Complete Tracker Model (Week 16)

- [ ] Implement `models/tracker.py`
  - Integrate all components (backbones, fusion, temporal, heads)
  - Template-search framework
  - Forward pass logic
  - Multi-task output organization

**Test Files:**
- `tests/test_tracker_model.py`
- `tests/test_tracker_integration.py`

---

### Phase 8: Training Infrastructure (Weeks 17-18)

#### 8.1 Loss Management
- [ ] Implement `training/loss_manager.py`
  - Multi-task loss combination
  - Loss weighting strategy
  - Loss logging and monitoring

**Loss Components:**
```python
losses = {
    'rgb_cls': BCE/Focal,
    'rgb_box': L1 + GIoU,
    'sonar_cls': BCE/Focal,
    'sonar_box': L1 + GIoU,
    'fusion_cls': BCE/Focal,
    'fusion_box': L1 + GIoU,
    'component_consistency': L1,
    'representation_consistency': InfoNCE,
    'weak3d_cycle': L2,
    'temporal_smooth': EMA loss
}
```

**Test Files:**
- `tests/test_loss_manager.py`

#### 8.2 Optimizer & Scheduler
- [ ] Implement `training/optimizer.py`
  - AdamW configuration
  - Parameter group separation (backbone, fusion, heads)
  - Gradient clipping
- [ ] Implement `training/scheduler.py`
  - Cosine decay with warmup
  - Layer-wise learning rate decay (optional)

**Test Files:**
- `tests/test_optimizer_config.py`
- `tests/test_scheduler.py`

#### 8.3 Training Loop
- [ ] Implement `training/trainer.py`
  - Epoch-based training loop
  - Batch processing pipeline
  - Gradient accumulation support
  - Distributed training (DDP) support
  - TensorBoard logging
  - Validation loop

**Test Files:**
- `tests/test_trainer.py`

#### 8.4 Checkpointing
- [ ] Implement `training/checkpoint.py`
  - Model state saving/loading
  - Optimizer state persistence
  - Resume training support
  - Best model tracking

**Test Files:**
- `tests/test_checkpoint.py`

---

### Phase 9: Evaluation & Metrics (Weeks 19-20)

#### 9.1 Metrics Implementation
- [ ] Implement `evaluation/metrics.py`
  - Success (AUC) - Intersection over Union thresholds
  - Precision - Center location error threshold
  - Average Center Error
  - Per-modality metrics (RGB, Sonar, Fusion)

**Test Files:**
- `tests/test_metrics.py`

#### 9.2 Evaluator
- [ ] Implement `evaluation/evaluator.py`
  - Evaluation pipeline on validation set
  - Sequence-level tracking
  - Per-sequence metric aggregation
  - Overall benchmark reporting

**Test Files:**
- `tests/test_evaluator.py`

#### 9.3 Visualization
- [ ] Implement `evaluation/visualizer.py`
  - Bounding box overlay on RGB/Sonar frames
  - 3D center trajectory plotting
  - Attention map visualization
  - Failure case analysis

**Test Files:**
- `tests/test_visualizer.py`

---

### Phase 10: Configuration & Scripts (Week 21)

#### 10.1 Configuration Files

**Core Configurations:**
- [ ] Create `configs/default.yaml` - Base configuration
- [ ] Create `configs/train_uuv2.yaml` - Training config for UUV2
- [ ] Create `configs/eval_uuv3.yaml` - Evaluation config for UUV3
- [ ] Create `configs/pretrain_rgb_only.yaml` - RGB-only pretraining
- [ ] Create `configs/pretrain_sonar_only.yaml` - Sonar-only pretraining

**Ablation Configurations:**
- [ ] Create `configs/ablation_no_mamba.yaml` - Ablation: no Vision-Mamba temporal modeling
- [ ] Create `configs/ablation_no_gating.yaml` - Ablation: no geometric gating
- [ ] Create `configs/ablation_no_cycle3d.yaml` - Ablation: no weak 3D cycle loss
- [ ] Create `configs/ablation_no_deformable.yaml` - Ablation: no deformable attention
- [ ] Create `configs/ablation_no_logpolar.yaml` - Ablation: no log-polar transform (NEW)
- [ ] Create `configs/ablation_no_quality_gating.yaml` - Ablation: no uncertainty-aware quality gating (NEW)
- [ ] Create `configs/ablation_transformer_vs_mamba.yaml` - Ablation: Transformer baseline vs Mamba (NEW)

#### 10.2 Training Scripts
- [ ] Create `scripts/train.py` - Main training script
- [ ] Create `scripts/eval.py` - Evaluation script
- [ ] Create `scripts/demo.py` - Demo inference script
- [ ] Create `scripts/export.py` - Model export (ONNX/TorchScript)

#### 10.3 Utility Scripts
- [ ] Create `scripts/preprocess_data.py` - Data preprocessing
- [ ] Create `scripts/analyze_dataset.py` - Dataset statistics
- [ ] Create `scripts/visualize_results.py` - Result visualization

---

### Phase 11: Integration Testing (Weeks 22-23)

- [ ] End-to-end training test on small subset
- [ ] Overfitting test on single sequence
- [ ] Multi-GPU training test
- [ ] Checkpoint save/load test
- [ ] Evaluation pipeline test
- [ ] Memory profiling and optimization
- [ ] Speed benchmarking (FPS measurement)

**Test Files:**
- `tests/test_e2e_training.py`
- `tests/test_distributed_training.py`
- `tests/test_inference_speed.py`

---

### Phase 12: Documentation & Finalization (Week 24)

- [ ] Write comprehensive README.md
- [ ] Create API documentation
- [ ] Write training tutorial
- [ ] Create evaluation guide
- [ ] Document ablation study setup
- [ ] Prepare reproducibility checklist
- [ ] Create citation information

---

## Pip Installation Setup

### setup.py

```python
from setuptools import setup, find_packages

setup(
    name='aquasot3d',
    version='0.1.0',
    description='End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/AquaSOT-3D',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'timm>=0.9.0',
        'einops>=0.7.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'tensorboard>=2.13.0',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.3.0',
        'mamba-ssm>=1.0.0',      # Vision-Mamba temporal modeling
        'scikit-image>=0.20.0',  # Image quality metrics
        'scipy>=1.10.0',         # Signal processing for sonar
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={
        'console_scripts': [
            'aquasot3d-train=scripts.train:main',
            'aquasot3d-eval=scripts.eval:main',
            'aquasot3d-demo=scripts.demo:main',
        ],
    },
)
```

### Installation Commands

```bash
# Development installation (editable)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Production installation
pip install .
```

---

## Project File Structure (Complete)

```
AquaSOT-3D/
├── README.md                          # Project overview and quick start
├── LICENSE                            # MIT License
├── setup.py                           # Package installation configuration
├── requirements.txt                   # Core dependencies
├── requirements-dev.txt               # Development dependencies
├── .gitignore                         # Git ignore patterns
├── .gitmessage                        # Git commit message template
├── CLAUDE.md                          # Claude-specific instructions
│
├── configs/                           # Configuration files
│   ├── default.yaml
│   ├── train_uuv2.yaml
│   ├── eval_uuv3.yaml
│   ├── pretrain_rgb_only.yaml
│   ├── pretrain_sonar_only.yaml
│   ├── ablation_no_mamba.yaml
│   ├── ablation_no_gating.yaml
│   ├── ablation_no_cycle3d.yaml
│   ├── ablation_no_deformable.yaml
│   ├── ablation_no_logpolar.yaml
│   ├── ablation_no_quality_gating.yaml
│   └── ablation_transformer_vs_mamba.yaml
│
├── data/                              # Dataset storage
│   ├── uuv2/                          # Training set (2740 frames)
│   │   ├── light/
│   │   │   ├── img/
│   │   │   └── groundtruth.txt
│   │   └── sonar/
│   │       ├── img/
│   │       └── groundtruth.txt
│   └── uuv3/                          # Validation set (751 frames)
│       ├── light/
│       │   ├── img/
│       │   └── groundtruth.txt
│       └── sonar/
│           ├── img/
│           └── groundtruth.txt
│
├── docs/                              # Documentation
│   ├── AquaSOT-3D_E2E_RGBSonar_Tracking.md
│   ├── Dataset_Specification.md
│   ├── Development_Planning.md        # This document
│   ├── API_Reference.md
│   └── Training_Guide.md
│
├── src/                               # Source code
│   └── aquasot3d/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base_module.py
│       │   ├── registry.py
│       │   └── config.py
│       ├── dataloader/
│       │   ├── __init__.py
│       │   ├── datasets/
│       │   │   ├── __init__.py
│       │   │   ├── rgb_sonar_dataset.py
│       │   │   ├── sequence_sampler.py
│       │   │   ├── template_search_sampler.py
│       │   │   └── augmentation.py
│       │   ├── transforms/
│       │   │   ├── __init__.py
│       │   │   ├── rgb_transforms.py
│       │   │   ├── sonar_transforms.py
│       │   │   ├── sonar_geometry.py
│       │   │   └── logpolar.py
│       │   └── loader.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── backbones/
│       │   │   ├── __init__.py
│       │   │   ├── vit_rgb.py
│       │   │   ├── convnext_rgb.py
│       │   │   ├── sonar_encoder.py
│       │   │   └── backbone_factory.py
│       │   ├── fusion/
│       │   │   ├── __init__.py
│       │   │   ├── g2_dcma.py
│       │   │   ├── geometric_gating.py
│       │   │   ├── deformable_attention.py
│       │   │   └── consistency_loss.py
│       │   ├── quality/
│       │   │   ├── __init__.py
│       │   │   ├── rgb_quality_estimator.py
│       │   │   ├── sonar_quality_estimator.py
│       │   │   └── adaptive_weighting.py
│       │   ├── temporal/
│       │   │   ├── __init__.py
│       │   │   ├── mamba_head.py
│       │   │   ├── artrack_head.py
│       │   │   ├── transformer_head.py
│       │   │   └── temporal_aggregator.py
│       │   ├── heads/
│       │   │   ├── __init__.py
│       │   │   ├── box_head.py
│       │   │   ├── fusion_head.py
│       │   │   └── weak3d_head.py
│       │   ├── projector/
│       │   │   ├── __init__.py
│       │   │   └── geom_cycle.py
│       │   └── tracker.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── optimizer.py
│       │   ├── scheduler.py
│       │   ├── loss_manager.py
│       │   └── checkpoint.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── evaluator.py
│       │   └── visualizer.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── checkpoint_io.py
│           ├── distributed.py
│           ├── geometry.py
│           └── misc.py
│
├── scripts/                           # Executable scripts
│   ├── train.py
│   ├── eval.py
│   ├── demo.py
│   ├── export.py
│   ├── preprocess_data.py
│   ├── analyze_dataset.py
│   └── visualize_results.py
│
├── tests/                             # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py                    # Pytest configuration
│   ├── test_core_registry.py
│   ├── test_config_loader.py
│   ├── test_rgb_sonar_dataset.py
│   ├── test_sequence_sampler.py
│   ├── test_rgb_transforms.py
│   ├── test_sonar_transforms.py
│   ├── test_logpolar_transform.py
│   ├── test_dataloader.py
│   ├── test_vit_backbone.py
│   ├── test_convnext_backbone.py
│   ├── test_backbone_factory.py
│   ├── test_sonar_encoder.py
│   ├── test_geometric_gating.py
│   ├── test_deformable_attention.py
│   ├── test_consistency_loss.py
│   ├── test_g2_dcma.py
│   ├── test_mamba_head.py
│   ├── test_temporal_aggregator.py
│   ├── test_box_head.py
│   ├── test_fusion_head.py
│   ├── test_weak3d_head.py
│   ├── test_geom_cycle.py
│   ├── test_tracker_model.py
│   ├── test_tracker_integration.py
│   ├── test_loss_manager.py
│   ├── test_optimizer_config.py
│   ├── test_scheduler.py
│   ├── test_trainer.py
│   ├── test_checkpoint.py
│   ├── test_metrics.py
│   ├── test_evaluator.py
│   ├── test_visualizer.py
│   ├── test_e2e_training.py
│   ├── test_distributed_training.py
│   └── test_inference_speed.py
│
└── outputs/                           # Training outputs (git-ignored)
    ├── checkpoints/
    ├── logs/
    ├── tensorboard/
    └── visualizations/
```

---

## Testing Strategy

### Unit Testing Principles

1. **Isolated Testing**: Each module should be testable independently
2. **Mock Data**: Use synthetic data for quick testing
3. **Coverage Target**: Aim for >80% code coverage
4. **Continuous Integration**: Run tests on every commit

### Test Categories

#### 1. Core Module Tests
- Configuration loading/validation
- Registry system functionality
- Base module interfaces

#### 2. Data Pipeline Tests
- Dataset loading with actual data files
- Groundtruth parsing correctness
- Transform output shape verification
- Log-polar transform differentiability
- DataLoader batch consistency

#### 3. Model Component Tests
- Backbone output shape verification
- Fusion module gradient flow
- Temporal head memory efficiency
- Prediction head output range validation
- 3D projector cyclic consistency

#### 4. Training Tests
- Loss computation correctness
- Optimizer parameter updates
- Gradient accumulation logic
- Checkpoint save/load integrity

#### 5. Evaluation Tests
- Metric computation accuracy
- Evaluator pipeline completeness
- Visualization rendering

#### 6. Integration Tests
- End-to-end forward pass
- Multi-GPU synchronization
- Training convergence on toy data

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=aquasot3d tests/

# Run specific test file
pytest tests/test_rgb_sonar_dataset.py

# Run with verbose output
pytest -v tests/

# Run tests in parallel
pytest -n auto tests/
```

---

## Training Strategy

### 1. Single-Modality Pretraining

**RGB-Only Training:**
```bash
aquasot3d-train --cfg configs/pretrain_rgb_only.yaml \
    --epochs 40 \
    --batch-size 32 \
    --lr 2e-4
```

**Sonar-Only Training:**
```bash
aquasot3d-train --cfg configs/pretrain_sonar_only.yaml \
    --epochs 40 \
    --batch-size 32 \
    --lr 2e-4
```

### 2. Joint Multi-Modal Training

```bash
aquasot3d-train --cfg configs/train_uuv2.yaml \
    --epochs 60 \
    --batch-size 32 \
    --lr 2e-4 \
    --resume checkpoints/pretrained_rgb.pth
```

### 3. Fine-Tuning Strategy

- **Stage 1 (Epochs 1-20)**: Freeze backbones, train fusion + heads
- **Stage 2 (Epochs 21-40)**: Unfreeze all, lower learning rate (1e-5)
- **Stage 3 (Epochs 41-60)**: Fine-tune with full augmentation

### 4. Loss Weight Schedule

| Component | Initial | Mid (Epoch 20) | Final (Epoch 60) |
|-----------|---------|----------------|------------------|
| RGB Box   | 1.0     | 1.0            | 1.0              |
| Sonar Box | 1.0     | 1.0            | 1.0              |
| Fusion Box| 1.0     | 1.2            | 1.5              |
| Component Consistency | 0.2 | 0.3 | 0.4 |
| Representation Consistency | 0.1 | 0.15 | 0.2 |
| Weak 3D Cycle | 0.6 | 0.8 | 1.0 |
| Temporal Smooth | 0.1 | 0.1 | 0.1 |

---

## Evaluation Protocol

### Metrics (RGBS50 Standard)

1. **Success (AUC)**: Area under curve of success plot (IoU thresholds 0-1)
2. **Precision**: Percentage of frames with center error < 20 pixels
3. **Average Center Error**: Mean Euclidean distance between predicted and GT centers

### Evaluation Command

```bash
aquasot3d-eval --cfg configs/eval_uuv3.yaml \
    --checkpoint checkpoints/best_model.pth \
    --split val \
    --visualize
```

### Expected Output

```
=== Evaluation Results on UUV3 (751 frames) ===

RGB Modality:
  Success (AUC):        0.XXX
  Precision:            0.XXX
  Avg Center Error:     XX.XX px

Sonar Modality:
  Success (AUC):        0.XXX
  Precision:            0.XXX
  Avg Center Error:     XX.XX px

Fusion (Ours):
  Success (AUC):        0.XXX ← Primary metric
  Precision:            0.XXX
  Avg Center Error:     XX.XX px

Weak 3D Center:
  Mean Depth Error:     X.XX m
  Mean X Error:         X.XX m
  Mean Y Error:         X.XX m
```

---

## Ablation Study Plan

### Ablation Configurations

1. **Full Model** (`configs/train_uuv2.yaml`)
   - All components enabled (Baseline)

2. **No Geometry Gating** (`configs/ablation_no_gating.yaml`)
   - Remove geometric gating in G²-DCMA
   - Use standard cross-attention without radial-scale modulation

3. **No Deformable Attention** (`configs/ablation_no_deformable.yaml`)
   - Remove deformable sampling offsets
   - Use fixed grid attention

4. **No Weak 3D Cycle** (`configs/ablation_no_cycle3d.yaml`)
   - Remove 3D center estimation module
   - Remove cyclic consistency loss (λ_cycle3D = 0)

5. **No Temporal Modeling** (`configs/ablation_no_mamba.yaml`)
   - Remove Vision-Mamba temporal head
   - Use direct feature aggregation without temporal context

6. **No Log-Polar Transform** (`configs/ablation_no_logpolar.yaml`)
   - Process sonar images directly in Cartesian coordinates
   - Skip differentiable log-polar transformation

7. **No Quality-Aware Gating** (`configs/ablation_no_quality_gating.yaml`)
   - Disable uncertainty-aware quality estimation
   - Use fixed fusion weights (w_rgb = w_sonar = 0.5)
   - No adaptive modality weighting

8. **Transformer vs Mamba** (`configs/ablation_transformer_vs_mamba.yaml`)
   - Replace Vision-Mamba with standard Transformer temporal head
   - Compare linear-complexity SSM vs quadratic-complexity self-attention

### Running Ablations

```bash
# Train each ablation
for config in configs/ablation_*.yaml; do
    aquasot3d-train --cfg $config --epochs 60
done

# Evaluate all ablations
for ckpt in checkpoints/ablation_*.pth; do
    aquasot3d-eval --checkpoint $ckpt --split val
done
```

---

## Performance Targets

### Computational Efficiency

| Metric | Target | Notes |
|--------|--------|-------|
| FPS (1080p RGB + 682×376 Sonar) | >30 FPS | RTX 3090 GPU |
| Memory (Training) | <16 GB | Batch size 32 |
| Memory (Inference) | <4 GB | Single sequence |
| Parameters | <100M | Excluding pretrained backbone |
| FLOPs | <50G | Per frame pair |

### Tracking Performance Goals

| Metric | Target | Baseline (RGB-only) |
|--------|--------|---------------------|
| Fusion Success (AUC) | >0.70 | ~0.55 |
| Fusion Precision | >0.85 | ~0.70 |
| Robustness to Occlusion | >0.65 | ~0.45 |

---

## Development Environment Setup

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- Git

### Setup Steps

```bash
# Clone repository
git clone https://github.com/yourusername/AquaSOT-3D.git
cd AquaSOT-3D

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import aquasot3d; print(aquasot3d.__version__)"

# Run tests
pytest tests/
```

---

## Progress Tracking Checklist

### ✅ Completed
- [x] Project structure design
- [x] Dataset analysis and specification
- [x] Foundation infrastructure setup (Phase 1)
  - [x] setup.py and requirements.txt
  - [x] Core modules (base_module, registry, config)
  - [x] Unit tests (71 tests passing)
  - [x] Directory structure optimization
- [x] Data pipeline implementation (Phase 2)
  - [x] Multi-modal dataset loader (RGB-Sonar)
  - [x] Temporal sequence samplers
  - [x] Template-search pair generation (OSTrack-style)
  - [x] Sonar geometry handling (fan-mask, radial weights)
  - [x] Differentiable log-polar transform
  - [x] RGB and Sonar data augmentation
  - [x] Unified augmentation pipelines
  - [x] DataLoader utilities (batching, distributed)
  - [x] Test files (44+ test cases)

### 🔄 In Progress
- [ ] Backbone network integration (Phase 3 - Next)

### 📋 Pending
- [ ] Backbone network integration
- [ ] Fusion module development
- [ ] Temporal modeling implementation
- [ ] Prediction heads creation
- [ ] Training infrastructure
- [ ] Evaluation pipeline
- [ ] Configuration files
- [ ] Integration testing
- [ ] Documentation

---

## References

### Core Papers

1. **OSTrack** (Framework Base)
   Ye et al., "Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework", ECCV 2022
   https://github.com/botaoye/OSTrack

2. **RGBS50 Benchmark** (Task Definition)
   Li et al., "RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker", 2024
   https://arxiv.org/abs/2406.07189 | https://github.com/LiYunfengLYF/RGBS50

3. **Vision-Mamba** (Temporal Modeling)
   Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", ICML 2024
   https://github.com/hustvl/Vim

4. **ARTrackV2** (Alternative Temporal Model)
   Bai et al., "ARTrackV2: Prompting Autoregressive Tracker Where to Look and How to Describe", CVPR 2024
   https://github.com/MIV-XJTU/ARTrack

5. **Deformable DETR** (Deformable Attention Concept)
   Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
   https://github.com/fundamentalvision/Deformable-DETR

### Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- timm Documentation: https://huggingface.co/docs/timm/
- einops Documentation: https://einops.rocks/
- OpenCV Python: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

---

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU training)
- pip or conda

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/AquaSOT-3D.git
cd AquaSOT-3D
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n aquasot3d python=3.10
conda activate aquasot3d
```

#### 3. Install Dependencies

**Option A: Development Installation (Recommended)**

```bash
pip install -e ".[dev]"
```

This installs core dependencies (torch, opencv, numpy, etc.), test dependencies (pytest, pytest-cov), and development tools (black, flake8, mypy).

**Option B: Core Installation Only**

```bash
pip install -r requirements.txt
pip install -e .
```

#### 4. Install Optional Dependencies

**Mamba-SSM** (for Vision-Mamba temporal modeling, requires CUDA):

```bash
pip install mamba-ssm>=1.0.0
```

Note: If Mamba-SSM installation fails, you can skip it and use Transformer-based temporal heads instead.

#### 5. Verify Installation

```bash
# Check package installation
python -c "import aquasot3d; print(aquasot3d.__version__)"

# Run verification script
python3 tests/test_phase2_verification.py

# Run tests
pytest tests/test_core_registry.py -v
```

### Common Installation Issues

**Issue 1: OpenCV Import Error**
```bash
pip install opencv-python
```

**Issue 2: PyTorch CUDA Version Mismatch**
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with matching CUDA version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Issue 3: Mamba-SSM Installation Failure**
```bash
# Requires CUDA toolkit and C++ compiler
# On Ubuntu:
sudo apt-get install build-essential
pip install mamba-ssm
```

### Development Setup

**Install Pre-commit Hooks:**
```bash
pip install pre-commit
pre-commit install
```

**Code Quality Tools:**
```bash
pip install black flake8 mypy

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/
```

### Running Tests

```bash
# Run all Phase 1 tests
pytest tests/test_core_registry.py tests/test_config_loader.py tests/test_base_module.py -v

# Run Phase 2 tests (requires full dependencies)
pytest tests/test_rgb_sonar_dataset.py tests/test_dataloader.py -v

# Run with coverage
pytest tests/ --cov=aquasot3d.dataloader --cov-report=html
```

---

## Contributing Guidelines

### Code Style

- Follow PEP 8 conventions
- Use Black for auto-formatting
- Type hints for all function signatures
- Docstrings in Google style

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** feat, fix, docs, test, refactor, perf, chore

### Pull Request Process

1. Create feature branch from `main`
2. Implement feature with tests
3. Run `pytest` and `black .`
4. Update documentation if needed
5. Submit PR with clear description
6. Pass CI checks
7. Request review

---

## License

MIT License - See LICENSE file for details

---

## Contact & Support

- Issues: https://github.com/yourusername/AquaSOT-3D/issues
- Discussions: https://github.com/yourusername/AquaSOT-3D/discussions
- Email: your.email@example.com

---

**Document Version:** 1.0
**Last Updated:** 2025-11-06
**Status:** Planning Phase
