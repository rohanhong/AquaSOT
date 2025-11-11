# AquaSOT-3D API Reference

**Version:** 0.1.0
**Date:** 2025-11-11
**License:** MIT

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Infrastructure](#core-infrastructure)
4. [Data Pipeline](#data-pipeline)
5. [Model Components](#model-components)
6. [Training Infrastructure](#training-infrastructure)
7. [Evaluation Tools](#evaluation-tools)
8. [Utility Modules](#utility-modules)
9. [Configuration System](#configuration-system)
10. [Quick Start Examples](#quick-start-examples)

---

## Overview

AquaSOT-3D provides a comprehensive API for end-to-end RGB+Sonar single-object tracking with weak 3D estimation. The system is built on a modular architecture with clear separation of concerns:

- **Core Infrastructure**: Registry system, base classes, configuration management
- **Data Pipeline**: Multi-modal data loading, transformations, and augmentation
- **Model Components**: Backbones, fusion modules, temporal heads, prediction heads
- **Training Infrastructure**: Loss management, optimization, scheduling, checkpointing
- **Evaluation Tools**: Metrics computation, evaluation pipelines, visualization

### Design Principles

1. **Modularity**: All components are independent and reusable
2. **Registry Pattern**: Dynamic component registration and lookup
3. **Configuration-Driven**: YAML-based configuration with inheritance
4. **Type Safety**: PyTorch tensor operations with shape validation
5. **Multi-Modal**: Native support for RGB and Sonar modalities

### Coordinate System Convention

All 3D operations use the Fusion Frame coordinate system:
- **X-axis**: Points right (positive direction)
- **Y-axis**: Points down (positive direction)
- **Z-axis**: Points forward (positive direction, depth/range)

Bounding boxes follow the format: `[x, y, w, h]` (top-left corner, width, height)

---

## Architecture

### System Overview

```
Input: RGB + Sonar Image Pairs
         |
         v
    [Data Pipeline]
    - Dataset loading
    - Template-search sampling
    - Transform & augmentation
         |
         v
    [Backbone Feature Extraction]
    - RGB Branch: ViT/ConvNeXt
    - Sonar Branch: Log-Polar + Encoder
         |
         v
    [G²-DCMA Fusion Module]
    - Geometry-gated attention
    - Deformable cross-modal alignment
    - Dual consistency constraints
         |
         v
    [Temporal Modeling]
    - Vision-Mamba (linear complexity)
    - Temporal feature aggregation
         |
         v
    [Prediction Heads]
    - Box Head: 2D bounding boxes (RGB, Sonar, Fusion)
    - Fusion Head: Quality-weighted fusion
    - Weak3D Head: (X, Y, Z) estimation
         |
         v
Output: Tracking boxes + Weak 3D center
```

### Module Dependencies

```
aquasot3d
├── core (base classes, registry, config)
├── dataloader (datasets, transforms, samplers)
│   └── depends on: core
├── models (backbones, fusion, temporal, heads, tracker)
│   └── depends on: core, dataloader.transforms
├── training (loss, optimizer, scheduler, checkpoint, trainer)
│   └── depends on: core, models
├── evaluation (metrics, evaluator, visualizer)
│   └── depends on: core, models
└── utils (config management, geometry utilities)
    └── depends on: (none - standalone)
```

---

## Core Infrastructure

The core infrastructure provides foundational components used across the entire system.

### Module: `aquasot3d.core`

#### Registry System

**Class: `Registry`**

Dynamic component registration and lookup system following the factory pattern.

```python
from aquasot3d.core import Registry

# Create custom registry
CUSTOM_REGISTRY = Registry(name='custom_components')

# Register component
@CUSTOM_REGISTRY.register('my_component')
class MyComponent:
    pass

# Retrieve component
ComponentClass = CUSTOM_REGISTRY.get('my_component')
instance = ComponentClass()
```

**Methods:**

- `register(name: str, force: bool = False)` - Decorator to register a class/function
- `get(name: str)` - Retrieve registered component by name
- `contains(name: str) -> bool` - Check if component is registered
- `list() -> List[str]` - List all registered component names

**Pre-defined Registries:**

```python
from aquasot3d.core import (
    BACKBONES,           # Backbone networks
    FUSION_MODULES,      # Fusion modules
    TEMPORAL_HEADS,      # Temporal modeling heads
    PREDICTION_HEADS,    # Prediction heads
    QUALITY_ESTIMATORS,  # Quality estimation modules
    TRACKERS,            # Complete tracker systems
    DATASETS,            # Dataset classes
    TRANSFORMS,          # Data transformations
    SAMPLERS,            # Data samplers
    AUGMENTATIONS,       # Data augmentation
    OPTIMIZERS,          # Optimizer builders
    SCHEDULERS,          # LR scheduler builders
    LOSSES,              # Loss functions
    METRICS,             # Evaluation metrics
    EVALUATORS           # Evaluation pipelines
)
```

**Example:**

```python
from aquasot3d.core import BACKBONES
from aquasot3d.models.backbones import ViTRGBBackbone

# Registered automatically via decorator
backbone = BACKBONES.get('vit_rgb')(
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12
)
```

---

#### Base Module Classes

**Class: `BaseModule`**

Abstract base class for all PyTorch modules in AquaSOT-3D.

```python
from aquasot3d.core import BaseModule
import torch.nn as nn

class MyModule(BaseModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize layers

    def forward(self, x):
        # Forward pass implementation
        return output
```

**Methods:**

- `forward(*args, **kwargs)` - Abstract method, must be implemented
- `init_weights()` - Initialize module weights (optional override)
- `get_config() -> Dict` - Return module configuration

**Subclasses:**

- `BaseBackbone` - Base class for backbone networks
- `BaseFusion` - Base class for fusion modules
- `BaseHead` - Base class for prediction heads

**Class: `BaseBackbone`**

Base class for feature extraction backbones.

```python
from aquasot3d.core import BaseBackbone

class MyBackbone(BaseBackbone):
    def __init__(self, input_channels=3, output_dim=768):
        super().__init__()
        # Define layers

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            features: Output features (B, D, H', W')
        """
        return features

    @property
    def output_channels(self) -> int:
        """Number of output feature channels"""
        return self._output_channels
```

**Required Properties:**

- `output_channels: int` - Number of output feature channels

**Class: `BaseFusion`**

Base class for cross-modal fusion modules.

```python
from aquasot3d.core import BaseFusion

class MyFusion(BaseFusion):
    def forward(self, rgb_features, sonar_features, **kwargs):
        """
        Args:
            rgb_features: RGB features (B, D_rgb, H, W)
            sonar_features: Sonar features (B, D_sonar, H, W)

        Returns:
            fused_features: Fused features (B, D_fused, H, W)
        """
        return fused_features
```

**Class: `BaseHead`**

Base class for prediction heads.

```python
from aquasot3d.core import BaseHead

class MyHead(BaseHead):
    def forward(self, features):
        """
        Args:
            features: Input features (B, D, H, W)

        Returns:
            predictions: Dict containing prediction outputs
        """
        return {'prediction': output}
```

---

#### Configuration System

**Class: `Config`**

Configuration container with attribute-style access (deprecated, use `ConfigDict`).

**Class: `ConfigDict`**

Modern dictionary with attribute-style access and validation.

```python
from aquasot3d.core import ConfigDict

# Create config
config = ConfigDict({
    'model': {
        'type': 'AquaSOT3DTracker',
        'backbone': {
            'rgb': {'type': 'vit', 'patch_size': 16},
            'sonar': {'type': 'sonar_encoder', 'output_dim': 384}
        }
    },
    'training': {
        'lr': 0.0002,
        'batch_size': 32,
        'epochs': 60
    }
})

# Access via attributes
print(config.model.type)  # 'AquaSOT3DTracker'
print(config.training.lr)  # 0.0002

# Access via dict keys
print(config['model']['backbone']['rgb']['type'])  # 'vit'

# Check existence
if 'model' in config:
    print("Model config exists")

# Iterate
for key, value in config.items():
    print(f"{key}: {value}")
```

**Methods:**

- `__getattr__(key: str)` - Attribute-style access
- `__getitem__(key: str)` - Dictionary-style access
- `__setitem__(key: str, value)` - Set value
- `__contains__(key: str) -> bool` - Check key existence
- `get(key: str, default=None)` - Safe get with default
- `keys()` - Get all keys
- `values()` - Get all values
- `items()` - Get key-value pairs
- `to_dict() -> Dict` - Convert to plain dictionary

---

## Data Pipeline

The data pipeline handles loading, transformation, and augmentation of multi-modal tracking data.

### Module: `aquasot3d.dataloader`

#### Dataset Classes

**Class: `RGBSonarDataset`**

Multi-modal dataset for paired RGB and Sonar tracking sequences.

```python
from aquasot3d.dataloader import RGBSonarDataset

dataset = RGBSonarDataset(
    root_dir='data/uuv2',
    sequences=['light', 'tank'],
    mode='train',
    rgb_size=(256, 256),
    sonar_size=(256, 256),
    enable_augmentation=True
)

# Get sample
sample = dataset[0]
print(sample.keys())
# ['rgb_template', 'rgb_search', 'sonar_template', 'sonar_search',
#  'rgb_bbox_template', 'rgb_bbox_search', 'sonar_bbox_template',
#  'sonar_bbox_search', 'rgb_valid', 'sonar_valid']
```

**Constructor Parameters:**

- `root_dir: str` - Root directory containing sequences
- `sequences: List[str]` - List of sequence names to load
- `mode: str` - Mode: 'train', 'val', or 'test'
- `rgb_size: Tuple[int, int]` - Target RGB image size (H, W)
- `sonar_size: Tuple[int, int]` - Target Sonar image size (H, W)
- `template_search_gap: int = 30` - Max frame gap for template-search pairs
- `enable_augmentation: bool = True` - Enable data augmentation
- `missing_bbox_strategy: str = 'zero'` - How to handle missing boxes: 'zero', 'skip', 'propagate'

**Methods:**

- `__len__() -> int` - Number of samples
- `__getitem__(idx: int) -> Dict[str, torch.Tensor]` - Get sample by index
- `get_sequence_info(seq_name: str) -> Dict` - Get metadata for sequence

**Output Sample Format:**

```python
{
    'rgb_template': Tensor(3, H, W),        # RGB template frame
    'rgb_search': Tensor(3, H, W),          # RGB search frame
    'sonar_template': Tensor(1, H, W),      # Sonar template frame
    'sonar_search': Tensor(1, H, W),        # Sonar search frame
    'rgb_bbox_template': Tensor(4,),        # [x, y, w, h]
    'rgb_bbox_search': Tensor(4,),          # [x, y, w, h]
    'sonar_bbox_template': Tensor(4,),      # [x, y, w, h]
    'sonar_bbox_search': Tensor(4,),        # [x, y, w, h]
    'rgb_valid': Tensor(1,),                # 1 if valid, 0 if missing
    'sonar_valid': Tensor(1,),              # 1 if valid, 0 if missing
}
```

---

**Class: `TemplateSearchSampler`**

Sampler for generating template-search pairs following OSTrack paradigm.

```python
from aquasot3d.dataloader.datasets import TemplateSearchSampler

sampler = TemplateSearchSampler(
    sequence_length=100,
    max_gap=30,
    template_size=128,
    search_size=256
)

# Sample pair
template_idx, search_idx = sampler.sample_pair()
```

**Constructor Parameters:**

- `sequence_length: int` - Total frames in sequence
- `max_gap: int = 30` - Maximum frame gap between template and search
- `template_size: int = 128` - Template region size
- `search_size: int = 256` - Search region size
- `random_sampling: bool = True` - Enable random sampling

**Methods:**

- `sample_pair() -> Tuple[int, int]` - Sample (template_idx, search_idx)
- `reset()` - Reset sampler state

---

#### Transforms

**Module: `aquasot3d.dataloader.transforms`**

**Class: `LogPolarTransform`**

Differentiable log-polar coordinate transformation for sonar images.

```python
from aquasot3d.dataloader.transforms import LogPolarTransform

transform = LogPolarTransform(
    output_size=(256, 256),
    max_range=10.0,
    fov_angle=90.0,
    sensor_position=(128, 256),
    learnable_scales=True
)

# Apply transform
sonar_cartesian = torch.randn(1, 1, 512, 256)  # Fan-shaped
sonar_logpolar = transform(sonar_cartesian)    # (1, 1, 256, 256)
```

**Constructor Parameters:**

- `output_size: Tuple[int, int]` - Output image size (H, W)
- `max_range: float` - Maximum range in meters
- `fov_angle: float = 90.0` - Field of view angle in degrees
- `sensor_position: Tuple[float, float] = None` - Sensor position (x, y), auto-computed if None
- `learnable_scales: bool = True` - Enable learnable scale parameters (alpha, beta)
- `interpolation: str = 'bilinear'` - Interpolation mode

**Methods:**

- `forward(x: Tensor) -> Tensor` - Apply log-polar transform
  - Input: `(B, C, H_in, W_in)` - Cartesian sonar image
  - Output: `(B, C, H_out, W_out)` - Log-polar representation

**Learnable Parameters:**

- `alpha: Tensor` - Radial scale parameter (learnable)
- `beta: Tensor` - Angular scale parameter (learnable)

**Dynamic Size Handling:**

The transform automatically adapts to different input sizes by:
1. Scaling sensor position proportionally
2. Adjusting max range based on image height
3. Preserving FOV angle and intrinsic properties

---

**Class: `RGBTransforms`**

Standard transformations for RGB images.

```python
from aquasot3d.dataloader.transforms import RGBTransforms

rgb_transform = RGBTransforms(
    mode='train',
    size=(256, 256),
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    augmentation_prob=0.8
)

# Apply
rgb_image = torch.randn(3, 480, 640)
rgb_transformed = rgb_transform(rgb_image)  # (3, 256, 256)
```

**Available Augmentations (train mode):**

- Random horizontal flip
- Color jittering (brightness, contrast, saturation, hue)
- Random Gaussian blur
- Random grayscale
- Normalization (ImageNet statistics by default)

**Class: `SonarTransforms`**

Sonar-specific transformations.

```python
from aquasot3d.dataloader.transforms import SonarTransforms

sonar_transform = SonarTransforms(
    mode='train',
    size=(256, 256),
    add_speckle_noise=True,
    noise_std=0.05
)

sonar_image = torch.randn(1, 512, 256)
sonar_transformed = sonar_transform(sonar_image)  # (1, 256, 256)
```

**Available Augmentations (train mode):**

- Speckle noise simulation
- Random sector occlusion
- Radial scaling jitter
- Intensity normalization

---

#### Data Loading

**Function: `build_dataloader`**

Factory function to create DataLoader with proper configuration.

```python
from aquasot3d.dataloader import build_dataloader

train_loader = build_dataloader(
    dataset=train_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    distributed=False
)
```

**Parameters:**

- `dataset: Dataset` - PyTorch dataset instance
- `batch_size: int` - Batch size
- `num_workers: int = 4` - Number of data loading workers
- `shuffle: bool = True` - Shuffle data
- `pin_memory: bool = True` - Pin memory for GPU transfer
- `distributed: bool = False` - Enable distributed sampling
- `drop_last: bool = False` - Drop last incomplete batch

**Returns:**

- `DataLoader` - Configured PyTorch DataLoader

---

## Model Components

The model components implement the complete neural network architecture for tracking.

### Module: `aquasot3d.models`

#### Backbone Networks

**Class: `ViTRGBBackbone`**

Vision Transformer backbone for RGB feature extraction.

```python
from aquasot3d.models.backbones import ViTRGBBackbone

backbone = ViTRGBBackbone(
    img_size=256,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    pretrained=True,
    pretrained_path='pretrained/vit_base_patch16.pth'
)

# Forward pass
rgb_input = torch.randn(2, 3, 256, 256)
features = backbone(rgb_input)  # (2, 768, 16, 16)
```

**Constructor Parameters:**

- `img_size: int = 256` - Input image size
- `patch_size: int = 16` - Patch size for tokenization
- `in_channels: int = 3` - Input channels
- `embed_dim: int = 768` - Embedding dimension
- `depth: int = 12` - Number of transformer blocks
- `num_heads: int = 12` - Number of attention heads
- `mlp_ratio: float = 4.0` - MLP expansion ratio
- `drop_rate: float = 0.0` - Dropout rate
- `attn_drop_rate: float = 0.0` - Attention dropout rate
- `drop_path_rate: float = 0.1` - Stochastic depth rate
- `pretrained: bool = True` - Load pretrained weights
- `pretrained_path: str = None` - Path to pretrained checkpoint

**Forward Pass:**

```python
def forward(self, x: Tensor) -> Tensor:
    """
    Args:
        x: Input images (B, 3, H, W)

    Returns:
        features: (B, embed_dim, H//patch_size, W//patch_size)
    """
```

**Output Shape:**

- Input: `(B, 3, H, W)`
- Output: `(B, embed_dim, H // patch_size, W // patch_size)`

---

**Class: `ConvNeXtRGBBackbone`**

ConvNeXt backbone for RGB feature extraction (alternative to ViT).

```python
from aquasot3d.models.backbones import ConvNeXtRGBBackbone

backbone = ConvNeXtRGBBackbone(
    in_channels=3,
    depths=[3, 3, 27, 3],
    dims=[128, 256, 512, 1024],
    drop_path_rate=0.1,
    layer_scale_init_value=1e-6,
    pretrained=True
)

features = backbone(rgb_input)  # (2, 1024, H/32, W/32)
```

**Constructor Parameters:**

- `in_channels: int = 3` - Input channels
- `depths: List[int] = [3, 3, 27, 3]` - Depth of each stage
- `dims: List[int] = [128, 256, 512, 1024]` - Feature dimensions per stage
- `drop_path_rate: float = 0.1` - Stochastic depth rate
- `layer_scale_init_value: float = 1e-6` - Layer scale initialization
- `pretrained: bool = True` - Load pretrained weights

**Output Channels:** `dims[-1]` (default: 1024)

---

**Class: `SonarEncoder`**

Specialized encoder for sonar data with log-polar preprocessing.

```python
from aquasot3d.models.backbones import SonarEncoder

encoder = SonarEncoder(
    in_channels=1,
    output_dim=384,
    use_logpolar=True,
    logpolar_size=(256, 256),
    encoder_type='convnext_tiny',
    pretrained=False
)

sonar_input = torch.randn(2, 1, 512, 256)  # Fan-shaped
features = encoder(sonar_input)  # (2, 384, H', W')
```

**Constructor Parameters:**

- `in_channels: int = 1` - Input channels (1 for sonar)
- `output_dim: int = 384` - Output feature dimension
- `use_logpolar: bool = True` - Enable log-polar preprocessing
- `logpolar_size: Tuple[int, int] = (256, 256)` - Log-polar output size
- `encoder_type: str = 'convnext_tiny'` - Encoder architecture
- `pretrained: bool = False` - Load pretrained weights

**Forward Pass:**

```python
def forward(self, x: Tensor) -> Tensor:
    """
    Args:
        x: Sonar images (B, 1, H, W) - Can be varying sizes

    Returns:
        features: (B, output_dim, H', W')
    """
```

**Dynamic Size Support:**

The encoder handles varying sonar image sizes automatically through:
- Dynamic log-polar transformation
- Adaptive pooling layers

---

**Class: `MultiModalBackbone`**

Combined backbone for joint RGB and Sonar processing.

```python
from aquasot3d.models.backbones import MultiModalBackbone

backbone = MultiModalBackbone(
    rgb_backbone_config={
        'type': 'vit',
        'patch_size': 16,
        'embed_dim': 768
    },
    sonar_backbone_config={
        'type': 'sonar_encoder',
        'output_dim': 384,
        'use_logpolar': True
    }
)

# Forward pass
rgb = torch.randn(2, 3, 256, 256)
sonar = torch.randn(2, 1, 512, 256)
rgb_feat, sonar_feat = backbone(rgb, sonar)
```

**Constructor Parameters:**

- `rgb_backbone_config: Dict` - RGB backbone configuration
- `sonar_backbone_config: Dict` - Sonar backbone configuration
- `feature_alignment: str = 'interpolate'` - How to align feature maps

**Forward Pass:**

```python
def forward(
    self,
    rgb: Tensor,
    sonar: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        rgb: RGB images (B, 3, H_rgb, W_rgb)
        sonar: Sonar images (B, 1, H_sonar, W_sonar)

    Returns:
        rgb_features: (B, D_rgb, H', W')
        sonar_features: (B, D_sonar, H', W')
    """
```

---

**Factory Function: `build_backbone`**

```python
from aquasot3d.models.backbones import build_backbone

backbone = build_backbone(
    backbone_type='vit',
    config={
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'pretrained': True
    }
)
```

**Parameters:**

- `backbone_type: str` - Type: 'vit', 'convnext', 'sonar_encoder'
- `config: Dict` - Backbone configuration

**Supported Types:**

- `'vit'` - Vision Transformer (ViTRGBBackbone)
- `'convnext'` - ConvNeXt (ConvNeXtRGBBackbone)
- `'sonar_encoder'` - Sonar Encoder (SonarEncoder)

---

#### Fusion Modules

**Module: `aquasot3d.models.fusion`**

**Class: `G2DCMA`**

Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA) module.

```python
from aquasot3d.models.fusion import G2DCMA

fusion = G2DCMA(
    rgb_dim=768,
    sonar_dim=384,
    fusion_dim=512,
    num_heads=8,
    num_deformable_points=4,
    use_geometric_gating=True,
    use_deformable_attention=True,
    dropout=0.1
)

# Forward pass
rgb_feat = torch.randn(2, 768, 16, 16)
sonar_feat = torch.randn(2, 384, 16, 16)
fused_feat = fusion(rgb_feat, sonar_feat)  # (2, 512, 16, 16)
```

**Constructor Parameters:**

- `rgb_dim: int` - RGB feature dimension
- `sonar_dim: int` - Sonar feature dimension
- `fusion_dim: int` - Output fusion dimension
- `num_heads: int = 8` - Number of attention heads
- `num_deformable_points: int = 4` - Number of deformable sampling points
- `use_geometric_gating: bool = True` - Enable geometry-aware gating
- `use_deformable_attention: bool = True` - Enable deformable attention
- `dropout: float = 0.1` - Dropout rate

**Forward Pass:**

```python
def forward(
    self,
    rgb_features: Tensor,
    sonar_features: Tensor,
    rgb_quality: Optional[Tensor] = None,
    sonar_quality: Optional[Tensor] = None
) -> Tensor:
    """
    Args:
        rgb_features: (B, D_rgb, H, W)
        sonar_features: (B, D_sonar, H, W)
        rgb_quality: Optional (B, 1) quality scores
        sonar_quality: Optional (B, 1) quality scores

    Returns:
        fused_features: (B, fusion_dim, H, W)
    """
```

**Key Features:**

1. **Geometric Gating**: Uses radial distance and scale priors to gate attention
2. **Deformable Attention**: Learnable sampling offsets for spatial alignment
3. **Dual Consistency**: Component and representation consistency constraints

---

**Class: `GeometricGating`**

Geometry-aware gating mechanism for cross-modal fusion.

```python
from aquasot3d.models.fusion import GeometricGating

gating = GeometricGating(
    input_dim=512,
    hidden_dim=256,
    use_radial_prior=True,
    use_scale_prior=True
)

# Compute gating weights
rgb_feat = torch.randn(2, 512, 16, 16)
sonar_feat = torch.randn(2, 512, 16, 16)
gate_weights = gating(rgb_feat, sonar_feat)  # (2, 1, 16, 16)
```

**Constructor Parameters:**

- `input_dim: int` - Input feature dimension
- `hidden_dim: int = 256` - Hidden layer dimension
- `use_radial_prior: bool = True` - Use sonar radial distance prior
- `use_scale_prior: bool = True` - Use RGB scale prior

**Output:** Gating weights in range [0, 1]

---

**Class: `DeformableAttention`**

Deformable cross-attention for spatial alignment.

```python
from aquasot3d.models.fusion import DeformableAttention

attention = DeformableAttention(
    embed_dim=512,
    num_heads=8,
    num_points=4,
    dropout=0.1
)

# Cross-attention
query = torch.randn(2, 512, 256)  # (B, D, N_query)
key = torch.randn(2, 512, 256)     # (B, D, N_key)
value = torch.randn(2, 512, 256)   # (B, D, N_value)
output = attention(query, key, value)  # (2, 512, 256)
```

**Constructor Parameters:**

- `embed_dim: int` - Embedding dimension
- `num_heads: int = 8` - Number of attention heads
- `num_points: int = 4` - Number of sampling points per head
- `dropout: float = 0.1` - Dropout rate

---

**Class: `ConsistencyLoss`**

Dual consistency constraints (component + representation).

```python
from aquasot3d.models.fusion import ConsistencyLoss

loss_fn = ConsistencyLoss(
    component_weight=0.2,
    representation_weight=0.1,
    temperature=0.07
)

# Compute loss
rgb_3d = torch.randn(2, 3)  # RGB 3D predictions
sonar_3d = torch.randn(2, 3)  # Sonar 3D predictions
rgb_repr = torch.randn(2, 512)  # RGB representations
sonar_repr = torch.randn(2, 512)  # Sonar representations

loss_dict = loss_fn(rgb_3d, sonar_3d, rgb_repr, sonar_repr)
# {'component_loss': ..., 'representation_loss': ..., 'total_loss': ...}
```

**Constructor Parameters:**

- `component_weight: float = 0.2` - Weight for component consistency (3D)
- `representation_weight: float = 0.1` - Weight for representation consistency (InfoNCE)
- `temperature: float = 0.07` - Temperature for InfoNCE loss

---

#### Temporal Modeling

**Module: `aquasot3d.models.temporal`**

**Class: `VisionMambaHead`**

Vision-Mamba temporal head for linear-complexity sequence modeling.

```python
from aquasot3d.models.temporal import VisionMambaHead

temporal_head = VisionMambaHead(
    input_dim=512,
    hidden_dim=512,
    num_layers=4,
    d_state=16,
    d_conv=4,
    expand_factor=2
)

# Process sequence
features = torch.randn(2, 10, 512, 16, 16)  # (B, T, D, H, W)
temporal_output = temporal_head(features)  # (2, 512, 16, 16)
```

**Constructor Parameters:**

- `input_dim: int` - Input feature dimension
- `hidden_dim: int = 512` - Hidden dimension
- `num_layers: int = 4` - Number of Mamba layers
- `d_state: int = 16` - State dimension for SSM
- `d_conv: int = 4` - Convolution kernel size
- `expand_factor: int = 2` - Expansion factor for inner dimension

**Forward Pass:**

```python
def forward(self, x: Tensor) -> Tensor:
    """
    Args:
        x: Sequential features (B, T, D, H, W)

    Returns:
        output: Temporally aggregated features (B, D, H, W)
    """
```

**Advantages:**

- Linear complexity O(T) vs Transformer's O(T²)
- Effective for long sequences
- Maintains spatial structure

---

**Class: `TransformerTemporalHead`**

Transformer-based temporal head (baseline comparison).

```python
from aquasot3d.models.temporal import TransformerTemporalHead

temporal_head = TransformerTemporalHead(
    input_dim=512,
    num_heads=8,
    num_layers=4,
    dropout=0.1,
    max_seq_len=50
)

temporal_output = temporal_head(features)  # (2, 512, 16, 16)
```

**Constructor Parameters:**

- `input_dim: int` - Input feature dimension
- `num_heads: int = 8` - Number of attention heads
- `num_layers: int = 4` - Number of transformer layers
- `dropout: float = 0.1` - Dropout rate
- `max_seq_len: int = 50` - Maximum sequence length

---

**Class: `TemporalAggregator`**

Lightweight temporal feature aggregation.

```python
from aquasot3d.models.temporal import TemporalAggregator

aggregator = TemporalAggregator(
    method='ema',
    alpha=0.9,
    window_size=5
)

# Aggregate over time
current_feat = torch.randn(2, 512, 16, 16)
aggregated_feat = aggregator.update(current_feat)  # (2, 512, 16, 16)
```

**Aggregation Methods:**

- `'ema'` - Exponential moving average
- `'window'` - Sliding window average
- `'attention'` - Attention-weighted aggregation

---

#### Prediction Heads

**Module: `aquasot3d.models.heads`**

**Class: `BoxHead`**

2D bounding box prediction head.

```python
from aquasot3d.models.heads import BoxHead

box_head = BoxHead(
    input_dim=512,
    hidden_dim=256,
    num_layers=3,
    use_cls=True,
    use_quality=True
)

# Predict boxes
features = torch.randn(2, 512, 16, 16)
predictions = box_head(features)
# {
#     'pred_boxes': (2, 4),  # [x, y, w, h]
#     'pred_cls': (2, 1),    # classification score
#     'pred_quality': (2, 1)  # quality score
# }
```

**Constructor Parameters:**

- `input_dim: int` - Input feature dimension
- `hidden_dim: int = 256` - Hidden layer dimension
- `num_layers: int = 3` - Number of MLP layers
- `use_cls: bool = True` - Predict classification score
- `use_quality: bool = True` - Predict quality score

**Output Format:**

```python
{
    'pred_boxes': Tensor(B, 4),      # [x, y, w, h] in normalized coords
    'pred_cls': Tensor(B, 1),        # Classification confidence [0, 1]
    'pred_quality': Tensor(B, 1)     # Quality score [0, 1]
}
```

---

**Class: `FusionHead`**

Multi-modal fusion head for combining RGB and Sonar predictions.

```python
from aquasot3d.models.heads import FusionHead

fusion_head = FusionHead(
    rgb_dim=768,
    sonar_dim=384,
    fusion_dim=512,
    use_quality_weighting=True,
    use_uncertainty_gating=True
)

# Fuse predictions
rgb_feat = torch.randn(2, 768, 16, 16)
sonar_feat = torch.randn(2, 384, 16, 16)
rgb_quality = torch.tensor([[0.9], [0.7]])
sonar_quality = torch.tensor([[0.8], [0.6]])

predictions = fusion_head(
    rgb_feat, sonar_feat,
    rgb_quality=rgb_quality,
    sonar_quality=sonar_quality
)
# {
#     'fusion_boxes': (2, 4),
#     'fusion_cls': (2, 1),
#     'fusion_quality': (2, 1),
#     'rgb_weight': (2, 1),
#     'sonar_weight': (2, 1)
# }
```

**Constructor Parameters:**

- `rgb_dim: int` - RGB feature dimension
- `sonar_dim: int` - Sonar feature dimension
- `fusion_dim: int` - Fusion dimension
- `use_quality_weighting: bool = True` - Use quality-based weighting
- `use_uncertainty_gating: bool = True` - Use uncertainty-aware gating

**Fusion Strategy:**

```
fusion_weight = softmax([w_rgb, w_sonar])
where w_i = quality_i * (1 - is_occluded_i)
```

---

**Class: `Weak3DHead`**

Weak 3D center estimation head (X, Y, Z).

```python
from aquasot3d.models.heads import Weak3DHead

weak3d_head = Weak3DHead(
    rgb_dim=768,
    sonar_dim=384,
    hidden_dim=256,
    use_cyclic_consistency=True,
    coordinate_system='fusion'
)

# Predict 3D center
rgb_feat = torch.randn(2, 768, 16, 16)
sonar_feat = torch.randn(2, 384, 16, 16)
rgb_bbox = torch.randn(2, 4)
sonar_bbox = torch.randn(2, 4)

predictions = weak3d_head(
    rgb_feat, sonar_feat,
    rgb_bbox=rgb_bbox,
    sonar_bbox=sonar_bbox
)
# {
#     'weak_3d_center': (2, 3),  # [X, Y, Z] in Fusion Frame
#     'rgb_3d': (2, 3),           # RGB branch 3D prediction
#     'sonar_3d': (2, 3),         # Sonar branch 3D prediction
#     'cyclic_loss': scalar       # 3D→2D cyclic consistency loss
# }
```

**Constructor Parameters:**

- `rgb_dim: int` - RGB feature dimension
- `sonar_dim: int` - Sonar feature dimension
- `hidden_dim: int = 256` - Hidden layer dimension
- `use_cyclic_consistency: bool = True` - Enable 3D→2D cyclic loss
- `coordinate_system: str = 'fusion'` - Coordinate frame: 'fusion', 'rgb', 'sonar'

**3D Center Estimation:**

- **X (horizontal)**: Regressed from RGB and Sonar branches, enforced with consistency loss
- **Y (vertical)**: Primarily from RGB (dominant modality for vertical)
- **Z (depth/range)**: Extracted from Sonar radial activation distribution

**Cyclic Consistency:**

The head enforces 3D→2D→3D consistency:
1. Predict 3D center from features and 2D boxes
2. Project 3D center back to 2D boxes (both RGB and Sonar)
3. Compute reprojection error as cyclic loss

---

#### Complete Tracker

**Class: `AquaSOT3DTracker`**

Complete end-to-end tracking system.

```python
from aquasot3d.models import AquaSOT3DTracker

tracker = AquaSOT3DTracker(
    backbone_config={
        'rgb': {'type': 'vit', 'embed_dim': 768},
        'sonar': {'type': 'sonar_encoder', 'output_dim': 384}
    },
    fusion_config={
        'fusion_dim': 512,
        'num_heads': 8,
        'use_geometric_gating': True
    },
    temporal_config={
        'type': 'vision_mamba',
        'num_layers': 4
    },
    head_config={
        'use_weak3d': True,
        'use_cyclic_consistency': True
    }
)

# Forward pass
rgb_template = torch.randn(2, 3, 128, 128)
rgb_search = torch.randn(2, 3, 256, 256)
sonar_template = torch.randn(2, 1, 256, 128)
sonar_search = torch.randn(2, 1, 512, 256)

outputs = tracker(
    rgb_template=rgb_template,
    rgb_search=rgb_search,
    sonar_template=sonar_template,
    sonar_search=sonar_search,
    rgb_bbox_template=torch.randn(2, 4),
    sonar_bbox_template=torch.randn(2, 4)
)
```

**Constructor Parameters:**

- `backbone_config: Dict` - Backbone configuration (RGB + Sonar)
- `fusion_config: Dict` - G²-DCMA fusion configuration
- `temporal_config: Dict` - Temporal head configuration
- `head_config: Dict` - Prediction heads configuration
- `freeze_backbone: bool = False` - Freeze backbone weights

**Forward Pass Output:**

```python
{
    # Primary outputs
    'fusion_boxes': Tensor(B, 4),     # Main tracking boxes
    'weak_3d_center': Tensor(B, 3),   # Weak 3D center (X, Y, Z)

    # Auxiliary outputs
    'rgb_boxes': Tensor(B, 4),        # RGB branch boxes
    'sonar_boxes': Tensor(B, 4),      # Sonar branch boxes
    'rgb_cls': Tensor(B, 1),          # RGB classification
    'sonar_cls': Tensor(B, 1),        # Sonar classification
    'fusion_cls': Tensor(B, 1),       # Fusion classification

    # Quality scores
    'rgb_quality': Tensor(B, 1),      # RGB quality
    'sonar_quality': Tensor(B, 1),    # Sonar quality
    'fusion_quality': Tensor(B, 1),   # Fusion quality

    # Fusion weights
    'rgb_weight': Tensor(B, 1),       # RGB contribution weight
    'sonar_weight': Tensor(B, 1),     # Sonar contribution weight

    # Losses (for training)
    'losses': {
        'rgb_cls_loss': ...,
        'rgb_box_loss': ...,
        'sonar_cls_loss': ...,
        'sonar_box_loss': ...,
        'fusion_cls_loss': ...,
        'fusion_box_loss': ...,
        'component_consistency_loss': ...,
        'representation_consistency_loss': ...,
        'weak3d_cyclic_loss': ...,
        'total_loss': ...
    }
}
```

**Factory Function:**

```python
from aquasot3d.models import build_tracker

tracker = build_tracker(config_dict_or_path)
```

---

## Training Infrastructure

### Module: `aquasot3d.training`

#### Loss Management

**Class: `LossManager`**

Multi-task loss management with dynamic weighting.

```python
from aquasot3d.training import LossManager

loss_manager = LossManager(
    loss_config={
        'rgb_cls': {'type': 'focal', 'weight': 1.0},
        'rgb_box': {'type': 'giou', 'weight': 2.0},
        'sonar_cls': {'type': 'focal', 'weight': 1.0},
        'sonar_box': {'type': 'giou', 'weight': 2.0},
        'fusion_cls': {'type': 'focal', 'weight': 1.5},
        'fusion_box': {'type': 'giou', 'weight': 3.0},
        'component_consistency': {'type': 'l1', 'weight': 0.2},
        'representation_consistency': {'type': 'infonce', 'weight': 0.1},
        'weak3d_cyclic': {'type': 'l2', 'weight': 0.6}
    },
    dynamic_weighting=True
)

# Compute losses
predictions = tracker(...)
ground_truth = {...}
losses = loss_manager(predictions, ground_truth)
# {'total_loss': ..., 'rgb_cls_loss': ..., ...}
```

**Constructor Parameters:**

- `loss_config: Dict[str, Dict]` - Loss configuration per component
- `dynamic_weighting: bool = True` - Enable dynamic weight adjustment
- `use_uncertainty_weighting: bool = False` - Use uncertainty-based weighting

**Loss Types:**

- Classification: `'bce'`, `'focal'`, `'quality_focal'`
- Bounding Box: `'l1'`, `'smooth_l1'`, `'giou'`, `'ciou'`
- Consistency: `'l1'`, `'l2'`, `'smooth_l1'`, `'infonce'`

---

#### Optimizer Configuration

**Function: `build_optimizer`**

Build optimizer with parameter grouping and weight decay rules.

```python
from aquasot3d.training import build_optimizer

optimizer = build_optimizer(
    model=tracker,
    optimizer_type='adamw',
    base_lr=0.0002,
    weight_decay=0.0001,
    backbone_lr_mult=0.1,      # Backbone uses 10% of base_lr
    no_decay_keywords=['bias', 'norm']
)
```

**Parameters:**

- `model: nn.Module` - Model to optimize
- `optimizer_type: str` - Optimizer: 'adamw', 'adam', 'sgd'
- `base_lr: float` - Base learning rate
- `weight_decay: float = 0.0001` - Weight decay coefficient
- `backbone_lr_mult: float = 0.1` - Backbone LR multiplier
- `no_decay_keywords: List[str]` - Keywords for no weight decay
- `**kwargs` - Additional optimizer arguments

**Supported Optimizers:**

- `'adamw'` - AdamW (recommended)
- `'adam'` - Adam
- `'sgd'` - SGD with momentum

---

**Function: `get_parameter_groups`**

Create parameter groups with custom learning rates and weight decay.

```python
from aquasot3d.training import get_parameter_groups

param_groups = get_parameter_groups(
    model=tracker,
    base_lr=0.0002,
    backbone_lr_mult=0.1,
    weight_decay=0.0001,
    no_decay_keywords=['bias', 'norm', 'logpolar']
)

optimizer = torch.optim.AdamW(param_groups)
```

**Parameter Groups:**

1. Backbone with decay: LR = base_lr * backbone_lr_mult, WD = weight_decay
2. Backbone no decay: LR = base_lr * backbone_lr_mult, WD = 0
3. Other with decay: LR = base_lr, WD = weight_decay
4. Other no decay: LR = base_lr, WD = 0

---

**Function: `clip_gradients`**

Gradient clipping to prevent explosion.

```python
from aquasot3d.training import clip_gradients

# By norm (recommended)
clip_gradients(model.parameters(), max_norm=1.0, norm_type=2)

# By value
clip_gradients(model.parameters(), clip_value=0.5)
```

---

#### Learning Rate Scheduling

**Function: `build_scheduler`**

Build learning rate scheduler with warmup.

```python
from aquasot3d.training import build_scheduler

scheduler = build_scheduler(
    optimizer=optimizer,
    scheduler_type='cosine',
    max_epoch=60,
    warmup_epoch=5,
    min_lr=1e-7,
    warmup_method='linear'
)

# Step per epoch
for epoch in range(max_epoch):
    train_one_epoch(...)
    scheduler.step()
```

**Parameters:**

- `optimizer: Optimizer` - PyTorch optimizer
- `scheduler_type: str` - Scheduler: 'cosine', 'step', 'poly', 'exponential'
- `max_epoch: int` - Total training epochs
- `warmup_epoch: int = 0` - Warmup epochs
- `min_lr: float = 0` - Minimum learning rate
- `warmup_method: str = 'linear'` - Warmup: 'linear', 'constant'

**Supported Schedulers:**

- `'cosine'` - Cosine annealing (recommended)
- `'step'` - Step decay
- `'poly'` - Polynomial decay
- `'exponential'` - Exponential decay

---

**Class: `CosineAnnealingWarmupRestarts`**

Cosine annealing with warmup and restarts.

```python
from aquasot3d.training import CosineAnnealingWarmupRestarts

scheduler = CosineAnnealingWarmupRestarts(
    optimizer=optimizer,
    first_cycle_steps=30,
    cycle_mult=1.0,
    max_lr=0.0002,
    min_lr=1e-7,
    warmup_steps=5,
    gamma=1.0
)

# Step per epoch
scheduler.step()
```

---

#### Checkpoint Management

**Class: `CheckpointManager`**

Comprehensive checkpoint saving and loading.

```python
from aquasot3d.training import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir='outputs/checkpoints',
    keep_last_n=5,
    save_best=True,
    monitor_metric='success_auc',
    mode='max'
)

# Save checkpoint
checkpoint_manager.save(
    epoch=10,
    model=tracker,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics={'success_auc': 0.75},
    is_best=True
)

# Load checkpoint
checkpoint = checkpoint_manager.load(
    checkpoint_path='outputs/checkpoints/best_model.pth',
    model=tracker,
    optimizer=optimizer,
    scheduler=scheduler,
    strict=True
)
epoch = checkpoint['epoch']
```

**Constructor Parameters:**

- `checkpoint_dir: str` - Directory for checkpoints
- `keep_last_n: int = 5` - Keep last N checkpoints
- `save_best: bool = True` - Save best model separately
- `monitor_metric: str = 'success_auc'` - Metric to monitor for best model
- `mode: str = 'max'` - 'max' or 'min'

**Methods:**

- `save(epoch, model, optimizer, scheduler, metrics, **kwargs)` - Save checkpoint
- `load(checkpoint_path, model, optimizer, scheduler, strict)` - Load checkpoint
- `get_best_checkpoint() -> str` - Get path to best checkpoint
- `get_latest_checkpoint() -> str` - Get path to latest checkpoint

---

#### Training Loop

**Class: `Trainer`**

Complete training loop with AMP, DDP, logging.

```python
from aquasot3d.training import Trainer

trainer = Trainer(
    model=tracker,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_manager=loss_manager,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_manager=checkpoint_manager,
    device='cuda',
    max_epochs=60,
    gradient_accumulation_steps=1,
    use_amp=True,
    use_ddp=False,
    log_interval=10,
    val_interval=1,
    tensorboard_dir='outputs/tensorboard'
)

# Run training
trainer.train()
```

**Constructor Parameters:**

- `model: nn.Module` - Model to train
- `optimizer: Optimizer` - Optimizer
- `scheduler: _LRScheduler` - Learning rate scheduler
- `loss_manager: LossManager` - Loss management
- `train_loader: DataLoader` - Training data loader
- `val_loader: DataLoader` - Validation data loader
- `checkpoint_manager: CheckpointManager` - Checkpoint management
- `device: str = 'cuda'` - Device
- `max_epochs: int = 60` - Maximum epochs
- `gradient_accumulation_steps: int = 1` - Gradient accumulation
- `use_amp: bool = True` - Use automatic mixed precision
- `use_ddp: bool = False` - Use distributed data parallel
- `log_interval: int = 10` - Logging frequency (iterations)
- `val_interval: int = 1` - Validation frequency (epochs)
- `tensorboard_dir: str = None` - TensorBoard log directory

**Methods:**

- `train()` - Run complete training
- `train_one_epoch(epoch)` - Train single epoch
- `validate(epoch)` - Run validation
- `resume(checkpoint_path)` - Resume from checkpoint

**Factory Function:**

```python
from aquasot3d.training import build_trainer

trainer = build_trainer(config)
trainer.train()
```

---

## Evaluation Tools

### Module: `aquasot3d.evaluation`

#### Metrics

**Function: `compute_iou`**

Compute Intersection over Union (IoU) between bounding boxes.

```python
from aquasot3d.evaluation import compute_iou

boxes1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]])
boxes2 = torch.tensor([[15, 15, 45, 45], [25, 25, 65, 65]])
iou = compute_iou(boxes1, boxes2)  # (2,)
```

**Parameters:**

- `boxes1: Tensor` - Boxes (N, 4) in format [x, y, w, h]
- `boxes2: Tensor` - Boxes (N, 4) in format [x, y, w, h]

**Returns:**

- `iou: Tensor` - IoU scores (N,)

---

**Function: `compute_center_error`**

Compute center location error in pixels.

```python
from aquasot3d.evaluation import compute_center_error

pred_boxes = torch.tensor([[10, 10, 50, 50]])
gt_boxes = torch.tensor([[12, 11, 50, 50]])
error = compute_center_error(pred_boxes, gt_boxes)  # (1,)
```

---

**Function: `compute_success_curve`**

Compute success curve (IoU thresholds vs success rate).

```python
from aquasot3d.evaluation import compute_success_curve

ious = torch.tensor([0.3, 0.5, 0.7, 0.8, 0.9])
thresholds, success_rates = compute_success_curve(ious)
# thresholds: [0.0, 0.05, 0.1, ..., 1.0]
# success_rates: [%, %, ...]
```

---

**Function: `compute_success_auc`**

Compute Area Under Curve (AUC) for success plot.

```python
from aquasot3d.evaluation import compute_success_auc

ious = torch.tensor([0.3, 0.5, 0.7, 0.8, 0.9])
auc = compute_success_auc(ious)  # Scalar value
```

---

**Function: `compute_precision_curve`**

Compute precision curve (distance thresholds vs precision).

```python
from aquasot3d.evaluation import compute_precision_curve

center_errors = torch.tensor([5.0, 10.0, 15.0, 20.0, 25.0])
thresholds, precisions = compute_precision_curve(center_errors)
```

---

**Function: `compute_precision_at_threshold`**

Compute precision at specific threshold (default 20 pixels).

```python
from aquasot3d.evaluation import compute_precision_at_threshold

center_errors = torch.tensor([5.0, 10.0, 15.0, 20.0, 25.0])
precision = compute_precision_at_threshold(center_errors, threshold=20.0)
```

---

**Function: `compute_sequence_metrics`**

Compute all metrics for a single sequence.

```python
from aquasot3d.evaluation import compute_sequence_metrics

pred_boxes = torch.randn(100, 4)
gt_boxes = torch.randn(100, 4)

metrics = compute_sequence_metrics(pred_boxes, gt_boxes)
# {
#     'success_auc': ...,
#     'precision': ...,
#     'center_error_mean': ...,
#     'center_error_std': ...,
#     'fps': ...
# }
```

---

**Function: `compute_multi_modal_metrics`**

Compute metrics for all modalities (RGB, Sonar, Fusion).

```python
from aquasot3d.evaluation import compute_multi_modal_metrics

predictions = {
    'rgb_boxes': rgb_pred,
    'sonar_boxes': sonar_pred,
    'fusion_boxes': fusion_pred
}

ground_truth = {
    'rgb_boxes': rgb_gt,
    'sonar_boxes': sonar_gt
}

metrics = compute_multi_modal_metrics(predictions, ground_truth)
# {
#     'rgb': {'success_auc': ..., 'precision': ...},
#     'sonar': {'success_auc': ..., 'precision': ...},
#     'fusion': {'success_auc': ..., 'precision': ...}
# }
```

---

#### Evaluation Pipeline

**Class: `TrackingEvaluator`**

Complete evaluation pipeline for tracking performance.

```python
from aquasot3d.evaluation import TrackingEvaluator

evaluator = TrackingEvaluator(
    model=tracker,
    data_loader=val_loader,
    device='cuda',
    save_results=True,
    output_dir='outputs/evaluation',
    visualize=True
)

# Run evaluation
results = evaluator.evaluate()
# {
#     'overall': {'success_auc': ..., 'precision': ...},
#     'per_sequence': {...},
#     'per_modality': {...}
# }

# Print report
evaluator.print_report(results)
```

**Constructor Parameters:**

- `model: nn.Module` - Tracker model
- `data_loader: DataLoader` - Evaluation data loader
- `device: str = 'cuda'` - Device
- `save_results: bool = True` - Save prediction results
- `output_dir: str = 'outputs/evaluation'` - Output directory
- `visualize: bool = False` - Generate visualizations

**Methods:**

- `evaluate() -> Dict` - Run complete evaluation
- `evaluate_sequence(seq_name) -> Dict` - Evaluate single sequence
- `print_report(results)` - Print formatted report
- `save_results(results, filename)` - Save results to JSON

---

**Function: `evaluate_model`**

Convenience function for quick evaluation.

```python
from aquasot3d.evaluation import evaluate_model

results = evaluate_model(
    model=tracker,
    config_path='configs/eval/eval_uuv3.yaml',
    checkpoint_path='outputs/checkpoints/best_model.pth',
    device='cuda'
)
```

---

#### Visualization

**Class: `TrackingVisualizer`**

Visualization tools for tracking results.

```python
from aquasot3d.evaluation import TrackingVisualizer

visualizer = TrackingVisualizer(
    output_dir='outputs/visualizations',
    show_confidence=True,
    show_3d=True,
    fps=10
)

# Visualize frame
visualizer.visualize_frame(
    rgb_image=rgb_img,
    sonar_image=sonar_img,
    predictions={
        'rgb_boxes': rgb_box,
        'sonar_boxes': sonar_box,
        'fusion_boxes': fusion_box,
        'weak_3d_center': center_3d
    },
    ground_truth={
        'rgb_boxes': rgb_gt,
        'sonar_boxes': sonar_gt
    },
    frame_idx=0,
    sequence_name='uuv3/light'
)

# Create video
visualizer.create_video(
    frames_dir='outputs/visualizations/uuv3_light',
    output_path='outputs/videos/uuv3_light.mp4'
)

# Plot metrics
visualizer.plot_metrics_curves(
    metrics_dict={'success_auc': 0.75, 'precision': 0.82},
    save_path='outputs/plots/metrics.png'
)
```

**Constructor Parameters:**

- `output_dir: str` - Output directory
- `show_confidence: bool = True` - Show confidence scores
- `show_3d: bool = False` - Show 3D center visualization
- `fps: int = 10` - Video frame rate

**Methods:**

- `visualize_frame(...)` - Visualize single frame
- `visualize_sequence(...)` - Visualize entire sequence
- `create_video(frames_dir, output_path)` - Create video from frames
- `plot_success_curve(ious, save_path)` - Plot success curve
- `plot_precision_curve(errors, save_path)` - Plot precision curve
- `plot_metrics_curves(metrics_dict, save_path)` - Plot all metrics

---

## Utility Modules

### Module: `aquasot3d.utils`

#### Configuration Management

See [Configuration System](#configuration-system) section.

#### Geometry Utilities

**Module: `aquasot3d.utils.geometry`**

**Class: `CoordinateTransform`**

Learnable coordinate transformation for Sonar→Fusion without calibration.

```python
from aquasot3d.utils.geometry import CoordinateTransform

transform = CoordinateTransform(
    init_identity=True,
    learnable=True
)

# Transform 3D points
sonar_3d = torch.randn(2, 3)  # (B, 3) in Sonar frame
fusion_3d = transform(sonar_3d)  # (B, 3) in Fusion frame
```

---

**Function: `project_3d_to_2d_rgb`**

Weak perspective projection for RGB modality.

```python
from aquasot3d.utils.geometry import project_3d_to_2d_rgb

center_3d = torch.tensor([[1.0, 0.5, 5.0]])  # (B, 3) [X, Y, Z]
bbox_2d = project_3d_to_2d_rgb(
    center_3d,
    focal_length=500.0,
    image_size=(256, 256)
)  # (B, 4) [x, y, w, h]
```

---

**Function: `project_3d_to_2d_sonar`**

Learned projection for Sonar modality.

```python
from aquasot3d.utils.geometry import project_3d_to_2d_sonar

center_3d = torch.tensor([[1.0, 0.5, 5.0]])
transform = CoordinateTransform()
bbox_2d = project_3d_to_2d_sonar(
    center_3d,
    transform=transform,
    max_range=10.0,
    fov_angle=90.0,
    image_size=(256, 256)
)
```

---

**Function: `bbox_to_center`**

Convert bounding box to center point.

```python
from aquasot3d.utils.geometry import bbox_to_center

bbox = torch.tensor([[10, 10, 50, 50]])  # [x, y, w, h]
center = bbox_to_center(bbox)  # [[35, 35]]
```

---

**Function: `center_to_bbox`**

Convert center point to bounding box.

```python
from aquasot3d.utils.geometry import center_to_bbox

center = torch.tensor([[35, 35]])
size = torch.tensor([[50, 50]])
bbox = center_to_bbox(center, size)  # [[10, 10, 50, 50]]
```

---

**Class: `CyclicConsistencyLoss`**

3D→2D→3D cyclic consistency loss.

```python
from aquasot3d.utils.geometry import CyclicConsistencyLoss

loss_fn = CyclicConsistencyLoss(
    loss_type='smooth_l1',
    weight_rgb=1.0,
    weight_sonar=1.0
)

# Compute loss
center_3d_pred = torch.randn(2, 3)
bbox_2d_rgb_gt = torch.randn(2, 4)
bbox_2d_sonar_gt = torch.randn(2, 4)

loss = loss_fn(
    center_3d_pred,
    bbox_2d_rgb_gt,
    bbox_2d_sonar_gt,
    project_fn_rgb=project_3d_to_2d_rgb,
    project_fn_sonar=project_3d_to_2d_sonar
)
```

---

## Configuration System

### Module: `aquasot3d.utils.config`

**Function: `load_config`**

Load YAML configuration with inheritance and overrides.

```python
from aquasot3d.utils.config import load_config

# Load with inheritance
config = load_config('configs/train/train_uuv2.yaml')

# Load with command-line overrides
config = load_config(
    'configs/train/train_uuv2.yaml',
    overrides=[
        'training.lr=0.001',
        'training.batch_size=64',
        'model.backbone.rgb.pretrained=false'
    ]
)
```

**Parameters:**

- `config_path: str` - Path to YAML config file
- `overrides: List[str] = None` - Command-line overrides (key.subkey=value format)

**Returns:**

- `config: ConfigDict` - Configuration dictionary with attribute access

**Configuration Inheritance:**

```yaml
# configs/base/default.yaml
training:
  lr: 0.0002
  batch_size: 32

# configs/train/train_uuv2.yaml
_base_: ../base/default.yaml
training:
  epochs: 60
  # lr and batch_size inherited from base
```

---

**Function: `save_config`**

Save configuration to YAML file.

```python
from aquasot3d.utils.config import save_config

save_config(config, 'outputs/config.yaml')
```

---

**Function: `validate_config`**

Validate configuration has required fields.

```python
from aquasot3d.utils.config import validate_config

required_fields = [
    'model.type',
    'training.lr',
    'training.epochs'
]

try:
    validate_config(config, required_fields)
except ValueError as e:
    print(f"Invalid config: {e}")
```

---

**Function: `print_config`**

Print configuration in readable format.

```python
from aquasot3d.utils.config import print_config

print_config(config)
```

---

## Quick Start Examples

### Example 1: Data Loading

```python
from aquasot3d.dataloader import RGBSonarDataset, build_dataloader
from aquasot3d.dataloader.transforms import (
    RGBTransforms, SonarTransforms, LogPolarTransform
)

# Create dataset
dataset = RGBSonarDataset(
    root_dir='data/uuv2',
    sequences=['light', 'tank'],
    mode='train',
    rgb_size=(256, 256),
    sonar_size=(256, 256)
)

# Create dataloader
train_loader = build_dataloader(
    dataset=dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)

# Iterate
for batch in train_loader:
    rgb_template = batch['rgb_template']  # (B, 3, 256, 256)
    sonar_template = batch['sonar_template']  # (B, 1, 256, 256)
    # ... training code
```

### Example 2: Model Building

```python
from aquasot3d.models import build_tracker

# Build from config
config = {
    'model': {
        'type': 'AquaSOT3DTracker',
        'backbone': {
            'rgb': {
                'type': 'vit',
                'patch_size': 16,
                'embed_dim': 768,
                'pretrained': True
            },
            'sonar': {
                'type': 'sonar_encoder',
                'output_dim': 384,
                'use_logpolar': True
            }
        },
        'fusion': {
            'fusion_dim': 512,
            'num_heads': 8,
            'use_geometric_gating': True
        },
        'temporal': {
            'type': 'vision_mamba',
            'num_layers': 4
        },
        'heads': {
            'use_weak3d': True,
            'use_cyclic_consistency': True
        }
    }
}

tracker = build_tracker(config['model'])
tracker = tracker.cuda()
```

### Example 3: Training

```python
from aquasot3d.training import (
    LossManager, build_optimizer, build_scheduler,
    CheckpointManager, Trainer
)

# Setup training components
loss_manager = LossManager(config['losses'])
optimizer = build_optimizer(tracker, 'adamw', base_lr=0.0002)
scheduler = build_scheduler(optimizer, 'cosine', max_epoch=60)
checkpoint_manager = CheckpointManager('outputs/checkpoints')

# Build trainer
trainer = Trainer(
    model=tracker,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_manager=loss_manager,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_manager=checkpoint_manager,
    max_epochs=60,
    use_amp=True
)

# Run training
trainer.train()
```

### Example 4: Evaluation

```python
from aquasot3d.evaluation import TrackingEvaluator

# Load checkpoint
checkpoint = torch.load('outputs/checkpoints/best_model.pth')
tracker.load_state_dict(checkpoint['model'])

# Create evaluator
evaluator = TrackingEvaluator(
    model=tracker,
    data_loader=val_loader,
    device='cuda',
    save_results=True,
    output_dir='outputs/evaluation',
    visualize=True
)

# Run evaluation
results = evaluator.evaluate()
evaluator.print_report(results)
```

### Example 5: Inference

```python
from aquasot3d.models import build_tracker
import torch

# Load model
tracker = build_tracker(config)
checkpoint = torch.load('checkpoints/best_model.pth')
tracker.load_state_dict(checkpoint['model'])
tracker = tracker.cuda().eval()

# Prepare input
rgb_template = load_image('frame_0001.jpg')  # (1, 3, 128, 128)
rgb_search = load_image('frame_0030.jpg')    # (1, 3, 256, 256)
sonar_template = load_sonar('sonar_0001.png')  # (1, 1, 256, 128)
sonar_search = load_sonar('sonar_0030.png')    # (1, 1, 512, 256)
rgb_bbox_template = torch.tensor([[64, 64, 32, 32]])
sonar_bbox_template = torch.tensor([[128, 64, 64, 32]])

# Inference
with torch.no_grad():
    outputs = tracker(
        rgb_template=rgb_template.cuda(),
        rgb_search=rgb_search.cuda(),
        sonar_template=sonar_template.cuda(),
        sonar_search=sonar_search.cuda(),
        rgb_bbox_template=rgb_bbox_template.cuda(),
        sonar_bbox_template=sonar_bbox_template.cuda()
    )

# Extract results
fusion_box = outputs['fusion_boxes'][0]  # (4,) [x, y, w, h]
weak_3d = outputs['weak_3d_center'][0]   # (3,) [X, Y, Z]
confidence = outputs['fusion_cls'][0]    # (1,) score
```

---

## API Index

### Core Infrastructure

- `BaseModule`, `BaseBackbone`, `BaseFusion`, `BaseHead`
- `Registry`, `BACKBONES`, `FUSION_MODULES`, `TEMPORAL_HEADS`, `PREDICTION_HEADS`
- `Config`, `ConfigDict`

### Data Pipeline

- `RGBSonarDataset`, `TemplateSearchSampler`
- `LogPolarTransform`, `RGBTransforms`, `SonarTransforms`
- `build_dataloader`

### Model Components

- `ViTRGBBackbone`, `ConvNeXtRGBBackbone`, `SonarEncoder`, `MultiModalBackbone`
- `G2DCMA`, `GeometricGating`, `DeformableAttention`, `ConsistencyLoss`
- `VisionMambaHead`, `TransformerTemporalHead`, `TemporalAggregator`
- `BoxHead`, `FusionHead`, `Weak3DHead`
- `AquaSOT3DTracker`, `build_tracker`

### Training Infrastructure

- `LossManager`, `build_loss_manager`
- `build_optimizer`, `get_parameter_groups`, `clip_gradients`
- `build_scheduler`, `CosineAnnealingWarmupRestarts`
- `CheckpointManager`, `save_checkpoint`, `load_checkpoint`
- `Trainer`, `build_trainer`

### Evaluation Tools

- `compute_iou`, `compute_center_error`, `compute_success_auc`, `compute_precision_at_threshold`
- `compute_sequence_metrics`, `compute_multi_modal_metrics`
- `TrackingEvaluator`, `evaluate_model`
- `TrackingVisualizer`

### Utility Modules

- `load_config`, `save_config`, `validate_config`, `print_config`
- `CoordinateTransform`, `project_3d_to_2d_rgb`, `project_3d_to_2d_sonar`
- `bbox_to_center`, `center_to_bbox`, `CyclicConsistencyLoss`

---

## References

1. **OSTrack**: Ye et al., "Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework", ECCV 2022
2. **Vision-Mamba**: Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", ICML 2024
3. **RGB-Sonar Tracking Benchmark**: Li et al., "RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker", arXiv:2406.07189, 2024
4. **Deformable DETR**: Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021

---

**Document End**

For implementation details, please refer to the source code in `src/aquasot3d/`.
For usage tutorials, see `README.md` and training guides in `docs/`.
For development planning, see `docs/Development_Planning.md`.
