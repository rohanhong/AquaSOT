# AquaSOT: Mathematical Formulation and Algorithmic Framework

**End-to-End Dual-Modal RGB-Sonar Single Object Tracking**

**Version**: 2.1 (Synchronized with current implementation)
**Last Updated**: 2026-01-30
**Based on**: `lib/models/aquasot/`, `lib/train/`, `lib/utils/`, `lib/test/evaluation/`

---

## Table of Contents

1. [Introduction and Problem Formulation](#1-introduction-and-problem-formulation)
2. [Model Architecture](#2-model-architecture)
3. [Loss Functions](#3-loss-functions)
4. [Data Processing Pipeline](#4-data-processing-pipeline)
5. [Template Update Strategy](#5-template-update-strategy)
6. [Training Configuration](#6-training-configuration)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [References](#8-references)

---

## 1. Introduction and Problem Formulation

### 1.1 Problem Statement

AquaSOT addresses **calibration-free dual-modal single object tracking** in underwater environments, where RGB camera and Forward-Looking Sonar (FLS) sensors are:
- **Temporally aligned** but **spatially unaligned** (no known extrinsic calibration)
- Subject to complementary failure modes (turbidity affects RGB, acoustic noise affects sonar)

The system learns cross-modal correspondence implicitly through end-to-end training.

**Formal Definition**:

$$
\mathcal{T}: (I_t^{rgb}, I_t^{sonar}, B_0^{rgb}, B_0^{sonar}) \rightarrow (B_t^{rgb}, B_t^{sonar})
$$

where $t \in \{1, 2, ..., T\}$ and $B = (c_x, c_y, w, h)$ represents normalized bounding box coordinates.

### 1.2 Input Specification

**Template Input** (first frame):

$$
\mathcal{T} = \{T^{rgb}, T^{sonar}, B_0^{rgb}, B_0^{sonar}\}
$$

where:
- $T^{rgb} \in \mathbb{R}^{3 \times 128 \times 128}$: RGB template image
- $T^{sonar} \in \mathbb{R}^{1 \times 128 \times 128}$: Sonar template image (grayscale)
- $B_0^{rgb}, B_0^{sonar}$: Initial bounding boxes in $(x, y, w, h)$ format

**Search Input** (current frame):

$$
\mathcal{S}_t = \{S_t^{rgb}, S_t^{sonar}\}
$$

where:
- $S_t^{rgb} \in \mathbb{R}^{3 \times 256 \times 256}$: RGB search region
- $S_t^{sonar} \in \mathbb{R}^{1 \times 256 \times 256}$: Sonar search region

### 1.3 Output Specification

**Predictions at frame $t$**:

$$
\mathcal{O}_t = \{\hat{B}_t^{rgb}, \hat{B}_t^{sonar}, s_t^{rgb}, s_t^{sonar}\}
$$

where:
- $\hat{B}_t^{rgb}, \hat{B}_t^{sonar} \in \mathbb{R}^4$: Predicted bounding boxes $(c_x, c_y, w, h)$ normalized to $[0, 1]$
- $s_t^{rgb}, s_t^{sonar} \in [0, 1]$: Confidence scores

### 1.4 Coordinate System and Bounding Box Formats

**Bounding Box Formats**:

| Format | Description | Usage |
|--------|-------------|-------|
| $(x, y, w, h)$ | Top-left corner + size | Dataset annotations |
| $(c_x, c_y, w, h)$ | Center + size | Model internal |
| $(x_1, y_1, x_2, y_2)$ | Top-left + bottom-right | IoU computation |

**Format Conversions**:

$$
\begin{aligned}
\text{xywh} \rightarrow \text{cxcywh}: & \quad (c_x, c_y) = (x + w/2, y + h/2) \\
\text{cxcywh} \rightarrow \text{xyxy}: & \quad (x_1, y_1, x_2, y_2) = (c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2)
\end{aligned}
$$

---

## 2. Model Architecture

### 2.1 Architecture Overview

AquaSOT follows a dual-stream architecture based on OSTrack:

```
RGB Template [B, 3, 128, 128]  ----+
                                   +--> ViT RGB Backbone --> RGB Features [B, L, 768]
RGB Search [B, 3, 256, 256]    ----+                              |
                                                                  v
                                                         DCMA Fusion (2 layers)
                                                                  |
Sonar Template [B, 1, 128, 128] ---+                              v
                                   +--> ViT Sonar Backbone --> Sonar Features [B, L, 768]
Sonar Search [B, 1, 256, 256]   ---+                              |
                                                                  v
                                           +----------------------+----------------------+
                                           |                                             |
                                           v                                             v
                                     rgb_box_head                                 sonar_box_head
                                           |                                             |
                                           v                                             v
                                   RGB Bbox [B, 4]                               Sonar Bbox [B, 4]
                                   RGB Score Map [B, 1, 16, 16]                  Sonar Score Map [B, 1, 16, 16]
```

**Note**: AquaSOT uses **separate prediction heads** (`rgb_box_head`, `sonar_box_head`) to prevent prediction drift caused by modality FOV asymmetry (RGB FOV ~60-70 deg, Sonar FOV ~120-160 deg).

**Source**: `lib/models/aquasot/aquasot.py:16-177`

### 2.2 Vision Transformer Backbone

The backbone is ViT-Base with MAE pretraining, adapted for tracking.

**Source**: `lib/models/aquasot/vit.py:89-165`

#### 2.2.1 Patch Embedding

Input images are divided into non-overlapping patches:

$$
\mathbf{x}_p = \text{PatchEmbed}(\mathbf{I}) \in \mathbb{R}^{N \times D}
$$

where:
- $N = (H/P) \times (W/P)$: Number of patches
- $P = 16$: Patch size
- $D = 768$: Embedding dimension

For template ($128 \times 128$): $N_z = 64$ patches
For search ($256 \times 256$): $N_s = 256$ patches

**Combined sequence length**: $L = N_z + N_s = 64 + 256 = 320$ patches

#### 2.2.2 Self-Attention

**Source**: `lib/models/aquasot/vit.py:30-59`

Multi-head self-attention with scaled dot-product:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

where:
- $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times N \times d_k}$
- $d_k = D / H = 768 / 12 = 64$: Head dimension
- $H = 12$: Number of attention heads

#### 2.2.3 Transformer Block

**Source**: `lib/models/aquasot/vit.py:62-86`

Pre-normalization architecture:

$$
\begin{aligned}
\mathbf{x}' &= \mathbf{x} + \text{DropPath}\big(\text{Attn}(\text{LayerNorm}(\mathbf{x}))\big) \\
\mathbf{x}'' &= \mathbf{x}' + \text{DropPath}\big(\text{MLP}(\text{LayerNorm}(\mathbf{x}'))\big)
\end{aligned}
$$

MLP structure:

$$
\text{MLP}(\mathbf{x}) = \text{Linear}_2\big(\text{GELU}(\text{Linear}_1(\mathbf{x}))\big)
$$

where $\text{Linear}_1: D \rightarrow 4D$ and $\text{Linear}_2: 4D \rightarrow D$.

#### 2.2.4 Channel Adaptation for Sonar

**Source**: `lib/models/aquasot/vit.py:167-193`

For 1-channel sonar input, pretrained RGB weights are adapted:

$$
\mathbf{W}_{sonar} = \frac{1}{3}\sum_{c=1}^{3} \mathbf{W}_{rgb}^{(c)}
$$

**Configuration** (from `lib/config/aquasot/config.py`):

| Parameter | RGB | Sonar |
|-----------|-----|-------|
| `IN_CHANNELS` | 3 | 1 |
| `EMBED_DIM` | 768 | 768 |
| `DEPTH` | 12 | 12 |
| `NUM_HEADS` | 12 | 12 |
| `MLP_RATIO` | 4.0 | 4.0 |
| `DROP_PATH_RATE` | 0.1 | 0.1 |

### 2.3 Cross-Modal Attention Fusion (DCMA)

**Source**: `lib/models/aquasot/fusion.py:19-156`

DCMA performs bidirectional cross-attention between RGB and Sonar features using standard scaled dot-product attention.

#### 2.3.1 Cross-Modal Attention

**Source**: `lib/models/aquasot/fusion.py:19-62`

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{W}_Q \cdot \mathbf{x}_{query} \\
\mathbf{K} &= \mathbf{W}_K \cdot \mathbf{x}_{kv} \\
\mathbf{V} &= \mathbf{W}_V \cdot \mathbf{x}_{kv} \\
\text{Attn} &= \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} \\
\text{out} &= \mathbf{W}_O \cdot \text{Attn}
\end{aligned}
$$

where $d_k = D / H = 768 / 8 = 96$.

#### 2.3.2 DCMA Layer (Bidirectional)

**Source**: `lib/models/aquasot/fusion.py:65-122`

Each layer performs bidirectional cross-attention with residual connections:

$$
\begin{aligned}
\mathbf{F}_{rgb}' &= \mathbf{F}_{rgb} + \text{CrossAttn}(\text{LN}(\mathbf{F}_{rgb}), \mathbf{F}_{sonar}) \\
\mathbf{F}_{sonar}' &= \mathbf{F}_{sonar} + \text{CrossAttn}(\text{LN}(\mathbf{F}_{sonar}), \mathbf{F}_{rgb}) \\
\mathbf{F}_{rgb}^{out} &= \mathbf{F}_{rgb}' + \text{FFN}(\text{LN}(\mathbf{F}_{rgb}')) \\
\mathbf{F}_{sonar}^{out} &= \mathbf{F}_{sonar}' + \text{FFN}(\text{LN}(\mathbf{F}_{sonar}'))
\end{aligned}
$$

FFN structure:

$$
\text{FFN}(\mathbf{x}) = \text{Dropout}\big(\text{Linear}_2(\text{Dropout}(\text{GELU}(\text{Linear}_1(\mathbf{x}))))\big)
$$

where $\text{Linear}_1: D \rightarrow 4D$ and $\text{Linear}_2: 4D \rightarrow D$.

**Configuration**:

| Parameter | Value | Source |
|-----------|-------|--------|
| `NUM_LAYERS` | 2 | `cfg.MODEL.FUSION.NUM_LAYERS` |
| `NUM_HEADS` | 8 | `cfg.MODEL.FUSION.NUM_HEADS` |
| `EMBED_DIM` | 768 | `cfg.MODEL.FUSION.EMBED_DIM` |
| `DROPOUT` | 0.1 | `cfg.MODEL.FUSION.DROPOUT` |
| `RGB_FUSION_WEIGHT` | 0.3 | `cfg.MODEL.FUSION.RGB_FUSION_WEIGHT` |
| `SONAR_FUSION_WEIGHT` | 0.3 | `cfg.MODEL.FUSION.SONAR_FUSION_WEIGHT` |

#### 2.3.3 Configurable Fusion Weights

**Source**: `lib/models/aquasot/fusion.py:101-125`

The cross-attention residual connections use configurable weights to control fusion strength:

$$
\begin{aligned}
\mathbf{F}_{rgb}' &= \mathbf{F}_{rgb} + \alpha_{rgb} \cdot \text{CrossAttn}(\text{LN}(\mathbf{F}_{rgb}), \mathbf{F}_{sonar}) \\
\mathbf{F}_{sonar}' &= \mathbf{F}_{sonar} + \alpha_{sonar} \cdot \text{CrossAttn}(\text{LN}(\mathbf{F}_{sonar}), \mathbf{F}_{rgb})
\end{aligned}
$$

where:
- $\alpha_{rgb} = 0.3$: Fusion weight for RGB receiving Sonar information
- $\alpha_{sonar} = 0.3$: Fusion weight for Sonar receiving RGB information
- $\alpha = 0$: No cross-modal influence (independent processing)
- $\alpha = 1$: Full fusion (original behavior)

The weak fusion ($\alpha = 0.3$) reduces cross-modal interference while still enabling complementary information exchange.

### 2.4 CenterPredictor Head

**Source**: `lib/models/layers/head.py:31-165`

The prediction head uses three parallel convolutional branches. AquaSOT uses **separate prediction heads** for each modality:
- `rgb_box_head`: Learns RGB-specific prediction patterns (faster motion, larger variance)
- `sonar_box_head`: Learns Sonar-specific prediction patterns (slower motion, stable variance)

This separation is critical because shared heads cause Sonar predictions to drift toward RGB motion patterns due to gradient averaging during training.

#### 2.4.1 Branch Architecture

Each branch follows Conv-BN-ReLU pattern with channel reduction:

$$
768 \xrightarrow{\text{conv}} 256 \xrightarrow{\text{conv}} 128 \xrightarrow{\text{conv}} 64 \xrightarrow{\text{conv}} 32 \xrightarrow{\text{conv}} C_{out}
$$

**Output Maps**:

$$
\begin{aligned}
\mathbf{M}_{score} &= \sigma(\text{Conv}_{score}(\mathbf{F}_{search})) \in \mathbb{R}^{B \times 1 \times 16 \times 16} \\
\mathbf{M}_{size} &= \sigma(\text{Conv}_{size}(\mathbf{F}_{search})) \in \mathbb{R}^{B \times 2 \times 16 \times 16} \\
\mathbf{M}_{offset} &= \text{Conv}_{offset}(\mathbf{F}_{search}) \in \mathbb{R}^{B \times 2 \times 16 \times 16}
\end{aligned}
$$

where $\sigma(\cdot)$ is the sigmoid function clamped to $[10^{-4}, 1-10^{-4}]$.

#### 2.4.2 Bounding Box Calculation

**Source**: `lib/models/layers/head.py:96-124`

1. Find peak location in score map:

$$
(i_x, i_y) = \text{argmax}_{x,y}(\mathbf{M}_{score})
$$

2. Extract predictions at peak:

$$
\begin{aligned}
(o_x, o_y) &= \mathbf{M}_{offset}[:, :, i_y, i_x] \\
(w, h) &= \mathbf{M}_{size}[:, :, i_y, i_x]
\end{aligned}
$$

3. Compute normalized bounding box:

$$
\hat{B} = \left(\frac{i_x + o_x}{f_s}, \frac{i_y + o_y}{f_s}, w, h\right)
$$

where $f_s = 16$ is the feature map size (search size / stride = 256 / 16).

---

## 3. Loss Functions

### 3.1 Focal Loss

**Source**: `lib/utils/focal_loss.py:10-59`

Focal loss addresses class imbalance between foreground (target center) and background in heatmaps.

**Definition**:

$$
\mathcal{L}_{focal} = \frac{1}{N_{pos}}\left(\sum_{pos} -(1-p)^\alpha \log(p) + \sum_{neg} -(1-y)^\beta p^\alpha \log(1-p)\right)
$$

where:
- $p$: Predicted probability (clamped to $[10^{-4}, 1-10^{-4}]$)
- $y$: Ground truth Gaussian heatmap value
- $pos$: Locations where $y \geq 0.99$
- $neg$: Locations where $y < 0.99$
- $N_{pos}$: Number of positive locations

**Hyperparameters**:
- $\alpha = 2$: Focusing parameter for positives (emphasizes hard positives)
- $\beta = 4$: Focusing parameter for negatives (suppresses easy negatives)

### 3.2 Generalized IoU Loss

**Source**: `lib/utils/box_ops.py:93-136`

GIoU loss provides scale-invariant bounding box regression.

**Definition**:

$$
\begin{aligned}
\text{IoU} &= \frac{|A \cap B|}{|A \cup B|} \\
\text{GIoU} &= \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|} \\
\mathcal{L}_{GIoU} &= 1 - \text{GIoU}
\end{aligned}
$$

where:
- $A$: Predicted box
- $B$: Ground truth box
- $C$: Smallest enclosing box containing both $A$ and $B$

**Properties**:
- $\text{GIoU} \in [-1, 1]$
- $\mathcal{L}_{GIoU} \in [0, 2]$

### 3.3 L1 Loss

**Source**: `lib/utils/box_ops.py:139-149`

Smooth bounding box regression:

$$
\mathcal{L}_{L1} = \frac{1}{4}\sum_{i=1}^{4}\left|\hat{B}^{(i)} - B_{gt}^{(i)}\right|
$$

where boxes are in $(x_1, y_1, x_2, y_2)$ format.

### 3.4 Total Loss

**Source**: `lib/train/actors/aquasot.py:166-171`

The total loss combines all components for both modalities:

$$
\mathcal{L}_{total} = \lambda_{giou}(\mathcal{L}_{giou}^{rgb} + \mathcal{L}_{giou}^{sonar}) + \lambda_{l1}(\mathcal{L}_{l1}^{rgb} + \mathcal{L}_{l1}^{sonar}) + \lambda_{focal}(\mathcal{L}_{focal}^{rgb} + \mathcal{L}_{focal}^{sonar})
$$

**Default Weights** (from `lib/config/aquasot/config.py`):

| Weight | Value | Source |
|--------|-------|--------|
| $\lambda_{giou}$ | 2.0 | `cfg.TRAIN.GIOU_WEIGHT` |
| $\lambda_{l1}$ | 5.0 | `cfg.TRAIN.L1_WEIGHT` |
| $\lambda_{focal}$ | 1.0 | `cfg.TRAIN.FOCAL_WEIGHT` |

---

## 4. Data Processing Pipeline

### 4.1 Template and Search Region Cropping

**Source**: `lib/train/data/processing.py:38-327`

Regions are cropped based on target bounding box with area expansion.

**Crop Size Calculation**:

$$
s_{crop} = \left\lceil\sqrt{w \cdot h} \cdot f_{area}\right\rceil
$$

where:
- $w, h$: Target width and height
- $f_{area}$: Area expansion factor (template: 2.0, search: 4.0)

**Configuration**:

| Parameter | Template | Search | Source |
|-----------|----------|--------|--------|
| Output Size | 128 | 256 | `cfg.DATA.*.SIZE` |
| Area Factor | 2.0 | 4.0 | `cfg.DATA.*.FACTOR` |
| Center Jitter | 0.0 | 1.0 | `cfg.DATA.*.CENTER_JITTER` |
| Scale Jitter | 0.0 | 0.25 | `cfg.DATA.*.SCALE_JITTER` |

### 4.2 Spatial Jitter Synchronization

**Source**: `lib/train/data/processing.py:89-131`

RGB and Sonar use **identical** spatial jitter parameters to maintain alignment:

**Scale Jitter**:

$$
(w', h') = (w, h) \cdot \exp(\sigma_{scale} \cdot \mathcal{N}(0, 1))
$$

**Center Jitter**:

$$
\begin{aligned}
d_{max} &= \sqrt{w' \cdot h'} \cdot f_{center} \\
(c_x', c_y') &= (c_x, c_y) + d_{max} \cdot (\mathcal{U}(-0.5, 0.5), \mathcal{U}(-0.5, 0.5))
\end{aligned}
$$

where:
- $(w, h)$: Original bounding box dimensions
- $(w', h')$: Jittered bounding box dimensions
- $\sigma_{scale} = 0.25$: Scale jitter standard deviation
- $f_{center} = 1.0$: Center jitter factor

### 4.3 Data Augmentation

**Source**: `lib/config/aquasot/config.py:201-236`

#### 4.3.1 Joint Augmentation (Synchronized)

| Augmentation | Probability | Source |
|--------------|-------------|--------|
| Horizontal Flip | 0.5 | `cfg.DATA.AUGMENTATION.HORIZONTAL_FLIP_PROB` |

#### 4.3.2 RGB-Specific Augmentation

| Augmentation | Parameters | Source |
|--------------|------------|--------|
| Brightness | $\pm 0.2$ | `cfg.DATA.AUGMENTATION.RGB.BRIGHTNESS` |
| Contrast | $\pm 0.3$ | `cfg.DATA.AUGMENTATION.RGB.CONTRAST` |
| Saturation | $\pm 0.3$ | `cfg.DATA.AUGMENTATION.RGB.SATURATION` |
| Hue | $\pm 0.2$ | `cfg.DATA.AUGMENTATION.RGB.HUE` |
| Grayscale | 5% prob | `cfg.DATA.AUGMENTATION.RGB.GRAYSCALE_PROB` |
| Gaussian Blur | 30% prob | `cfg.DATA.AUGMENTATION.RGB.BLUR_PROB` |
| Occlusion | 20% prob | `cfg.DATA.AUGMENTATION.RGB.OCCLUSION_PROB` |

#### 4.3.3 Sonar-Specific Augmentation

| Augmentation | Parameters | Source |
|--------------|------------|--------|
| Speckle Noise | level=0.1, 50% prob | `cfg.DATA.AUGMENTATION.SONAR.SPECKLE_*` |
| Occlusion | 20% prob | `cfg.DATA.AUGMENTATION.SONAR.OCCLUSION_PROB` |

### 4.4 Gaussian Heatmap Generation

**Source**: `lib/utils/heatmap_utils.py:30-76`

Ground truth heatmaps are generated as 2D Gaussians centered on the target.

**Sigma Calculation**:

$$
\begin{aligned}
\sigma_x &= \max\left(\frac{w_{norm} \cdot f_s}{6.0}, 1.0\right) \\
\sigma_y &= \max\left(\frac{h_{norm} \cdot f_s}{6.0}, 1.0\right)
\end{aligned}
$$

where:
- $w_{norm}, h_{norm} \in [0, 1]$: Normalized target width and height relative to search region
- $f_s = 16$: Feature map size (search_size / stride = 256 / 16)

**Gaussian Distribution**:

$$
G(x, y) = \exp\left(-\frac{(x - c_x)^2}{2\sigma_x^2} - \frac{(y - c_y)^2}{2\sigma_y^2}\right)
$$

where $(c_x, c_y)$ are target center coordinates in feature map space.

### 4.5 Normalization

**Source**: `lib/config/aquasot/config.py:141-144`

| Modality | Mean | Std | Source |
|----------|------|-----|--------|
| RGB | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | ImageNet |
| Sonar | [0.5] | [0.5] | `cfg.DATA.SONAR_*` |

### 4.6 Modality Dropout (Testing-Time Ablation)

**Source**: `lib/utils/modality_dropout.py:28-77`

For ablation experiments, modality dropout can be applied at test time to evaluate single-modal performance using a dual-modal trained model:

$$
I_{dropped} = \begin{cases}
\mathbf{0} & \text{if } p_{drop} = 1.0 \\
I & \text{otherwise}
\end{cases}
$$

When a modality is dropped:
1. Input tensor is replaced with zeros
2. Model produces low confidence score ($\approx 0$) for that modality
3. Target Loss Handler detects dropped modality as LOST
4. Primary output uses the active modality exclusively

**Configuration**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLED` | False | Enable modality dropout |
| `P_DROP_RGB` | 0.0 | Probability of dropping RGB (1.0 = Sonar-only) |
| `P_DROP_SONAR` | 0.0 | Probability of dropping Sonar (1.0 = RGB-only) |

**Note**: Training always uses both modalities to learn robust dual-modal representations. Dropout is testing-time only for ablation studies.

---

## 5. Template Update Strategy

### 5.1 MixFormer Three-Tier Strategy

**Source**: `lib/test/tracker/aquasot_tracker.py:163-370`

During inference, templates are updated using a three-tier strategy inspired by MixFormer.

#### 5.1.1 Tier Structure

| Tier | Description | Update Rule |
|------|-------------|-------------|
| Tier 1 | First-frame template | Never updated |
| Tier 2 | Online template queue | FIFO, size=2 |
| Tier 3 | Best-frame cache | Highest score in interval |

#### 5.1.2 Update Algorithm

**Source**: `lib/test/tracker/aquasot_tracker.py:296-370`

At each frame $t$:

1. **Score Decay** (optional):
   $$s_{max} \leftarrow s_{max} \cdot \gamma_{decay}$$

2. **Best-Frame Update**:
   $$
   \text{if } s_t > \tau_{score} \land s_t > s_{max}:
   \begin{cases}
   T_{best} \leftarrow \text{crop}(I_t, B_t) \\
   s_{max} \leftarrow s_t
   \end{cases}
   $$

3. **Queue Push** (every $T_{interval}$ frames):
   $$
   \text{if } t \mod T_{interval} = 0:
   \begin{cases}
   \text{push}(Q_{online}, T_{best}) & \text{FIFO} \\
   s_{max} \leftarrow -1 & \text{reset}
   \end{cases}
   $$

**Configuration** (from `lib/config/aquasot/config.py`):

| Parameter | Value | Source |
|-----------|-------|--------|
| $\tau_{score}$ | 0.5 | `cfg.TEST.TEMPLATE_UPDATE.SCORE_THRESHOLD` |
| $T_{interval}$ | 200 | `cfg.TEST.TEMPLATE_UPDATE.UPDATE_INTERVAL` |
| $|Q_{online}|$ | 2 | `cfg.TEST.TEMPLATE_UPDATE.ONLINE_SIZE` |
| $\gamma_{decay}$ | 1.0 | `cfg.TEST.TEMPLATE_UPDATE.MAX_SCORE_DECAY` |

### 5.2 Target Loss Detection and Recovery

**Source**: `lib/test/tracker/target_loss_handler.py`

AquaSOT implements dual-modal independent state machines for robust target tracking. Each modality (RGB and Sonar) maintains its own tracking state, enabling Sonar to continue tracking when the target leaves the RGB field of view.

#### 5.2.1 Tracking States

| State | Description | Output Behavior |
|-------|-------------|-----------------|
| STABLE | Target confidently tracked | Use prediction, allow template update |
| UNCERTAIN | Low confidence | Use prediction cautiously, no template update |
| LOST | Target lost | Hold last stable position, no template update |

#### 5.2.2 Score Classification

The confidence score $s$ is classified into tracking flags:

$$
f(s) = \begin{cases}
\texttt{NOT FOUND} & \text{if } s < \tau_{nf} \\
\texttt{UNCERTAIN} & \text{if } \tau_{nf} \leq s < \tau_{s} \\
\texttt{STABLE} & \text{if } s \geq \tau_{s}
\end{cases}
$$

where:
- $\tau_{nf} = 0.01$: Not-found threshold (`NOT_FOUND_THRESHOLD`)
- $\tau_{unc} = 0.05$: Uncertain threshold (`UNCERTAIN_THRESHOLD`)
- $\tau_{s} = 0.10$: Stable threshold (`STABLE_THRESHOLD`)

**Note**: The extremely low thresholds ($\tau_{nf}=0.01$, $\tau_{s}=0.10$) mean the tracker remains in STABLE state unless confidence drops near zero. This design prioritizes continuous tracking over premature loss detection.

#### 5.2.3 State Machine Transitions

**Lost Detection** (with hysteresis to prevent oscillation):

$$
\text{state} = \begin{cases}
\text{UNCERTAIN} & \text{if } c_{lost} \geq 1 \text{ (from STABLE)} \\
\text{LOST} & \text{if } c_{lost} \geq N_{lost}
\end{cases}
$$

**Recovery Detection**:

$$
\text{state} = \begin{cases}
\text{STABLE} & \text{if } c_{found} \geq N_{quick} \text{ (from UNCERTAIN)} \\
\text{STABLE} & \text{if } c_{found} \geq N_{found} \text{ (from LOST)}
\end{cases}
$$

where:
- $c_{lost}$: Consecutive frames with `target_lost = True`
- $c_{found}$: Consecutive frames with `target_lost = False`
- $N_{lost} = 2$: Frames to confirm LOST (`LOST_CONFIRM_FRAMES`)
- $N_{found} = 5$: Frames to confirm recovery from LOST (`FOUND_CONFIRM_FRAMES`)
- $N_{quick} = 2$: Frames for quick recovery from UNCERTAIN (`QUICK_RECOVERY_FRAMES`)

#### 5.2.4 Search Region Validation

Predictions are validated against a search region centered on the previous position:

$$
\mathit{valid} = \begin{cases}
\texttt{True} & \text{if } |c_x^{pred} - c_x^{prev}| \leq \frac{w \cdot K}{2} \land |c_y^{pred} - c_y^{prev}| \leq \frac{h \cdot K}{2} \\
\texttt{False} & \text{otherwise}
\end{cases}
$$

where $K = 5.0$ is the search region multiplier (`SEARCH_REGION_K`).

#### 5.2.5 Primary Output Selection

When both modalities are STABLE:

$$
B_{primary} = \begin{cases}
B_{rgb} & \text{if } \mathit{priority} = \texttt{rgb-first} \\
B_{sonar} & \text{if } \mathit{priority} = \texttt{sonar-first} \\
\arg\max_{m \in \{rgb, sonar\}}(s_m) & \text{if } \mathit{priority} = \texttt{higher-score}
\end{cases}
$$

When one modality is LOST, the STABLE modality is used exclusively. When both are LOST, the last stable RGB position is used as fallback.

#### 5.2.6 Selective Template Update

Templates are only updated when:
1. Modality state is STABLE
2. Confidence score exceeds `HIGH_CONFIDENCE_THRESHOLD` (0.7)

This prevents template pollution from unreliable predictions during occlusion or target loss.

#### 5.2.7 Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| `ENABLED` | True | `cfg.TEST.TARGET_LOSS.ENABLED` |
| $\tau_{nf}$ | 0.01 | `cfg.TEST.TARGET_LOSS.NOT_FOUND_THRESHOLD` |
| $\tau_{unc}$ | 0.05 | `cfg.TEST.TARGET_LOSS.UNCERTAIN_THRESHOLD` |
| $\tau_{s}$ | 0.10 | `cfg.TEST.TARGET_LOSS.STABLE_THRESHOLD` |
| $N_{lost}$ | 2 | `cfg.TEST.TARGET_LOSS.LOST_CONFIRM_FRAMES` |
| $N_{found}$ | 5 | `cfg.TEST.TARGET_LOSS.FOUND_CONFIRM_FRAMES` |
| $N_{quick}$ | 2 | `cfg.TEST.TARGET_LOSS.QUICK_RECOVERY_FRAMES` |
| $K$ | 5.0 | `cfg.TEST.TARGET_LOSS.SEARCH_REGION_K` |
| priority | higher_score | `cfg.TEST.TARGET_LOSS.PRIMARY_PRIORITY` |

#### 5.2.8 Dynamic Search Factor

**Source**: `lib/config/aquasot/config.py:291-312`

The search region factor is dynamically adjusted based on tracking state to improve target recovery:

$$
f_{search}(state) = \begin{cases}
f_{stable} = 4.0 & \text{if } state = \text{STABLE} \\
f_{uncertain} = 5.0 & \text{if } state = \text{UNCERTAIN} \\
f_{lost} = 7.0 & \text{if } state = \text{LOST}
\end{cases}
$$

**Configuration**:

| Parameter | Value | Source |
|-----------|-------|--------|
| `ENABLED` | True | `cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.ENABLED` |
| `STABLE` | 4.0 | `cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.STABLE` |
| `UNCERTAIN` | 5.0 | `cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.UNCERTAIN` |
| `LOST` | 7.0 | `cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.LOST` |

RGB and Sonar modalities can have independent search factors via `RGB_STABLE`, `RGB_UNCERTAIN`, `RGB_LOST`, `SONAR_STABLE`, `SONAR_UNCERTAIN`, `SONAR_LOST` parameters.

---

## 6. Training Configuration

### 6.1 Optimizer

**Source**: `lib/config/aquasot/config.py:78-111`

**AdamW** with decoupled weight decay:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L} \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L})^2 \\
\theta_t &= \theta_{t-1} - \eta_t \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon} - \lambda_{wd} \eta_t \theta_{t-1}
\end{aligned}
$$

**Hyperparameters**:

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning Rate | $10^{-4}$ | `cfg.TRAIN.LR` |
| Weight Decay | $10^{-4}$ | `cfg.TRAIN.WEIGHT_DECAY` |
| $\beta_1, \beta_2$ | 0.9, 0.999 | Default AdamW |
| Backbone LR Multiplier | 0.1 | `cfg.TRAIN.BACKBONE_MULTIPLIER` |

### 6.2 Learning Rate Schedule

**Step Decay**:

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T_{drop} \rfloor}
$$

| Parameter | Value | Source |
|-----------|-------|--------|
| $\gamma$ | 0.1 | `cfg.TRAIN.SCHEDULER.GAMMA` |
| $T_{drop}$ | 80 | `cfg.TRAIN.LR_DROP_EPOCH` |
| Warmup Epochs | 5 | `cfg.TRAIN.SCHEDULER.WARMUP_EPOCHS` |
| Warmup LR | $10^{-5}$ | `cfg.TRAIN.SCHEDULER.WARMUP_LR` |

### 6.3 Gradient Clipping

$$
\text{if } \|\nabla \mathcal{L}\|_2 > \tau_{clip}: \quad \nabla \mathcal{L} \leftarrow \frac{\tau_{clip}}{\|\nabla \mathcal{L}\|_2} \nabla \mathcal{L}
$$

| Parameter | Value | Source |
|-----------|-------|--------|
| $\tau_{clip}$ | 0.1 | `cfg.TRAIN.GRAD_CLIP_NORM` |

### 6.4 Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Epochs | 100 | `cfg.TRAIN.EPOCH` |
| Batch Size | 2 | `cfg.TRAIN.BATCH_SIZE` |
| Gradient Accumulation | 4 | `cfg.TRAIN.GRADIENT_ACCUMULATION` |
| Effective Batch Size | 8 | Batch Size $\times$ Accumulation |

---

## 7. Evaluation Metrics

### 7.0 Evaluation Settings

**Source**: `lib/config/aquasot/config.py:314-318`, `lib/test/evaluation/experiment.py:23-57`

AquaSOT follows GOT-10k style evaluation with visibility filtering and multi-modal support.

#### 7.0.1 Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FILTER_VISIBILITY` | True | Filter out absent frames (GT=0,0,0,0) |
| `SKIP_INIT_FRAME` | True | Skip frame 0 (initialization frame) |
| `EVAL_MODE` | 'dual' | Evaluation mode: 'dual', 'rgb_only', 'sonar_only' |

#### 7.0.2 Visibility Filtering

Following GOT-10k evaluation pattern, frames where the ground truth box is $(0, 0, 0, 0)$ (target absent/occluded) are excluded from metric computation:

$$
\mathcal{F}_{eval} = \{t \mid B_t^{gt} \neq (0, 0, 0, 0) \land t > 0\}
$$

This ensures metrics reflect actual tracking performance rather than absence detection capability.

**Visibility Mask Computation** (`lib/test/evaluation/experiment.py:249-273`):

$$
\text{visible}(t) = \neg(\text{all}(B_t^{gt} = 0)) \land (w_t > 0) \land (h_t > 0)
$$

#### 7.0.3 Multi-Modal Evaluation

Metrics are computed separately for RGB and Sonar modalities:
- **RGB metrics**: IoU, precision, center error for RGB predictions vs RGB ground truth
- **Sonar metrics**: IoU, precision, center error for Sonar predictions vs Sonar ground truth
- **Overall metrics**: Weighted average by number of visible frames per modality

### 7.1 Success Rate (AUC)

**Definition**: Fraction of frames where IoU exceeds threshold $\theta$:

$$
S(\theta) = \frac{1}{N}\sum_{t=1}^{N}\mathbb{1}[\text{IoU}_t > \theta]
$$

**Area Under Curve**:

$$
\text{AUC} = \int_0^1 S(\theta) \, d\theta
$$

### 7.2 Precision

**Center Error**:

$$
e_t = \sqrt{(c_x^{pred} - c_x^{gt})^2 + (c_y^{pred} - c_y^{gt})^2}
$$

**Precision at Threshold** (default 20 pixels):

$$
P_{20} = \frac{1}{N}\sum_{t=1}^{N}\mathbb{1}[e_t < 20]
$$

### 7.3 Average Center Error

$$
\text{ACE} = \frac{1}{N}\sum_{t=1}^{N} e_t
$$

---

## 8. References

### Primary References

1. **OSTrack** (Base Framework)
   - Ye, B., Chang, H., Ma, B., & Shan, S. (2022). *Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework*. ECCV 2022.
   - [arXiv:2203.11991](https://arxiv.org/abs/2203.11991) | [Code](https://github.com/botaoye/OSTrack)

2. **MixFormer** (Template Update Strategy)
   - Cui, Y., Jiang, C., Wang, L., & Wu, G. (2022). *MixFormer: End-to-End Tracking with Iterative Mixed Attention*. CVPR 2022.
   - [arXiv:2203.11082](https://arxiv.org/abs/2203.11082) | [Code](https://github.com/MCG-NJU/MixFormer)

3. **Vision Transformer**
   - Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
   - [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

### Loss Functions

4. **Focal Loss**
   - Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.
   - [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

5. **Generalized IoU**
   - Rezatofighi, H., et al. (2019). *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*. CVPR 2019.
   - [arXiv:1902.09630](https://arxiv.org/abs/1902.09630)

### Optimization

6. **AdamW**
   - Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019.
   - [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

### Model Architecture

7. **CenterNet** (Center-based Detection)
   - Zhou, X., Wang, D., & Krahenbuhl, P. (2019). *Objects as Points*.
   - [arXiv:1904.07850](https://arxiv.org/abs/1904.07850)

### Dataset

8. **RGBS50** (RGB-Sonar Tracking Benchmark)
   - Li, Y., et al. (2024). *RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker*.
   - [arXiv:2406.07189](https://arxiv.org/abs/2406.07189) | [Code](https://github.com/LiYunfengLYF/RGBS50)

---

## Appendix: Code Reference Index

| Section | Implementation File | Lines |
|---------|---------------------|-------|
| Model Architecture | `lib/models/aquasot/aquasot.py` | 16-177 |
| ViT Backbone | `lib/models/aquasot/vit.py` | 89-165 |
| Attention Module | `lib/models/aquasot/vit.py` | 30-59 |
| Transformer Block | `lib/models/aquasot/vit.py` | 62-86 |
| DCMA Fusion | `lib/models/aquasot/fusion.py` | 128-167 |
| Cross-Modal Attention | `lib/models/aquasot/fusion.py` | 19-62 |
| DCMA Layer | `lib/models/aquasot/fusion.py` | 65-125 |
| Fusion Weights | `lib/models/aquasot/fusion.py` | 101-119 |
| CenterPredictor | `lib/models/layers/head.py` | 31-165 |
| Focal Loss | `lib/utils/focal_loss.py` | 10-59 |
| GIoU Loss | `lib/utils/box_ops.py` | 93-136 |
| Heatmap Generation | `lib/utils/heatmap_utils.py` | 30-76 |
| Loss Computation | `lib/train/actors/aquasot.py` | 116-227 |
| Template Update | `lib/test/tracker/aquasot_tracker.py` | 296-370 |
| Target Loss Handler | `lib/test/tracker/target_loss_handler.py` | 1-403 |
| Modality Dropout | `lib/utils/modality_dropout.py` | 28-77 |
| Evaluation Experiment | `lib/test/evaluation/experiment.py` | 1-766 |
| Visibility Filtering | `lib/test/evaluation/experiment.py` | 249-273 |
| Checkpoint Migration | `lib/train/trainers/base_trainer.py` | 125-148 |
| Data Processing | `lib/train/data/processing.py` | 38-378 |
| Configuration | `lib/config/aquasot/config.py` | 1-443 |
