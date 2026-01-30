# AquaSOT: End-to-End Dual-Modal RGB-Sonar Single Object Tracking

> A calibration-free dual-modal tracking framework that leverages complementary RGB and Sonar modalities for robust underwater object tracking.

---

## Table of Contents

1. [Introduction and Problem Definition](#1-introduction-and-problem-definition)
2. [System Architecture](#2-system-architecture)
3. [Model Components](#3-model-components)
4. [Training Pipeline](#4-training-pipeline)
5. [Inference and Template Update](#5-inference-and-template-update)
6. [Dataset](#6-dataset)
7. [Configuration](#7-configuration)
8. [Ablation Experiments](#8-ablation-experiments)
9. [References](#9-references)
10. [Appendix: Code Reference Index](#10-appendix-code-reference-index)

---

## 1. Introduction and Problem Definition

### 1.1 Problem Statement

AquaSOT addresses **calibration-free dual-modal single object tracking** in underwater environments. The system processes paired RGB camera and Forward-Looking Sonar (FLS) inputs that are:

- **Temporally aligned**: RGB and sonar frames are synchronized in time
- **Spatially unaligned**: No known extrinsic calibration between sensors

The system learns cross-modal correspondence implicitly through end-to-end training, eliminating the need for explicit sensor calibration.

### 1.2 Task Formulation

Given an initial frame with target annotations for both modalities, the tracker predicts bounding boxes in subsequent frames.

**Formal Definition**:

```
T: (I_t^rgb, I_t^sonar, B_0^rgb, B_0^sonar) -> (B_t^rgb, B_t^sonar)
```

where:
- `I_t^rgb`: RGB image at frame t
- `I_t^sonar`: Sonar image at frame t
- `B_0^rgb, B_0^sonar`: Initial bounding boxes from first frame
- `B_t^rgb, B_t^sonar`: Predicted bounding boxes at frame t

### 1.3 Input Specification

**Template Input** (first frame):
- RGB Template: `[B, 3, 128, 128]` - 3-channel color image
- Sonar Template: `[B, 1, 128, 128]` - 1-channel grayscale image
- RGB/Sonar Bounding Boxes: `(x, y, w, h)` format

**Search Input** (current frame):
- RGB Search: `[B, 3, 256, 256]` - Search region centered on previous prediction
- Sonar Search: `[B, 1, 256, 256]` - Search region centered on previous prediction

### 1.4 Output Specification

**Predictions at frame t**:
- RGB Bounding Box: `[B, 4]` in `(cx, cy, w, h)` normalized to `[0, 1]`
- Sonar Bounding Box: `[B, 4]` in `(cx, cy, w, h)` normalized to `[0, 1]`
- RGB Confidence Score: scalar in `[0, 1]`
- Sonar Confidence Score: scalar in `[0, 1]`

### 1.5 Coordinate Systems and Bounding Box Formats

| Format | Description | Usage |
|--------|-------------|-------|
| `(x, y, w, h)` | Top-left corner + size | Dataset annotations, evaluation output |
| `(cx, cy, w, h)` | Center + size | Model internal representation |
| `(x1, y1, x2, y2)` | Top-left + bottom-right | IoU computation, loss calculation |

**Format Conversions**:
- `xywh -> cxcywh`: `(cx, cy) = (x + w/2, y + h/2)`
- `cxcywh -> xyxy`: `(x1, y1, x2, y2) = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)`

---

## 2. System Architecture

### 2.1 Architecture Overview

AquaSOT follows a dual-stream architecture based on OSTrack [1]:

```
                              AquaSOT Architecture
    ============================================================================

    RGB Input:
      Template [B, 3, 128, 128] --+
                                  +--> ViT-Base RGB ---> RGB Features [B, 320, 768]
      Search   [B, 3, 256, 256] --+                              |
                                                                 |
                                                                 v
                                                          +-----------+
                                                          |   DCMA    |
                                                          |  Fusion   |
                                                          | (2 layers)|
                                                          +-----------+
                                                                 |
    Sonar Input:                                                 |
      Template [B, 1, 128, 128] --+                              |
                                  +--> ViT-Base Sonar -> Sonar Features [B, 320, 768]
      Search   [B, 1, 256, 256] --+

                                                                 |
                                                    +------------+------------+
                                                    |                         |
                                                    v                         v
                                            +--------------+          +--------------+
                                            | rgb_box_head |          |sonar_box_head|
                                            +--------------+          +--------------+
                                                    |                         |
                                                    v                         v
                                            RGB Bbox [B, 4]           Sonar Bbox [B, 4]
                                            RGB Score [B, 1, 16, 16]  Sonar Score [B, 1, 16, 16]

    Note: Separate prediction heads prevent prediction drift caused by asymmetric FOV
          (RGB FOV ~60-70 deg, Sonar FOV ~120-160 deg)

    ============================================================================
```

**Source**: `lib/models/aquasot/aquasot.py:16-177`

### 2.2 Design Principles

1. **Dual-Stream Processing**: Separate feature extraction preserves modality-specific information
2. **Cross-Modal Fusion**: DCMA enables information exchange between modalities
3. **Independent Predictions**: Per-modality bounding boxes allow separate evaluation
4. **Calibration-Free**: Cross-attention learns implicit correspondence without explicit calibration

### 2.3 Key Components

| Component | Implementation | Description |
|-----------|----------------|-------------|
| RGB Backbone | ViT-Base (3-channel) | 12 transformer blocks, 768-dim embeddings |
| Sonar Backbone | ViT-Base (1-channel) | Adapted from RGB with channel averaging |
| Fusion Module | DCMA (2 layers) | Bidirectional cross-modal attention |
| Prediction Head | CenterPredictor | Score map + size map + offset map |

---

## 3. Model Components

### 3.1 Vision Transformer Backbone

**Source**: `lib/models/aquasot/vit.py:89-165`

The backbone uses ViT-Base with MAE [4] pretraining, processing concatenated template-search sequences.

#### 3.1.1 Patch Embedding

Images are divided into 16x16 non-overlapping patches:

```
Patch Embedding: I -> x_p in R^(N x D)

where:
  N = (H/P) x (W/P)  # Number of patches
  P = 16             # Patch size
  D = 768            # Embedding dimension

Template (128x128): N_z = 64 patches (8x8 grid)
Search (256x256):   N_s = 256 patches (16x16 grid)
Combined sequence:  L = 64 + 256 = 320 tokens
```

#### 3.1.2 Transformer Block Architecture

Pre-normalization architecture with stochastic depth:

```
x' = x + DropPath(Attention(LayerNorm(x)))
x'' = x' + DropPath(MLP(LayerNorm(x')))
```

**Configuration**:
- Embedding dimension: 768
- Number of heads: 12
- MLP ratio: 4.0
- Drop path rate: 0.1
- Number of blocks: 12

#### 3.1.3 Channel Adaptation for Sonar

**Source**: `lib/models/aquasot/vit.py:167-193`

For 1-channel sonar input, pretrained 3-channel RGB weights are adapted:

```python
W_sonar = mean(W_rgb, dim=1, keepdim=True)  # Average across RGB channels
```

This enables transfer learning from ImageNet-pretrained weights.

### 3.2 Cross-Modal Attention Fusion (DCMA)

**Source**: `lib/models/aquasot/fusion.py:19-188`

DCMA performs bidirectional cross-attention between RGB and Sonar features using standard scaled dot-product attention.

**Important Note**: Despite the name "DCMA" (Deformable Cross-Modal Attention), the implementation uses **standard cross-attention**, not sparse deformable sampling.

#### 3.2.1 Cross-Attention Mechanism

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

RGB -> Sonar attention:
  Q = W_q * LayerNorm(F_rgb)    # Query from RGB
  K = W_k * F_sonar              # Key from Sonar
  V = W_v * F_sonar              # Value from Sonar
  F_rgb_attn = Attention(Q, K, V)

Sonar -> RGB attention:
  Q = W_q * LayerNorm(F_sonar)  # Query from Sonar
  K = W_k * F_rgb                # Key from RGB
  V = W_v * F_rgb                # Value from RGB
  F_sonar_attn = Attention(Q, K, V)
```

**Semantic Clarification**: The naming convention follows the "query_to_context" pattern:
- `rgb_to_sonar`: RGB features serve as **query**, Sonar features provide **context** -> RGB is enhanced with Sonar information
- `sonar_to_rgb`: Sonar features serve as **query**, RGB features provide **context** -> Sonar is enhanced with RGB information

This bidirectional exchange enables each modality to leverage complementary information from the other.

#### 3.2.2 DCMA Layer Structure

Each DCMA layer contains:
1. Bidirectional cross-attention (RGB->Sonar, Sonar->RGB)
2. Feed-forward networks (FFN) per modality
3. Layer normalization and residual connections

```
F_rgb' = F_rgb + CrossAttn(LN(F_rgb), F_sonar)
F_sonar' = F_sonar + CrossAttn(LN(F_sonar), F_rgb)
F_rgb_out = F_rgb' + FFN(LN(F_rgb'))
F_sonar_out = F_sonar' + FFN(LN(F_sonar'))
```

**Configuration**:
- Number of layers: 2
- Number of heads: 8
- Head dimension: 96 (768 / 8)
- Dropout: 0.1
- RGB Fusion Weight: 0.3
- Sonar Fusion Weight: 0.3

#### 3.2.3 Configurable Fusion Weights

**Source**: `lib/models/aquasot/fusion.py:101-125`

Fusion weights control the strength of cross-modal attention:

```
F_rgb' = F_rgb + alpha_rgb * CrossAttn(LN(F_rgb), F_sonar)
F_sonar' = F_sonar + alpha_sonar * CrossAttn(LN(F_sonar), F_rgb)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RGB_FUSION_WEIGHT` | 0.3 | RGB cross-modal attention strength |
| `SONAR_FUSION_WEIGHT` | 0.3 | Sonar cross-modal attention strength |

- Set to 0 for no cross-modal fusion (identity)
- Set to 1 for full fusion (original behavior)
- Default 0.3 provides weak fusion, reducing cross-modal interference while enabling complementary information exchange

#### 3.2.4 Modality Validity Detection

**Source**: `lib/models/aquasot/aquasot.py:73-74`, `lib/models/aquasot/fusion.py:125-145`

When a modality is dropped (input is all zeros), cross-attention becomes invalid. The model detects modality validity and uses self-attention as fallback:

```python
rgb_valid = rgb_search.abs().sum() > 1e-4
sonar_valid = sonar_search.abs().sum() > 1e-4
```

**Fallback Behavior**:
- Sonar dropped: RGB queries attend to RGB (self-attention via `rgb_to_sonar` module)
- RGB dropped: Sonar queries use `rgb_to_sonar` module (K/V projections trained for Sonar)

**Note**: For Sonar self-attention when RGB is dropped, the `rgb_to_sonar` module is used because its K/V projections are trained for Sonar features. Using `sonar_to_rgb` would apply K/V projections trained for RGB features to Sonar features, producing incorrect output distributions.

This ensures consistent feature distributions for the downstream prediction heads during single-modality inference.

### 3.3 CenterPredictor Head

**Source**: `lib/models/layers/head.py:31-165`

The prediction head uses three parallel convolutional branches. **Important**: AquaSOT uses **separate prediction heads** for each modality:
- `rgb_box_head`: Trained on RGB features
- `sonar_box_head`: Trained on Sonar features

This separation addresses the prediction drift problem caused by asymmetric FOV between modalities (Sonar FOV > RGB FOV). Shared heads cause Sonar predictions to drift toward RGB motion patterns due to gradient averaging during training.

#### 3.3.1 Branch Architecture

Each branch follows Conv-BN-ReLU pattern with channel reduction:

```
768 -> Conv -> 256 -> Conv -> 128 -> Conv -> 64 -> Conv -> 32 -> Conv -> C_out
```

**Output Maps**:
- Score Map: `[B, 1, 16, 16]` - Target center probability (sigmoid)
- Size Map: `[B, 2, 16, 16]` - Width and height prediction (sigmoid)
- Offset Map: `[B, 2, 16, 16]` - Sub-pixel refinement (linear)

#### 3.3.2 Bounding Box Decoding

1. Find peak location in score map: `(i_x, i_y) = argmax(score_map)`
2. Extract size and offset at peak: `(w, h) = size_map[:, :, i_y, i_x]`, `(o_x, o_y) = offset_map[:, :, i_y, i_x]`
3. Compute normalized bounding box: `B = ((i_x + o_x)/16, (i_y + o_y)/16, w, h)`

---

## 4. Training Pipeline

### 4.1 Data Augmentation

**Source**: `lib/train/data/processing.py:38-378`

AquaSOT uses **synchronized spatial augmentation** across modalities to maintain RGB-Sonar alignment.

#### 4.1.1 Spatial Jitter (Synchronized)

Pre-rolled random parameters ensure identical transforms for both modalities:

```python
# Scale jitter: exp(N(0, sigma_scale))
jittered_size = box[2:4] * exp(randn(2) * scale_jitter_factor)

# Center jitter
max_offset = sqrt(jittered_size.prod()) * center_jitter_factor
jittered_center = center + max_offset * (rand(2) - 0.5)
```

**Parameters**:
- Template: center_jitter=0.0, scale_jitter=0.0 (no jitter)
- Search: center_jitter=1.0, scale_jitter=0.25

#### 4.1.2 Joint Augmentation

| Augmentation | Probability | Synchronization |
|--------------|-------------|-----------------|
| Horizontal Flip | 0.5 | Same decision for RGB and Sonar |

#### 4.1.3 Modality-Specific Augmentation

**RGB Augmentation**:
| Transform | Parameters |
|-----------|------------|
| Brightness | +/- 0.2 |
| Color Jitter | contrast=0.3, saturation=0.3, hue=0.2 |
| Grayscale | 5% probability |
| Gaussian Blur | 30% probability, kernel 3-7 |
| Random Occlusion | 20% probability |

**Sonar Augmentation**:
| Transform | Parameters |
|-----------|------------|
| Speckle Noise | level=0.1, 50% probability (multiplicative) |
| Random Occlusion | 20% probability |

#### 4.1.4 Normalization

| Modality | Mean | Std |
|----------|------|-----|
| RGB | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |
| Sonar | [0.5] | [0.5] |

### 4.2 Loss Functions

**Source**: `lib/train/actors/aquasot.py:116-227`

The total loss combines three components for each modality:

```
L_total = lambda_giou * (L_giou_rgb + L_giou_sonar) +
          lambda_l1   * (L_l1_rgb + L_l1_sonar) +
          lambda_focal* (L_focal_rgb + L_focal_sonar)
```

**Default Weights**:
- `lambda_giou = 2.0`
- `lambda_l1 = 5.0`
- `lambda_focal = 1.0`

#### 4.2.1 GIoU Loss

**Source**: `lib/utils/box_ops.py:93-136`

Generalized Intersection over Union provides scale-invariant box regression:

```
IoU = |A intersect B| / |A union B|
GIoU = IoU - |C \ (A union B)| / |C|
L_giou = 1 - GIoU
```

where C is the smallest enclosing box. Range: `L_giou in [0, 2]`.

#### 4.2.2 L1 Loss

**Source**: `lib/utils/box_ops.py:139-149`

Direct coordinate regression in `(x1, y1, x2, y2)` format:

```
L_l1 = mean(|B_pred - B_gt|)
```

The mean is computed over all 4 coordinates per sample, then averaged across the batch.

#### 4.2.3 Focal Loss

**Source**: `lib/utils/focal_loss.py:10-59`

Addresses class imbalance in heatmap supervision:

```
L_focal = (1/N_pos) * (sum_pos(-(1-p)^alpha * log(p)) +
                       sum_neg(-(1-y)^beta * p^alpha * log(1-p)))
```

**Parameters**:
- `alpha = 2` (focusing parameter for positives)
- `beta = 4` (focusing parameter for negatives)
- Positive threshold: `y >= 0.99`

#### 4.2.4 Ground Truth Heatmap Generation

**Source**: `lib/utils/heatmap_utils.py:30-76`

Gaussian heatmaps with size-adaptive sigma:

```
sigma_x = max(w_norm * feat_sz / 6.0, 1.0)
sigma_y = max(h_norm * feat_sz / 6.0, 1.0)
G(x, y) = exp(-((x - cx)^2 / (2*sigma_x^2) + (y - cy)^2 / (2*sigma_y^2)))
```

### 4.3 Optimizer and Scheduler

**Source**: `lib/config/aquasot/config.py:78-121`

**Optimizer**: AdamW with decoupled weight decay

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Backbone LR Multiplier | 0.1 |
| Gradient Clip Norm | 0.1 |

**Scheduler**: Configurable (step/cosine)

*Default* (config.py): Step decay with warmup
| Parameter | Value |
|-----------|-------|
| Warmup Epochs | 5 |
| Warmup LR | 1e-5 |
| LR Drop Epoch | 80 |
| Gamma | 0.1 |

*vitb_256_dual.yaml*: Cosine annealing with warmup
| Parameter | Value |
|-----------|-------|
| Warmup Epochs | 5 |
| Warmup LR | 1e-5 |
| Min LR | 1e-6 |

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 2 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 8 |

### 4.4 Training Commands

```bash
# Basic training
python tracking/train.py --script aquasot --config vitb_256_dual

# With Weights & Biases logging
python tracking/train.py --script aquasot --config vitb_256_dual \
    --wandb --wandb_project aquasot

# Distributed training
python -m torch.distributed.launch --nproc_per_node=4 \
    tracking/train.py --script aquasot --config vitb_256_dual
```

---

## 5. Inference and Template Update

### 5.1 MixFormer Three-Tier Template Strategy

**Source**: `lib/test/tracker/aquasot_tracker.py:163-370`

During inference, templates are managed using a three-tier strategy inspired by MixFormer [2]:

#### 5.1.1 Tier Structure

| Tier | Description | Update Rule |
|------|-------------|-------------|
| Tier 1 | First-frame template | Never updated (fixed reference) |
| Tier 2 | Online template queue | FIFO queue, size=2 |
| Tier 3 | Best-frame cache | Highest score in current interval |

#### 5.1.2 Update Algorithm

At each frame t:

1. **Score Decay** (optional):
   ```
   s_max = s_max * gamma_decay
   ```

2. **Best-Frame Update**:
   ```
   if s_t > tau_score AND s_t > s_max:
       T_best = crop(I_t, B_t)
       s_max = s_t
   ```

3. **Queue Push** (every T_interval frames):
   ```
   if t % T_interval == 0:
       push(Q_online, T_best)  # FIFO
       s_max = -1  # Reset
   ```

**Configuration**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `SCORE_THRESHOLD` | 0.5 | Minimum confidence for template update |
| `UPDATE_INTERVAL` | 200 | Frames between queue updates |
| `ONLINE_SIZE` | 2 | Number of online templates |
| `MAX_SCORE_DECAY` | 1.0 | Decay rate (1.0 = no decay) |

### 5.2 Target Loss Detection and Recovery

**Source**: `lib/test/tracker/target_loss_handler.py`

AquaSOT implements a dual-modal independent state machine for each modality, enabling robust tracking when one modality temporarily loses the target.

#### 5.2.1 Tracking States

| State | Description | Template Update |
|-------|-------------|-----------------|
| STABLE | Confident tracking (score >= 0.10) | Yes (if score > 0.7) |
| UNCERTAIN | Low confidence (0.01 <= score < 0.10) | No |
| LOST | Target lost (score < 0.01 for 2+ frames) | No |

**Note**: The extremely low thresholds ($\tau_{nf}=0.01$, $\tau_{s}=0.10$) mean the tracker remains in STABLE state unless confidence drops near zero. This design prioritizes continuous tracking over premature loss detection.

#### 5.2.2 State Machine Logic

**Score Classification**:
- `score < 0.01`: "not_found" -> target_lost signal
- `0.01 <= score < 0.10`: "uncertain" -> target_lost signal
- `score >= 0.10`: "stable" -> target_found signal

**State Transitions**:
- STABLE -> UNCERTAIN: First lost frame
- UNCERTAIN -> LOST: 2 consecutive lost frames (hysteresis)
- LOST -> UNCERTAIN: First found frame
- UNCERTAIN -> STABLE: 2 consecutive found frames (quick recovery)

#### 5.2.3 Primary Output Selection

When both modalities are STABLE:
- `PRIMARY_PRIORITY = "higher_score"` (default): Use modality with higher confidence
- `PRIMARY_PRIORITY = "rgb_first"`: Always prefer RGB
- `PRIMARY_PRIORITY = "sonar_first"`: Always prefer Sonar

When one modality is LOST:
- Automatically use the STABLE modality (enables Sonar to track when target leaves RGB FOV)

#### 5.2.4 Selective Template Update

Templates are only updated when:
1. Modality state is STABLE
2. Confidence score exceeds `HIGH_CONFIDENCE_THRESHOLD` (0.7)

This prevents template pollution from unreliable predictions.

#### 5.2.5 Dynamic Search Factor

**Source**: `lib/config/aquasot/config.py:256-260`

The search region expands based on tracking state to increase recovery probability:

| State | Factor | Description |
|-------|--------|-------------|
| STABLE | 4.0 | Normal tracking search region |
| UNCERTAIN | 5.0 | Expanded search for low confidence |
| LOST | 7.0 | Maximum expansion for re-detection |

**Search Region Computation**:
```
search_size = factor * sqrt(w_target * h_target)
```

### 5.3 Evaluation Protocol

**Source**: `lib/test/evaluation/experiment.py`, `lib/test/evaluation/metrics.py`

#### 5.3.0 Evaluation Settings (GOT-10k Style)

**Source**: `lib/config/aquasot/config.py:266-279`

AquaSOT follows GOT-10k evaluation protocol with visibility filtering:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FILTER_VISIBILITY` | True | Exclude absent frames from metrics |
| `SKIP_INIT_FRAME` | True | Exclude first frame (initialization) |
| `EVAL_MODE` | dual | Evaluation mode: dual/rgb_only/sonar_only |

**Visibility Filtering**:

Frames with ground truth `(0, 0, 0, 0)` indicate target absence (occlusion or out-of-view). These frames are excluded from metric computation:

```
F_eval = {t | B_t^gt != (0, 0, 0, 0), t > 0}
```

This prevents penalizing the tracker for absent targets and aligns with standard tracking benchmark protocols.

**Evaluation Modes**:
- `dual`: Evaluate both RGB and Sonar predictions independently
- `rgb_only`: Evaluate RGB predictions only
- `sonar_only`: Evaluate Sonar predictions only

#### 5.3.1 Success Rate (AUC)

Area under the success curve:

```
S(theta) = (1/N) * sum(1[IoU_t > theta])
AUC = integral_0^1 S(theta) d_theta
```

#### 5.3.2 Precision

Center location error within threshold:

```
e_t = sqrt((cx_pred - cx_gt)^2 + (cy_pred - cy_gt)^2)
Precision@20 = (1/N) * sum(1[e_t < 20])
```

#### 5.3.3 Average Center Error

Mean Euclidean distance between predicted and ground truth centers:

```
ACE = (1/N) * sum(e_t)
```

### 5.4 Evaluation Commands

```bash
# Run evaluation on validation set
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/checkpoints/aquasot_ep100.pth.tar

# Evaluate validation set
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/checkpoints/aquasot_ep100.pth.tar \
    --sequences uuv3

# With visualization
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/checkpoints/aquasot_ep100.pth.tar \
    --vis
```

---

## 6. Dataset

See `docs/specs/Dataset_Specification.md` for detailed dataset documentation.

### 6.1 Overview

The AquaSOT dataset contains 12 sequences of synchronized RGB-Sonar underwater imagery:

- **Training**: uuv1, uuv2, uuv4, uuv5, uuv6, uuv7, uuv8, uuv9, uuv10, uuv11, uuv12 (11 sequences, 10,940 frames)
- **Validation**: uuv3 (1 sequence, 751 frames)

### 6.2 Directory Structure

```
data/
|-- uuv1/
|   |-- light/              # RGB modality
|   |   |-- img/            # RGB images
|   |   |-- groundtruth.txt # Bounding boxes (x,y,w,h)
|   |-- sonar/              # Sonar modality
|       |-- img_rect/       # Preprocessed rectangular images
|       |-- groundtruth_rect.txt  # Rect-converted boxes
|-- uuv2/ ... uuv12/
```

### 6.3 Preprocessing

Sonar images must be converted from polar to rectangular coordinates before training:

```bash
python scripts/preprocess_sonar.py --all
```

---

## 7. Configuration

### 7.1 Main Configuration File

**Source**: `lib/config/aquasot/config.py`

All default parameters are defined in `config.py`. YAML files only override these defaults.

### 7.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Model** | | |
| `MODEL.BACKBONE.RGB.TYPE` | vit_base_patch16_224 | RGB backbone architecture |
| `MODEL.BACKBONE.SONAR.IN_CHANNELS` | 1 | Sonar input channels (grayscale) |
| `MODEL.FUSION.TYPE` | dcma | Cross-modal fusion type |
| `MODEL.FUSION.NUM_LAYERS` | 2 | Number of DCMA layers |
| `MODEL.FUSION.NUM_HEADS` | 8 | Attention heads per layer |
| `MODEL.HEAD.TYPE` | CENTER | Prediction head type |
| **Data** | | |
| `DATA.TEMPLATE.SIZE` | 128 | Template image size |
| `DATA.SEARCH.SIZE` | 256 | Search region size |
| `DATA.TEMPLATE.FACTOR` | 2.0 | Template crop area factor |
| `DATA.SEARCH.FACTOR` | 4.0 | Search crop area factor |
| `DATA.SEARCH.CENTER_JITTER` | 1.0 | Center jitter factor |
| `DATA.SEARCH.SCALE_JITTER` | 0.25 | Scale jitter factor |
| **Training** | | |
| `TRAIN.EPOCH` | 100 | Number of training epochs |
| `TRAIN.BATCH_SIZE` | 2 | Batch size per GPU |
| `TRAIN.LR` | 0.0001 | Base learning rate |
| `TRAIN.GIOU_WEIGHT` | 2.0 | GIoU loss weight |
| `TRAIN.L1_WEIGHT` | 5.0 | L1 loss weight |
| `TRAIN.FOCAL_WEIGHT` | 1.0 | Focal loss weight |
| **Test** | | |
| `TEST.TEMPLATE_UPDATE.ENABLED` | True | Enable template update |
| `TEST.TEMPLATE_UPDATE.STRATEGY` | mixformer | Update strategy |
| `TEST.TEMPLATE_UPDATE.ONLINE_SIZE` | 2 | Online template queue size |
| `TEST.TEMPLATE_UPDATE.UPDATE_INTERVAL` | 200 | Frames between updates |
| `TEST.TEMPLATE_UPDATE.HIGH_CONFIDENCE_THRESHOLD` | 0.7 | Min score for template update |
| **Fusion** | | |
| `MODEL.FUSION.RGB_FUSION_WEIGHT` | 0.3 | RGB cross-modal attention strength |
| `MODEL.FUSION.SONAR_FUSION_WEIGHT` | 0.3 | Sonar cross-modal attention strength |
| **Target Loss Detection** | | |
| `TEST.TARGET_LOSS.ENABLED` | True | Enable target loss detection |
| `TEST.TARGET_LOSS.NOT_FOUND_THRESHOLD` | 0.01 | Score below = target lost |
| `TEST.TARGET_LOSS.UNCERTAIN_THRESHOLD` | 0.05 | Score below = uncertain |
| `TEST.TARGET_LOSS.STABLE_THRESHOLD` | 0.10 | Score above = confident |
| `TEST.TARGET_LOSS.LOST_CONFIRM_FRAMES` | 2 | Frames to confirm LOST |
| `TEST.TARGET_LOSS.FOUND_CONFIRM_FRAMES` | 5 | Frames to confirm recovery |
| `TEST.TARGET_LOSS.QUICK_RECOVERY_FRAMES` | 2 | Quick recovery from UNCERTAIN |
| `TEST.TARGET_LOSS.SEARCH_REGION_K` | 5.0 | Search region = K * target size |
| `TEST.TARGET_LOSS.PRIMARY_PRIORITY` | higher_score | Primary output selection |
| **Dynamic Search Factor** | | |
| `TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.STABLE` | 4.0 | Search factor for STABLE state |
| `TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.UNCERTAIN` | 5.0 | Search factor for UNCERTAIN state |
| `TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.LOST` | 7.0 | Search factor for LOST state |
| **Evaluation** | | |
| `TEST.EVALUATION.FILTER_VISIBILITY` | True | Exclude absent frames from metrics |
| `TEST.EVALUATION.SKIP_INIT_FRAME` | True | Exclude first frame |
| `TEST.EVALUATION.EVAL_MODE` | dual | dual/rgb_only/sonar_only |
| **Modality Dropout (Ablation)** | | |
| `DATA.MODALITY_DROPOUT.ENABLED` | False | Enable testing-time dropout |
| `DATA.MODALITY_DROPOUT.P_DROP_RGB` | 0.0 | Probability of dropping RGB |
| `DATA.MODALITY_DROPOUT.P_DROP_SONAR` | 0.0 | Probability of dropping Sonar |
| **BatchNorm Adaptation** | | |
| `TEST.BN_ADAPTATION.ENABLED` | False | Enable BN adaptation for ablation |
| `TEST.BN_ADAPTATION.NUM_BATCHES` | 30 | Batches for BN statistics recalculation |
| `TEST.BN_ADAPTATION.DATA_SOURCE` | 'test' | Data source for adaptation |
| **Default Test Sequences** | | |
| `DATA.TEST.SEQUENCES` | ["uuv3"] | Default sequences for test/ablation |

### 7.3 Configuration Files

| File | Description |
|------|-------------|
| `vitb_256_dual.yaml` | Main training configuration |
| `vitb_256_dual_scratch.yaml` | Training from scratch (no pretrain) |
| `vitb_256_dual_auto.yaml` | Auto-augmentation variant |

---

## 8. Ablation Experiments

AquaSOT supports inference-time ablation experiments using runtime CLI flags. All ablations use BatchNorm adaptation (`--bn_adapt`) to recalculate BN statistics for fair comparison.

### 8.1 Evaluation Modes

| Mode | Flag | Description |
|------|------|-------------|
| Dual (default) | `--eval_mode dual` | Full dual-modal tracking |
| RGB-only | `--eval_mode rgb_only` | Zeros Sonar input |
| Sonar-only | `--eval_mode sonar_only` | Zeros RGB input |
| No-fusion | `--eval_mode no_fusion` | Both modalities, fusion weights = 0 |

### 8.2 BatchNorm Adaptation

**Source**: `lib/utils/bn_adaptation.py`

**Why it's needed**: During dual-modal training, BatchNorm layers learn statistics from fused features. At single-modal inference, the feature distribution changes because cross-attention output differs. Without BN adaptation, this distribution mismatch degrades performance.

**How it works**:
1. Reset BN running_mean and running_var
2. Run forward passes with target modality configuration (30 batches, momentum=0.2)
3. BN statistics converge to the new distribution
4. Restore eval mode with adapted statistics

**When to use**: Always use `--bn_adapt` for ablation experiments.

**Related Work**: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021.

### 8.3 Running Ablation Experiments

```bash
# Dual-modal baseline (no BN adaptation needed)
python tracking/test.py --checkpoint model.pth --eval_mode dual

# RGB-only ablation (requires BN adaptation)
python tracking/test.py --checkpoint model.pth --eval_mode rgb_only --bn_adapt

# Sonar-only ablation (requires BN adaptation)
python tracking/test.py --checkpoint model.pth --eval_mode sonar_only --bn_adapt

# No-fusion ablation (requires BN adaptation)
python tracking/test.py --checkpoint model.pth --eval_mode no_fusion --bn_adapt
```

### 8.4 Technical Details

#### 8.4.1 Modality Validity Detection

**Source**: `lib/models/aquasot/aquasot.py:73-74`

```python
rgb_valid = rgb_search.abs().sum() > 1e-4
sonar_valid = sonar_search.abs().sum() > 1e-4
```

#### 8.4.2 Self-Attention Fallback

**Source**: `lib/models/aquasot/fusion.py:125-145`

When one modality is dropped:
- **Sonar dropped**: RGB uses self-attention via `rgb_to_sonar` module
- **RGB dropped**: Sonar uses `rgb_to_sonar` module (K/V projections trained for Sonar)

#### 8.4.3 BN Adaptation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEST.BN_ADAPTATION.ENABLED` | False | Enable via `--bn_adapt` |
| `TEST.BN_ADAPTATION.NUM_BATCHES` | 30 | Batches for adaptation |
| `TEST.BN_ADAPTATION.DATA_SOURCE` | 'test' | Data source for adaptation |

### 8.5 Expected Results

| Experiment | Description | Expected Behavior |
|------------|-------------|-------------------|
| Full Dual | Baseline dual-modal | Best overall performance |
| RGB-only | Sonar zeroed | Performance degradation in low-visibility conditions |
| Sonar-only | RGB zeroed | Lower precision due to coarser resolution |
| No-fusion | Fusion weights = 0 | No cross-modal information exchange |

**Note**: Without `--bn_adapt`, single-modal ablations show significantly degraded performance due to BN statistics mismatch.

---

## 9. References

### Primary References

[1] **OSTrack** (Base Framework)
- Ye, B., Chang, H., Ma, B., & Shan, S. (2022). *Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework*. ECCV 2022.
- [arXiv:2203.11991](https://arxiv.org/abs/2203.11991) | [Code](https://github.com/botaoye/OSTrack)
- Local: `external/papers/ostrack_eccv2022.pdf`

[2] **MixFormer** (Template Update Strategy)
- Cui, Y., Jiang, C., Wang, L., & Wu, G. (2022). *MixFormer: End-to-End Tracking with Iterative Mixed Attention*. CVPR 2022.
- [arXiv:2203.11082](https://arxiv.org/abs/2203.11082) | [Code](https://github.com/MCG-NJU/MixFormer)
- Local: `external/papers/mixformer_cvpr2022.pdf`

[3] **Vision Transformer** (Backbone Architecture)
- Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
- [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Local: `external/papers/vit_iclr2021.pdf`

[4] **MAE** (Pretraining Strategy)
- He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022.
- [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Local: `external/papers/mae_cvpr2022.pdf`

### Loss Functions

[5] **Focal Loss**
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.
- [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Local: `external/papers/focal_loss_iccv2017.pdf`

[6] **Generalized IoU**
- Rezatofighi, H., et al. (2019). *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*. CVPR 2019.
- [arXiv:1902.09630](https://arxiv.org/abs/1902.09630)
- Local: `external/papers/giou_cvpr2019.pdf`

### Dataset

[7] **RGBS50** (RGB-Sonar Tracking Benchmark)
- Li, Y., et al. (2024). *RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker*.
- [arXiv:2406.07189](https://arxiv.org/abs/2406.07189) | [Code](https://github.com/LiYunfengLYF/RGBS50)
- Local: `external/papers/rgbs50_2024.pdf`

### Architecture References

[8] **CenterNet** (Center-based Detection)
- Zhou, X., Wang, D., & Krahenbuhl, P. (2019). *Objects as Points*.
- [arXiv:1904.07850](https://arxiv.org/abs/1904.07850)

[9] **AdamW** (Optimizer)
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019.
- [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

[10] **Tent** (Test-Time Adaptation)
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). *Tent: Fully Test-Time Adaptation by Entropy Minimization*. ICLR 2021.
- [arXiv:2006.10726](https://arxiv.org/abs/2006.10726)

---

## 10. Appendix: Code Reference Index

### 10.1 Model Implementation

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Main Model | `lib/models/aquasot/aquasot.py` | 16-177 | AquaSOT class, forward pass |
| ViT Backbone | `lib/models/aquasot/vit.py` | 89-165 | VisionTransformer class |
| Attention | `lib/models/aquasot/vit.py` | 30-59 | Multi-head self-attention |
| Transformer Block | `lib/models/aquasot/vit.py` | 62-86 | Pre-norm transformer block |
| Channel Adaptation | `lib/models/aquasot/vit.py` | 167-193 | RGB to Sonar weight adaptation |
| DCMA Fusion | `lib/models/aquasot/fusion.py` | 128-167 | DCMA module |
| Cross-Modal Attention | `lib/models/aquasot/fusion.py` | 19-62 | CrossModalAttention class |
| DCMA Layer | `lib/models/aquasot/fusion.py` | 65-122 | DCMALayer class |
| CenterPredictor | `lib/models/layers/head.py` | 31-166 | Prediction head |

### 10.2 Training

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Training Actor | `lib/train/actors/aquasot.py` | 16-228 | Forward pass and loss computation |
| Loss Computation | `lib/train/actors/aquasot.py` | 116-227 | compute_losses method |
| Data Processing | `lib/train/data/processing.py` | 38-378 | MultiModalProcessing class |
| Jitter Params | `lib/train/data/processing.py` | 89-131 | Synchronized jitter |

### 10.3 Loss Functions

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Focal Loss | `lib/utils/focal_loss.py` | 10-59 | Focal loss implementation |
| GIoU Loss | `lib/utils/box_ops.py` | 93-136 | Generalized IoU |
| L1 Loss | `lib/utils/box_ops.py` | 139-149 | L1 bounding box loss |
| Heatmap Generation | `lib/utils/heatmap_utils.py` | 9-76 | Gaussian heatmap (generate_heatmap, generate_single_heatmap) |

### 10.4 Inference

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Tracker | `lib/test/tracker/aquasot_tracker.py` | 18-543 | AquaSOTTracker class |
| Template Update | `lib/test/tracker/aquasot_tracker.py` | 296-370 | Three-tier update strategy |
| Target Loss Handler | `lib/test/tracker/target_loss_handler.py` | 1-385 | Dual-modal state machine |
| Preprocessing | `lib/test/tracker/aquasot_tracker.py` | 459-500 | RGB/Sonar preprocessing |
| Evaluation | `lib/test/evaluation/experiment.py` | 23-766 | Evaluation experiment runner |
| Modality Dropout | `lib/utils/modality_dropout.py` | 28-77 | Testing-time ablation |
| BN Adaptation | `lib/utils/bn_adaptation.py` | 21-131 | BatchNorm adaptation for ablation |

### 10.5 Training Utilities

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Checkpoint Migration | `lib/train/trainers/base_trainer.py` | 125-148 | box_head to separate heads |

### 10.6 Configuration

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Default Config | `lib/config/aquasot/config.py` | 1-443 | All default parameters |
| Model Config | `lib/config/aquasot/config.py` | 19-73 | Model architecture params |
| Training Config | `lib/config/aquasot/config.py` | 78-131 | Training hyperparameters |
| Data Config | `lib/config/aquasot/config.py` | 135-236 | Data and augmentation |
| Test Config | `lib/config/aquasot/config.py` | 241-264 | Inference parameters |

---

## Document Information

- **Version**: 2.2
- **Last Updated**: 2026-01-31
- **Based on**: Current codebase implementation
- **Related Documents**:
  - `docs/specs/AquaSOT_Mathematical_Formulation.md` (detailed mathematical derivations)
  - `docs/specs/Dataset_Specification.md` (dataset details)
