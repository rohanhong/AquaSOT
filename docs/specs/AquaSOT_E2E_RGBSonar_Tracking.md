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
                                            | CenterHead   |          | CenterHead   |
                                            |    (RGB)     |          |   (Sonar)    |
                                            +--------------+          +--------------+
                                                    |                         |
                                                    v                         v
                                            RGB Bbox [B, 4]           Sonar Bbox [B, 4]
                                            RGB Score [B, 1, 16, 16]  Sonar Score [B, 1, 16, 16]

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

### 3.3 CenterPredictor Head

**Source**: `lib/models/layers/head.py:31-165`

The prediction head uses three parallel convolutional branches.

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

### 5.2 Evaluation Protocol

**Source**: `lib/test/evaluation/metrics.py`

#### 5.2.1 Success Rate (AUC)

Area under the success curve:

```
S(theta) = (1/N) * sum(1[IoU_t > theta])
AUC = integral_0^1 S(theta) d_theta
```

#### 5.2.2 Precision

Center location error within threshold:

```
e_t = sqrt((cx_pred - cx_gt)^2 + (cy_pred - cy_gt)^2)
Precision@20 = (1/N) * sum(1[e_t < 20])
```

#### 5.2.3 Average Center Error

Mean Euclidean distance between predicted and ground truth centers:

```
ACE = (1/N) * sum(e_t)
```

### 5.3 Evaluation Commands

```bash
# Run evaluation on validation set
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/checkpoints/aquasot_ep100.pth.tar

# Evaluate specific sequences
python tracking/test.py --tracker aquasot \
    --checkpoint outputs/train/checkpoints/aquasot_ep100.pth.tar \
    --sequences uuv3 uuv6 uuv10

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

- **Training**: uuv1, uuv2, uuv4, uuv5, uuv7, uuv8, uuv9, uuv11, uuv12 (9 sequences, 9,588 frames)
- **Validation**: uuv3, uuv6, uuv10 (3 sequences, 2,103 frames)

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

### 7.3 Configuration Files

| File | Description |
|------|-------------|
| `vitb_256_dual.yaml` | Main training configuration |
| `vitb_256_dual_scratch.yaml` | Training from scratch (no pretrain) |
| `vitb_256_dual_auto.yaml` | Auto-augmentation variant |
| `ablation_rgb_only.yaml` | RGB modality only |
| `ablation_sonar_only.yaml` | Sonar modality only |
| `ablation_no_fusion.yaml` | No cross-modal fusion |

---

## 8. Ablation Experiments

### 8.1 Available Ablation Configurations

| Configuration | Modification |
|--------------|--------------|
| `ablation_rgb_only.yaml` | Disable sonar branch, RGB-only tracking |
| `ablation_sonar_only.yaml` | Disable RGB branch, Sonar-only tracking |
| `ablation_no_fusion.yaml` | Replace DCMA with identity (no fusion) |

### 8.2 Running Ablations

```bash
# RGB-only ablation
python tracking/train.py --script aquasot --config ablation_rgb_only

# Sonar-only ablation
python tracking/train.py --script aquasot --config ablation_sonar_only

# No fusion ablation
python tracking/train.py --script aquasot --config ablation_no_fusion
```

### 8.3 Expected Comparisons

| Experiment | Purpose |
|------------|---------|
| Full vs RGB-only | Quantify sonar contribution |
| Full vs Sonar-only | Quantify RGB contribution |
| Full vs No-fusion | Quantify cross-modal fusion benefit |

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
| DCMA Fusion | `lib/models/aquasot/fusion.py` | 125-156 | DCMA module |
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
| Preprocessing | `lib/test/tracker/aquasot_tracker.py` | 459-500 | RGB/Sonar preprocessing |

### 10.5 Configuration

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Default Config | `lib/config/aquasot/config.py` | 1-386 | All default parameters |
| Model Config | `lib/config/aquasot/config.py` | 19-73 | Model architecture params |
| Training Config | `lib/config/aquasot/config.py` | 78-131 | Training hyperparameters |
| Data Config | `lib/config/aquasot/config.py` | 135-236 | Data and augmentation |
| Test Config | `lib/config/aquasot/config.py` | 241-264 | Inference parameters |

---

## Document Information

- **Version**: 2.0
- **Last Updated**: 2026-01-26
- **Based on**: Current codebase implementation
- **Related Documents**:
  - `docs/specs/AquaSOT_Mathematical_Formulation.md` (detailed mathematical derivations)
  - `docs/specs/Dataset_Specification.md` (dataset details)
