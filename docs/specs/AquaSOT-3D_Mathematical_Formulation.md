# AquaSOT-3D: Mathematical Formulation and Algorithmic Framework

**End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Center Estimation**

---

## Table of Contents

1. [Introduction and Problem Formulation](#1-introduction-and-problem-formulation)
2. [Mathematical Notation](#2-mathematical-notation)
3. [Sonar Coordinate System Transformation](#3-sonar-coordinate-system-transformation)
4. [G²-DCMA: Geometry-Gated Deformable Cross-Modal Attention](#4-g-dcma-geometry-gated-deformable-cross-modal-attention)
   - 4.1 [Geometric Gating Mechanism](#41-geometric-gating-mechanism)
   - 4.2 [Deformable Cross-Modal Attention](#42-deformable-cross-modal-attention)
   - 4.3 [Dual Consistency Loss](#43-dual-consistency-loss)
5. [Vision-Mamba Temporal Modeling](#5-vision-mamba-temporal-modeling)
6. [Weak 3D Center Estimation](#6-weak-3d-center-estimation)
7. [Geometric Cyclic Projection](#7-geometric-cyclic-projection)
8. [Complete Loss Function and Optimization](#8-complete-loss-function-and-optimization)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [References](#10-references)

---

## 1. Introduction and Problem Formulation

### 1.1 Problem Statement

AquaSOT-3D addresses the challenge of **calibration-free multi-modal single-object tracking** in underwater environments, where RGB camera and Forward-Looking Sonar (FLS) sensors are:
- **Temporally aligned** but **spatially unaligned** (no known extrinsic calibration)
- Subject to complementary failure modes (turbidity affects RGB, acoustic noise affects sonar)
- Require robust fusion without manual calibration

### 1.2 Input Specification

**Input at frame $t$**:

$$
\mathcal{I}_t = \{I_t^{\text{rgb}}, I_t^{\text{sonar}}, B_t^{\text{rgb}}, B_t^{\text{sonar}}, V_t^{\text{rgb}}, V_t^{\text{sonar}}\}
$$

where:
- $I_t^{\text{rgb}} \in \mathbb{R}^{3 \times H_{\text{rgb}} \times W_{\text{rgb}}}$: RGB image (standard: $1080 \times 1920$)
- $I_t^{\text{sonar}} \in \mathbb{R}^{1 \times H_{\text{sonar}} \times W_{\text{sonar}}}$: Sonar intensity image (preprocessed to rectangular coordinates: $376 \times 682$ pixels)
- $B_t^{\text{rgb}} = (c_x^{\text{rgb}}, c_y^{\text{rgb}}, w_t^{\text{rgb}}, h_t^{\text{rgb}})$: RGB bounding box in cxcywh format (training only)
- $B_t^{\text{sonar}} = (c_x^{\text{sonar}}, c_y^{\text{sonar}}, w_t^{\text{sonar}}, h_t^{\text{sonar}})$: Sonar bounding box in cxcywh format (training only)
- $V_t^{\text{rgb}}, V_t^{\text{sonar}} \in \{0, 1\}$: **Validity flags** indicating whether the sample is valid for loss computation

**Missing Frame Handling**: When modality $m$ is occluded or unavailable:
- Ground truth: $B_t^{m} = (c_x^{\text{default}}, c_y^{\text{default}}, w^{\text{default}}, h^{\text{default}})$ with $V_t^{m} = 0$
- System applies validity-aware loss masking (see Section 8.6)

### 1.3 Output Specification

**Output at frame $t$**:

$$
\mathcal{O}_t = \{\hat{B}_t^{\text{rgb}}, \hat{B}_t^{\text{sonar}}, \hat{B}_t^{\text{fusion}}, \hat{\mathbf{C}}_t^{3D}\}
$$

where:
- $\hat{B}_t^{\text{rgb}}$: Predicted RGB bounding box
- $\hat{B}_t^{\text{sonar}}$: Predicted Sonar bounding box
- $\hat{B}_t^{\text{fusion}}$: **Primary fusion bounding box** (main tracking output)
- $\hat{\mathbf{C}}_t^{3D} = (X_t, Y_t, Z_t) \in \mathbb{R}^3$: **Weak 3D center** in Fusion Frame

### 1.4 Coordinate System Convention

**Fusion Frame** (unified 3D coordinate system):

$$
\begin{cases}
X &: \text{Horizontal (right positive)} \\
Y &: \text{Vertical (down positive, image convention)} \\
Z &: \text{Depth/Forward (away from sensor positive)}
\end{cases}
$$

**Bounding Box Formats**:
- **cxcywh**: $(c_x, c_y, w, h)$ - center point + width/height (model internal format)
- **xywh**: $(x, y, w, h)$ - top-left corner + width/height (dataset format)
- **xyxy**: $(x_1, y_1, x_2, y_2)$ - top-left + bottom-right corners (IoU computation)

**Format Conversions**:
$$
\begin{aligned}
\text{cxcywh} \rightarrow \text{xywh}: & \quad (x, y, w, h) = (c_x - w/2, c_y - h/2, w, h) \\
\text{xywh} \rightarrow \text{cxcywh}: & \quad (c_x, c_y, w, h) = (x + w/2, y + h/2, w, h) \\
\text{cxcywh} \rightarrow \text{xyxy}: & \quad (x_1, y_1, x_2, y_2) = (c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2)
\end{aligned}
$$

**Alignment**: RGB frame is aligned with Fusion frame; Sonar-to-Fusion transformation is learned implicitly.

### 1.5 Key Technical Contributions

**1. G²-DCMA Fusion Module** (Sections 3-4)
Calibration-free cross-modal attention combining:
- Fixed polar-to-rectangular coordinate preprocessing
- Geometric gating $g(\mathbf{r}, \mathbf{s})$ from sonar radial and RGB scale cues
- Deformable sampling offsets $\Delta \mathbf{p}$ adapted from Deformable DETR [Zhu et al., 2021]
- Dual consistency: component-level ($X$ coordinate agreement) + representation-level (InfoNCE)

**2. Weak 3D Center Estimation** (Sections 6-7)
Depth estimation without extrinsic calibration:
- $Z$ from sonar radial activation via learnable weighted integral
- $X$ via dual-branch regression (RGB + Sonar) with adaptive fusion
- $Y$ from RGB-dominant prediction
- Geometric cyclic projection with inverse loss $\mathcal{L}_{\text{cycle}}^{3D} = \|\tilde{\mathbf{C}}^{3D} - \mathbf{C}^{3D}\|_1$

**3. Vision-Mamba Temporal Modeling** (Section 5)
Linear-complexity SSM for long-sequence tracking:
- $O(T)$ complexity vs. $O(T^2)$ for transformers
- Input-dependent parameters ($\mathbf{B}_t, \mathbf{C}_t, \Delta_t$) for adaptive history
- Bidirectional processing with occlusion-aware gating

**Comparison with SCANet (RGBS50 Baseline)**: SCANet uses spatial cross-attention (SCAM) with fixed global integration. AquaSOT-3D introduces learnable geometric gating, deformable alignment, weak 3D estimation with cyclic constraints, and linear-complexity temporal modeling.

---

## 2. Mathematical Notation

### 2.1 Core Symbols

**Dimension Variables**:

| Symbol | Description |
|--------|-------------|
| $B$ | Batch size |
| $T$ | Temporal sequence length |
| $C$, $D$ | Feature channel dimension |
| $H$, $W$ | Spatial height and width |

**Input/Output Tensors**:

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $I^{\text{rgb}}$ | RGB image | $\mathbb{R}^{3 \times H \times W}$ |
| $I^{\text{sonar}}$ | Sonar image | $\mathbb{R}^{1 \times H' \times W'}$ |
| $\hat{B}^{m}$ | Predicted box (modality $m$) | $\mathbb{R}^{4}$ (cx, cy, w, h) |
| $\mathbf{C}^{3D}$ | Weak 3D center | $\mathbb{R}^{3}$ (X, Y, Z) |

**Feature Representations**:

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $\mathcal{F}^{\text{rgb}}$ | RGB feature map | $\mathbb{R}^{B \times C \times H \times W}$ |
| $\mathcal{F}^{\text{sonar}}$ | Sonar feature map | $\mathbb{R}^{B \times C \times H' \times W'}$ |
| $\mathcal{F}^{\text{fused}}$ | Fused feature map | $\mathbb{R}^{B \times C \times H \times W}$ |
| $\mathbf{h}_t$ | Temporal state at time $t$ | $\mathbb{R}^{B \times D}$ |

**Module-Specific Symbols**:

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $\mathbf{r}$ | Radial activation profile | $\mathbb{R}^{B \times N_{\text{bins}}}$ ($N_{\text{bins}} = 50$) |
| $\mathbf{s}$ | Scale pyramid encoding | $\mathbb{R}^{B \times D_s}$ |
| $\mathbf{g}$ | Geometric gating weights | $\mathbb{R}^{B \times N_{\text{heads}}}$ ($N_{\text{heads}} = 8$) |
| $\Delta \mathbf{p}$ | Deformable sampling offsets | $\mathbb{R}^{B \times N_q \times H \times K \times 2}$ ($K = 4$) |
| $\tau$ | InfoNCE temperature | 0.07 |
| $\epsilon$ | Numerical stability constant | $10^{-6}$ or $10^{-7}$ |

### 2.2 Operation Notation

- $\otimes$ or $\odot$: Element-wise multiplication (Hadamard product)
- $\oplus$: Element-wise addition
- $\|\cdot\|_1$: L1 norm
- $\|\cdot\|_2$: L2 norm
- $\|\cdot\|_p$: $L_p$ norm
- $[\cdot;\cdot]$: Concatenation
- $\sigma(\cdot)$: Sigmoid function $\frac{1}{1 + e^{-x}}$
- $\text{softmax}(\cdot)$: Softmax normalization $\frac{e^{x_i}}{\sum_j e^{x_j}}$
- $\text{softplus}(\cdot)$: Softplus function $\log(1 + e^x)$
- $\text{GAP}(\cdot)$: Global Average Pooling
- $\mathbb{1}[\cdot]$: Indicator function
- $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$: Attention mechanism

---

## 3. Sonar Coordinate System Transformation

### 3.1 Motivation

Forward-Looking Sonar (FLS) images exhibit:
- **Fan-shaped field of view** with radial structure
- **Non-uniform spatial resolution**: higher near sensor, lower at range
- **Radial-angular parameterization** in native polar coordinates

To simplify the learning process and improve training stability, AquaSOT-3D employs a **polar-to-rectangular coordinate transformation** as a preprocessing step, eliminating the need for runtime coordinate transformations.

### 3.2 Polar-to-Rectangular Transformation

The transformation maps from polar coordinates to rectangular coordinates, where the rectangular representation provides a more uniform spatial structure for neural network processing.

#### 3.2.1 Mathematical Formulation

For each output pixel $(x_{\text{rect}}, y_{\text{rect}})$ in the rectangular coordinate system with dimensions $W_{\text{out}} \times H_{\text{out}}$:

**Step 1: Normalize rectangular coordinates to unit range**
$$
\begin{aligned}
\theta_{\text{norm}} &= \frac{x_{\text{rect}}}{W_{\text{out}} - 1} \in [0, 1] \\
r_{\text{norm}} &= \frac{H_{\text{out}} - 1 - y_{\text{rect}}}{H_{\text{out}} - 1} \in [0, 1]
\end{aligned}
$$

**Step 2: Map to physical polar coordinates**
$$
\begin{aligned}
\theta &= -\frac{\text{FOV}}{2} + \theta_{\text{norm}} \cdot \text{FOV} \\
r &= r_{\text{norm}} \cdot r_{\max}
\end{aligned}
$$

where:
- FOV: Field of view angle in radians (typically $130° \approx 2.27$ radians)
- $r_{\max}$: Maximum sensing range

**Step 3: Convert to Cartesian source coordinates**
$$
\begin{aligned}
x_{\text{src}} &= c_x + r \sin(\theta) \\
y_{\text{src}} &= c_y - r \cos(\theta)
\end{aligned}
$$

where $(c_x, c_y)$ represents the sensor position in the original polar coordinate frame.

**Step 4: Bilinear interpolation**
$$
I_{\text{rect}}(x_{\text{rect}}, y_{\text{rect}}) = \sum_{i,j \in \mathcal{N}} w_{ij} \cdot I_{\text{polar}}(i, j)
$$

where $\mathcal{N}$ denotes the 2×2 neighborhood around $(x_{\text{src}}, y_{\text{src}})$ and $w_{ij}$ are bilinear interpolation weights.

#### 3.2.2 Field-of-View Determination

The field of view is determined by the geometric configuration of the sonar sensor:

$$
\text{FOV} = \theta_{\max} - \theta_{\min}
$$

where $\theta_{\min}$ and $\theta_{\max}$ represent the angular boundaries of the sonar's sensing region.

### 3.3 Rectangular Coordinate Properties

After transformation, the sonar data exhibits the following properties in rectangular coordinates:

- **X-axis**: Represents bearing angle $\theta$ (horizontal dimension)
- **Y-axis**: Represents range distance $r$ (vertical dimension, with near range at bottom)
- **Sensor position**: Located at bottom-center $(c_x, c_y) = ((W-1)/2, H-1)$

**Important Note**: The rectangular coordinates encode **heterogeneous physical quantities** - angle along the X-axis and distance along the Y-axis. This has critical implications for distance computations (see Section 4.1).

### 3.4 Bounding Box Transformation

Target annotations must also be transformed from polar to rectangular coordinates. The transformation preserves object boundaries through a corner-based mapping approach.

#### 3.4.1 Corner-Based Transformation Algorithm

Given a bounding box $B_{\text{polar}} = (x, y, w, h)$ in polar coordinates:

1. **Extract corner points**:
$$
\mathcal{C} = \{(x, y), (x+w, y), (x+w, y+h), (x, y+h)\}
$$

2. **Transform each corner** $(x_c, y_c) \in \mathcal{C}$:
$$
\begin{aligned}
\Delta x &= x_c - c_x, \quad \Delta y = c_y - y_c \\
\theta_c &= \arctan2(\Delta x, \Delta y) \\
r_c &= \sqrt{\Delta x^2 + \Delta y^2} \\
\end{aligned}
$$

Then normalize and map to rectangular coordinates:
$$
\begin{aligned}
\theta_{\text{norm}} &= \frac{\theta_c - \theta_{\min}}{\theta_{\max} - \theta_{\min}} \\
r_{\text{norm}} &= \frac{r_c}{r_{\max}} \\
x_{\text{rect}} &= \theta_{\text{norm}} \cdot (W_{\text{out}} - 1) \\
y_{\text{rect}} &= (1 - r_{\text{norm}}) \cdot (H_{\text{out}} - 1)
\end{aligned}
$$

3. **Compute axis-aligned bounding box**:
$$
B_{\text{rect}} = \left(\min_{i}(x_i), \min_{i}(y_i), \max_{i}(x_i) - \min_{i}(x_i), \max_{i}(y_i) - \min_{i}(y_i)\right)
$$

### 3.5 Theoretical Advantages

The polar-to-rectangular transformation provides several theoretical benefits:

1. **Spatial Uniformity**: Rectangular grids align better with convolutional neural network architectures
2. **Training Stability**: Fixed transformation eliminates gradient flow through coordinate warping
3. **Computational Efficiency**: No runtime transformation overhead during training
4. **Resolution Consistency**: Unified spatial dimensions across all data samples

---

## 4. G²-DCMA: Geometry-Gated Deformable Cross-Modal Attention

G²-DCMA is the **core fusion module** that integrates three complementary mechanisms:

1. **Geometric Gating**: Modulates attention based on radial-scale consistency
2. **Deformable Attention**: Learns sparse sampling offsets for spatial alignment
3. **Dual Consistency**: Enforces component and representation consistency

### 4.1 Geometric Gating Mechanism

#### 4.1.1 Radial Profile Extraction (Sonar)

**Objective**: Extract 1D radial activation distribution from 2D sonar features.

Given sonar feature map $\mathcal{F}^{\text{sonar}} \in \mathbb{R}^{B \times C \times H' \times W'}$:

1. **Compute radial distance map**:

After polar-to-rectangular transformation, the coordinate system has special properties:
- X-axis represents bearing angle (horizontal field of view)
- Y-axis represents range distance from sensor

Therefore, the radial distance in rectangular coordinates is:

$$
R(i, j) = c_y - i, \quad \forall (i, j) \in [0, H') \times [0, W')
$$

where $(c_x, c_y) = (W'/2, H'-1)$ with sensor positioned at bottom-center.

**Important**: The Euclidean distance formula $\sqrt{(j - c_x)^2 + (i - c_y)^2}$ is **incorrect** here because it mixes heterogeneous physical quantities (angle and distance).

2. **Discretize into $N_{\text{bins}} = 50$ radial bins**:

$$
\text{bin}(i, j) = \left\lfloor \frac{R(i, j)}{R_{\max}} \cdot (N_{\text{bins}} - 1) \right\rfloor
$$

3. **Aggregate feature activations per bin** (with learnable channel weights $\mathbf{w} \in \mathbb{R}^C$):

First, normalize channel weights:

$$
w_c = \frac{\tilde{w}_c}{\sum_{c'=1}^{C} \tilde{w}_{c'} + \epsilon}, \quad \epsilon = 10^{-6}
$$

Then compute weighted channel aggregation followed by spatial binning:

$$
r_k = \frac{1}{|\mathcal{B}_k|} \sum_{(i,j) \in \mathcal{B}_k} \left( \sum_{c=1}^{C} w_c \cdot \mathcal{F}^{\text{sonar}}_{c,i,j} \right), \quad k = 0, \ldots, N_{\text{bins}}-1
$$

where $\mathcal{B}_k = \{(i,j) : \text{bin}(i,j) = k\}$ is the set of spatial locations in bin $k$.

4. **Encode radial profile**:

$$
\mathbf{r} = \text{MLP}_{\text{radial}}([r_0, r_1, \ldots, r_{N_{\text{bins}}-1}]) \in \mathbb{R}^{B \times D_r}
$$

#### 4.1.2 Scale Pyramid Extraction (RGB)

**Objective**: Extract scale/depth information from RGB feature pyramid.

Given RGB feature $\mathcal{F}^{\text{rgb}} \in \mathbb{R}^{B \times C \times H \times W}$:

1. **Generate feature pyramid** (if not provided):

$$
\mathcal{P}_{\ell} = \text{AvgPool}_{2^{\ell}}(\mathcal{F}^{\text{rgb}}), \quad \ell = 0, 1, \ldots, L-1
$$

2. **Extract scale-specific features**:

$$
\mathbf{s}_{\ell} = \text{GlobalAvgPool}\big(\text{Conv}_{1 \times 1}(\mathcal{P}_{\ell})\big) \in \mathbb{R}^{B \times (C/2)}
$$

3. **Attention-based scale aggregation**:

$$
\begin{aligned}
\mathbf{a} &= \text{softmax}\big(\text{MLP}_{\text{attn}}([\mathbf{s}_0; \mathbf{s}_1; \ldots; \mathbf{s}_{L-1}])\big) \in \mathbb{R}^{B \times L} \\
\mathbf{s}^{\text{agg}} &= \sum_{\ell=0}^{L-1} a_{\ell} \cdot \mathbf{s}_{\ell}
\end{aligned}
$$

4. **Encode scale information**:

$$
\mathbf{s} = \text{MLP}_{\text{scale}}(\mathbf{s}^{\text{agg}}) \in \mathbb{R}^{B \times D_s}
$$

#### 4.1.3 Geometric Gating Function

**Gating Network**: Learn geometric consistency function $g: \mathbb{R}^{D_r + D_s} \rightarrow [0, 1]^{H}$

$$
\mathbf{g} = \sigma\big(\text{MLP}_{\text{gate}}([\mathbf{r}; \mathbf{s}])\big) \in \mathbb{R}^{B \times H}
$$

where $H$ is the number of attention heads.

**Interpretation**:
- $g_h \approx 1$: Radial distance (sonar) and scale (RGB) are geometrically consistent → **amplify** attention
- $g_h \approx 0$: Geometric mismatch → **suppress** attention

#### 4.1.4 Gated Cross-Modal Attention

**Standard Multi-Head Attention**:

$$
\begin{aligned}
\mathbf{Q} &= \mathcal{F}^{\text{rgb}} \mathbf{W}_Q, \quad \mathbf{K} = \mathcal{F}^{\text{sonar}} \mathbf{W}_K, \quad \mathbf{V} = \mathcal{F}^{\text{sonar}} \mathbf{W}_V \\
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
\end{aligned}
$$

where $d_k = D / H$ is the head dimension ($D$: embedding dimension, $H$: number of heads).

**Geometry-Gated Attention** (per head $h$):

Gating is applied to attention scores **before** softmax:

$$
\begin{aligned}
\mathbf{A}_h &= \frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}} \quad \text{[B, N_q, N_k]} \\
\mathbf{A}_h^{\text{gated}} &= g_h \odot \mathbf{A}_h \quad \text{(element-wise gating, } g_h \in \mathbb{R}^{B} \text{ broadcasted)} \\
\text{Attn}_h^{\text{gated}} &= \text{softmax}(\mathbf{A}_h^{\text{gated}}) \mathbf{V}_h
\end{aligned}
$$

**Full Expression**:

$$
\mathcal{F}^{\text{gated}} = \text{Concat}\big(\text{Attn}_1^{\text{gated}}, \ldots, \text{Attn}_H^{\text{gated}}\big) \mathbf{W}_O
$$

---

### 4.2 Deformable Cross-Modal Attention

#### 4.2.1 Motivation

Geometric gating operates on **global** attention patterns. For **fine-grained spatial alignment**, we introduce **deformable sampling** inspired by Deformable DETR [Zhu et al., ICLR 2021].

#### 4.2.2 Offset Prediction Network

Given query features $\mathbf{Q} \in \mathbb{R}^{B \times N_q \times D}$, predict sampling offsets:

$$
\Delta \mathbf{p} = \tanh\big(\text{MLP}_{\text{offset}}(\mathbf{Q})\big) \cdot \sigma_{\text{scale}} \in \mathbb{R}^{B \times N_q \times H \times K \times 2}
$$

where:
- $H = 8$: Number of attention heads
- $K = 4$: Number of sampling points per head
- $\sigma_{\text{scale}} = 1.0$: Offset scale factor
- $\tanh(\cdot)$: Bounds offsets to $[-\sigma_{\text{scale}}, \sigma_{\text{scale}}]$

**Initialization**: $\text{MLP}_{\text{offset}}$ weights initialized to zero → **identity initialization** (no offset initially).

#### 4.2.3 Sampling Location Computation

1. **Reference points** (query grid centers in normalized coordinates $[-1, 1]$):

$$
\mathbf{p}_{\text{ref}} \in \mathbb{R}^{N_q \times 2}, \quad \mathbf{p}_{\text{ref}}^{(i)} = \left(\frac{2x_i}{W_q} - 1, \frac{2y_i}{H_q} - 1\right)
$$

2. **Absolute sampling locations**:

$$
\mathbf{p}_{\text{sample}} = \mathbf{p}_{\text{ref}} + \Delta \mathbf{p} \in \mathbb{R}^{B \times N_q \times H \times K \times 2}
$$

Clamped to valid range: $\mathbf{p}_{\text{sample}} \in [-1, 1]^2$

#### 4.2.4 Deformable Attention Mechanism

For each head $h$ and query location $i$:

1. **Sample key/value features** at offset locations using bilinear interpolation:

$$
\mathbf{K}_{h,i,k} = \mathcal{S}\big(\mathcal{F}^{\text{sonar}}_h, \mathbf{p}_{\text{sample}}^{(h,i,k)}\big), \quad k = 1, \ldots, K
$$

2. **Compute attention scores** over sampled points:

$$
\alpha_{h,i,k} = \frac{\exp\big(\mathbf{Q}_{h,i}^\top \mathbf{K}_{h,i,k} / \sqrt{d_k}\big)}{\sum_{k'=1}^{K} \exp\big(\mathbf{Q}_{h,i}^\top \mathbf{K}_{h,i,k'} / \sqrt{d_k}\big)}
$$

3. **Weighted aggregation**:

$$
\text{Out}_{h,i} = \sum_{k=1}^{K} \alpha_{h,i,k} \cdot \mathbf{V}_{h,i,k}
$$

**Full Formulation**:

$$
\mathcal{F}^{\text{deform}} = \text{Concat}\big(\text{Out}_1, \ldots, \text{Out}_H\big) \mathbf{W}_O
$$

#### 4.2.5 Multi-Scale Deformable Attention

For feature pyramid $\{\mathcal{P}_0, \mathcal{P}_1, \ldots, \mathcal{P}_{L-1}\}$:

$$
\mathcal{F}^{\text{multi}} = \sum_{\ell=0}^{L-1} w_{\ell} \cdot \text{Deform-Attn}(\mathbf{Q}, \mathcal{P}_{\ell}^{\text{key}}, \mathcal{P}_{\ell}^{\text{value}})
$$

where $w_{\ell} = \text{softmax}(\mathbf{w})_{\ell}$ are learnable level weights.

---

### 4.3 Dual Consistency Loss

Dual consistency enforces alignment at two levels:

#### 4.3.1 Component Consistency

**Objective**: RGB and Sonar branches should predict **consistent** 3D components (e.g., $X$ coordinate).

Let $\hat{X}^{\text{rgb}}, \hat{X}^{\text{sonar}} \in \mathbb{R}^B$ be independently predicted $X$ coordinates.

**Component Consistency Loss**:

$$
\mathcal{L}_{\text{comp}} = \|\hat{X}^{\text{rgb}} - \hat{X}^{\text{sonar}}\|_1
$$

Alternative loss functions:
- L2: $\|\hat{X}^{\text{rgb}} - \hat{X}^{\text{sonar}}\|_2^2$
- Smooth L1: $\text{SmoothL1}(\hat{X}^{\text{rgb}}, \hat{X}^{\text{sonar}})$

#### 4.3.2 Representation Consistency (InfoNCE)

**Objective**: Learn modality-invariant representations using contrastive learning.

**Projection Heads**: Map features to embedding space:

$$
\begin{aligned}
\mathbf{e}^{\text{rgb}} &= \text{Proj}_{\text{rgb}}(\mathcal{F}^{\text{rgb}}) \in \mathbb{R}^{B \times D_e} \\
\mathbf{e}^{\text{sonar}} &= \text{Proj}_{\text{sonar}}(\mathcal{F}^{\text{sonar}}) \in \mathbb{R}^{B \times D_e}
\end{aligned}
$$

where $\text{Proj}$ are 2-layer MLPs.

**InfoNCE Loss** (Noise Contrastive Estimation):

$$
\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp\big(\text{sim}(\mathbf{e}_i^{\text{rgb}}, \mathbf{e}_i^{\text{sonar}}) / \tau\big)}{\sum_{j=1}^{B} \exp\big(\text{sim}(\mathbf{e}_i^{\text{rgb}}, \mathbf{e}_j^{\text{sonar}}) / \tau\big)}
$$

where:
- $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$: Cosine similarity
- $\tau = 0.07$: Temperature parameter
  - **Interpretation**: Controls concentration of similarity distribution
  - Smaller $\tau$ (e.g., 0.01): Sharper distribution, emphasizes hard negatives
  - Larger $\tau$ (e.g., 0.5): Softer distribution, easier optimization
  - Default value $\tau = 0.07$ balances hard negative mining and stable optimization
- **Positive pairs**: $(\mathbf{e}_i^{\text{rgb}}, \mathbf{e}_i^{\text{sonar}})$ from same frame
- **Negative pairs**: $(\mathbf{e}_i^{\text{rgb}}, \mathbf{e}_j^{\text{sonar}})$ from different frames ($i \neq j$)

**Symmetric Loss** (bidirectional):

$$
\mathcal{L}_{\text{InfoNCE}}^{\text{sym}} = \frac{1}{2}\big(\mathcal{L}_{\text{InfoNCE}}^{\text{rgb} \rightarrow \text{sonar}} + \mathcal{L}_{\text{InfoNCE}}^{\text{sonar} \rightarrow \text{rgb}}\big)
$$

#### 4.3.3 Combined Dual Consistency Loss

$$
\mathcal{L}_{\text{dual}} = \lambda_{\text{comp}} \mathcal{L}_{\text{comp}} + \lambda_{\text{repr}} \mathcal{L}_{\text{InfoNCE}}^{\text{sym}}
$$

Default weights: $\lambda_{\text{comp}} = 0.2$, $\lambda_{\text{repr}} = 0.1$

---

## 5. Vision-Mamba Temporal Modeling

### 5.1 State Space Models (SSM)

**Motivation**: Transformer-based temporal models have $O(T^2)$ complexity. For long sequences (hundreds of frames), this is prohibitive. Vision-Mamba uses **Selective State Space Models** with $O(T)$ complexity.

### 5.2 Continuous-Time State Space Formulation

**Continuous SSM**:

$$
\begin{aligned}
\frac{d\mathbf{h}(t)}{dt} &= \mathbf{A} \mathbf{h}(t) + \mathbf{B} \mathbf{x}(t) \\
\mathbf{y}(t) &= \mathbf{C} \mathbf{h}(t) + \mathbf{D} \mathbf{x}(t)
\end{aligned}
$$

where:
- $\mathbf{h}(t) \in \mathbb{R}^{N}$: Latent state (hidden representation)
- $\mathbf{x}(t) \in \mathbb{R}^{D}$: Input signal
- $\mathbf{y}(t) \in \mathbb{R}^{D}$: Output signal
- $\mathbf{A} \in \mathbb{R}^{N \times N}$: State transition matrix
- $\mathbf{B} \in \mathbb{R}^{N \times D}$: Input projection
- $\mathbf{C} \in \mathbb{R}^{D \times N}$: Output projection
- $\mathbf{D} \in \mathbb{R}^{D \times D}$: Direct feedthrough (often identity or zero)

### 5.3 Discretization (Zero-Order Hold)

For discrete time $t_k = k\Delta$:

$$
\begin{aligned}
\mathbf{h}_k &= \overline{\mathbf{A}} \mathbf{h}_{k-1} + \overline{\mathbf{B}} \mathbf{x}_k \\
\mathbf{y}_k &= \mathbf{C} \mathbf{h}_k + \mathbf{D} \mathbf{x}_k
\end{aligned}
$$

where:

$$
\overline{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \overline{\mathbf{B}} = (\overline{\mathbf{A}} - \mathbf{I}) \mathbf{A}^{-1} \mathbf{B}
$$

**Note**: For small $\Delta$, the approximation $\overline{\mathbf{B}} \approx \Delta \mathbf{B}$ is often used in practice.

**Reference**: Gu & Dao (2024), Mamba, Appendix A.2

### 5.4 Selective SSM (Mamba)

**Key Innovation**: Make $\mathbf{B}$, $\mathbf{C}$, and $\Delta$ **input-dependent**:

$$
\begin{aligned}
\mathbf{B}_k &= \text{Linear}_B(\mathbf{x}_k) \\
\mathbf{C}_k &= \text{Linear}_C(\mathbf{x}_k) \\
\Delta_k &= \text{softplus}\big(\text{Linear}_{\Delta}(\mathbf{x}_k)\big)
\end{aligned}
$$

This enables **selective** state updates: important information is retained, irrelevant information is forgotten.

**Key Parameters**:
- State dimension: $d_{\text{state}} = 16$ (SSM state expansion factor)
- Convolution width: $d_{\text{conv}} = 4$ (local convolution kernel size)
- Expansion factor: $\text{expand} = 2$ (block expansion ratio)

### 5.5 Vision-Mamba Block

**Input**: Spatial feature sequence $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_T] \in \mathbb{R}^{T \times D}$

**Forward Pass**:

1. **Pre-normalization**:

$$
\tilde{\mathbf{X}} = \text{LayerNorm}(\mathbf{X})
$$

2. **Selective SSM** (as described above):

$$
\mathbf{H} = \text{SSM}(\tilde{\mathbf{X}}; \mathbf{A}, \mathbf{B}, \mathbf{C}, \Delta)
$$

3. **Residual connection**:

$$
\mathbf{X}^{\text{out}} = \mathbf{X} + \text{Dropout}(\mathbf{H})
$$

### 5.6 Bidirectional Processing

For tracking, both **past and future** context are useful:

$$
\begin{aligned}
\mathbf{H}^{\text{fwd}} &= \text{Mamba}^{\text{fwd}}(\mathbf{X}) \\
\mathbf{H}^{\text{bwd}} &= \text{Mamba}^{\text{bwd}}(\text{flip}(\mathbf{X}))
\end{aligned}
$$

**Fusion**:

$$
\mathbf{H}^{\text{bidir}} = \text{Linear}([\mathbf{H}^{\text{fwd}}; \mathbf{H}^{\text{bwd}}])
$$

### 5.7 Occlusion-Aware Gating

When target is occluded or modality quality is low, **gate** the temporal update:

$$
\begin{aligned}
q_t &= \sigma\big(\text{MLP}_{\text{quality}}(f_t^{\text{quality}})\big) \quad \text{or} \quad \text{predictor}(\mathcal{F}_t) \\
\mathbf{H}_t^{\text{gated}} &= \mathbb{1}[q_t > \tau_{\text{gate}}] \cdot \mathbf{H}_t
\end{aligned}
$$

where $\tau_{\text{gate}}$ is a learnable threshold.

### 5.8 Exponential Moving Average (EMA) Aggregation

**Motivation**: For multi-frame sequences ($T > 1$), aggregate temporal features with learnable momentum.

**EMA Formulation**: Given temporal states $\{\mathbf{h}_1, \ldots, \mathbf{h}_T\}$:

$$
\mathbf{m}_t = \alpha_t \odot \mathbf{m}_{t-1} + (1 - \alpha_t) \odot \mathbf{h}_t
$$

where:
- $\mathbf{m}_t$: EMA state at time $t$
- $\alpha_t \in [0, 1]^D$: **Learnable momentum** (per-channel)
- $\odot$: Element-wise multiplication

**Learnable Momentum**:

$$
\alpha_t = \sigma\big(\text{MLP}_{\text{momentum}}([\mathbf{h}_t; \mathbf{m}_{t-1}])\big)
$$

This allows the model to **adaptively control** temporal aggregation based on:
- Current frame quality/reliability ($\mathbf{h}_t$)
- Historical context ($\mathbf{m}_{t-1}$)

**Initialization**: $\mathbf{m}_0 = \mathbf{0}$ or $\mathbf{m}_0 = \mathbf{h}_1$

**Multi-Frame Training Configuration**:
- Sequence length: $T = 5$ frames (stage-3 training)
- Frame sampling: Uniform with temporal stride
- EMA momentum range: $\alpha \in [0.9, 0.999]$ (typical after training)

---

## 6. Weak 3D Center Estimation

### 6.1 Problem Formulation

**Objective**: Estimate 3D center $\mathbf{C}_t^{3D} = (X_t, Y_t, Z_t)$ **without** extrinsic calibration between RGB and Sonar.

**Strategy**:
- **$Z$ (Depth)**: From sonar radial activation (sonar provides range information)
- **$X$ (Horizontal)**: Dual-branch regression (RGB + Sonar) with consistency
- **$Y$ (Vertical)**: RGB-dominant (better vertical resolution)

### 6.2 Depth Estimation from Sonar Radial Activation

#### 6.2.1 Radial Activation Extraction

Given sonar feature map $\mathcal{F}^{\text{sonar}}$ and radial profile $\mathbf{r} = [r_0, r_1, \ldots, r_{N-1}] \in \mathbb{R}^{N}$ (see Section 4.1.1):

#### 6.2.2 Learnable Weighted Integral

$$
Z_{\text{weighted}} = \sum_{k=0}^{N-1} w_k \cdot r_k
$$

where $\mathbf{w} = \text{softmax}([\tilde{w}_0, \ldots, \tilde{w}_{N-1}])$ are learnable attention weights.

#### 6.2.3 MLP Refinement

$$
Z_{\text{mlp}} = \text{MLP}_{\text{depth}}(\mathbf{r})
$$

#### 6.2.4 Combined Depth Estimate

**Residual formulation**:

$$
Z = Z_{\text{mlp}} + Z_{\text{weighted}}
$$

where the weighted integral term acts as a residual correction to the MLP-predicted depth, improving estimation stability.

### 6.3 Horizontal Coordinate (X) with Dual-Branch Consistency

#### 6.3.1 Independent Predictions

$$
\begin{aligned}
X^{\text{rgb}} &= \text{MLP}_X^{\text{rgb}}(\mathcal{F}^{\text{rgb}}) \\
X^{\text{sonar}} &= \text{MLP}_X^{\text{sonar}}(\mathcal{F}^{\text{sonar}})
\end{aligned}
$$

#### 6.3.2 Adaptive Fusion

Compute fusion weights based on both modalities:

$$
\begin{aligned}
[\alpha_{\text{rgb}}, \alpha_{\text{sonar}}] &= \text{softmax}\big(\text{MLP}_{\text{gate}}([\mathcal{F}^{\text{rgb}}; \mathcal{F}^{\text{sonar}}])\big) \\
X_{\text{fused}} &= \alpha_{\text{rgb}} \cdot X^{\text{rgb}} + \alpha_{\text{sonar}} \cdot X^{\text{sonar}}
\end{aligned}
$$

#### 6.3.3 Consistency Loss

$$
\mathcal{L}_X^{\text{cons}} = \|X^{\text{rgb}} - X^{\text{sonar}}\|_1
$$

### 6.4 Vertical Coordinate (Y)

RGB provides better vertical resolution:

$$
Y = \text{MLP}_Y^{\text{rgb}}(\mathcal{F}^{\text{rgb}})
$$

### 6.5 Complete 3D Center

$$
\mathbf{C}_t^{3D} = [X_{\text{fused}}, Y, Z]^\top \in \mathbb{R}^3
$$


---

## 7. Geometric Cyclic Projection

### 7.1 Motivation

The weak 3D center $\mathbf{C}^{3D}$ should be **geometrically consistent** with observed 2D bounding boxes in both modalities. We enforce this via **cyclic consistency**: $3D \rightarrow 2D \rightarrow 3D$.

### 7.2 Learnable 3D→2D Projection

#### 7.2.1 Perspective Projection (RGB)

**Learnable intrinsic parameters**:

$$
\begin{aligned}
f_x, f_y &: \text{Focal lengths (learnable, initialized to 1.0)} \\
c_x^{\text{off}}, c_y^{\text{off}} &: \text{Principal point offsets (learnable, initialized to 0.0)} \\
s_x, s_y &: \text{Size scale factors (learnable, initialized to 1.0)}
\end{aligned}
$$

**Projection**:

$$
\begin{aligned}
Z_{\text{safe}} &= \max(Z, \epsilon) \quad \text{(avoid division by zero, } \epsilon = 0.1 \text{)} \\
\hat{c}_x &= f_x \cdot \frac{X}{Z_{\text{safe}}} + c_x^{\text{off}} \\
\hat{c}_y &= f_y \cdot \frac{Y}{Z_{\text{safe}}} + c_y^{\text{off}} \\
\hat{w} &= \frac{s_x}{Z_{\text{safe}}}, \quad \hat{h} = \frac{s_y}{Z_{\text{safe}}}
\end{aligned}
$$

Thus:

$$
\hat{B}^{\text{rgb}} = (\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h})
$$

**Note**: This is a **weak perspective projection** without explicit 3D object size. The principal point baseline $(c_x^0, c_y^0)$ is absorbed into the learnable offsets.

#### 7.2.2 MLP Projection (Sonar)

Sonar geometry is non-standard (fan-shaped), use flexible MLP:

$$
\hat{B}^{\text{sonar}} = \text{MLP}_{\text{proj}}^{\text{sonar}}(\mathbf{C}^{3D})
$$

where MLP ensures positive width/height via softplus:

$$
\hat{w}, \hat{h} = \text{softplus}(\text{MLP}(\cdot))
$$

#### 7.2.3 Inverse Projection (2D → 3D)

For cyclic consistency, we need inverse projection $\hat{B}^{m} \rightarrow \tilde{\mathbf{C}}^{3D}$.

**Perspective Inverse** (RGB):

Given $\hat{B}^{\text{rgb}} = (\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h})$:

$$
\begin{aligned}
\tilde{Z} &= \sqrt{\frac{s_x \cdot s_y}{\hat{w} \cdot \hat{h}}} \quad \text{(geometric mean of size scales)} \\
\tilde{X} &= \frac{(\hat{c}_x - c_x^{\text{off}}) \cdot \tilde{Z}}{f_x} \\
\tilde{Y} &= \frac{(\hat{c}_y - c_y^{\text{off}}) \cdot \tilde{Z}}{f_y}
\end{aligned}
$$

**Note**: This inverse is **ill-posed** due to depth-scale ambiguity. The cyclic loss constrains the learned projection to be approximately invertible.

**MLP Inverse** (Sonar):

$$
\tilde{\mathbf{C}}^{3D} = \text{MLP}_{\text{backward}}^{\text{sonar}}(\hat{B}^{\text{sonar}})
$$

### 7.3 Cyclic Consistency Loss

#### 7.3.1 Forward-Backward Consistency

$$
\begin{aligned}
\hat{B}^{m} &= \text{Proj}^{m}(\mathbf{C}^{3D}), \quad m \in \{\text{rgb}, \text{sonar}\} \\
\tilde{\mathbf{C}}^{3D} &= \text{Proj}^{m, -1}(\hat{B}^{m}) \quad \text{(inverse projection)}
\end{aligned}
$$

**Cyclic Loss**:

$$
\mathcal{L}_{\text{cycle}}^{3D} = \|\tilde{\mathbf{C}}^{3D} - \mathbf{C}^{3D}\|_1
$$

#### 7.3.2 Projection-to-Groundtruth Loss

If 2D ground truth $B_{\text{gt}}^{m}$ is available:

$$
\mathcal{L}_{\text{proj}}^{m} = \|\hat{B}^{m} - B_{\text{gt}}^{m}\|_1
$$

Total cyclic loss:

$$
\mathcal{L}_{\text{cycle}} = \mathcal{L}_{\text{cycle}}^{3D} + \lambda_{\text{proj}} \big(\mathcal{L}_{\text{proj}}^{\text{rgb}} + \mathcal{L}_{\text{proj}}^{\text{sonar}}\big)
$$

---

## 8. Complete Loss Function and Optimization

### 8.1 Multi-Task Loss Formulation

The complete training objective combines all loss components:

$$
\mathcal{L}_{\text{total}} = \sum_{m \in \{\text{rgb}, \text{sonar}, \text{fusion}\}} \mathcal{L}_{\text{box}}^{m} + \lambda_{\text{dual}} \mathcal{L}_{\text{dual}} + \lambda_{\text{cycle}} \mathcal{L}_{\text{cycle}} + \lambda_{\text{temp}} \mathcal{L}_{\text{temp}}
$$

### 8.2 Bounding Box Losses

For each modality $m \in \{\text{rgb}, \text{sonar}, \text{fusion}\}$:

$$
\mathcal{L}_{\text{box}}^{m} = \lambda_{\text{giou}} \mathcal{L}_{\text{GIoU}}^{m} + \lambda_{L1} \mathcal{L}_{L1}^{m} + \lambda_{\text{qfl}} \mathcal{L}_{\text{QFL}}^{m}
$$

#### 8.2.1 Generalized IoU Loss

$$
\text{GIoU}(B, \hat{B}) = \text{IoU}(B, \hat{B}) - \frac{|\mathcal{C} \setminus (B \cup \hat{B})|}{|\mathcal{C}|}
$$

where $\mathcal{C}$ is the smallest enclosing box containing both $B$ and $\hat{B}$.

$$
\mathcal{L}_{\text{GIoU}}^m = 1 - \text{GIoU}(B_{\text{gt}}^m, \hat{B}^m)
$$

#### 8.2.2 L1 Box Loss

$$
\mathcal{L}_{L1}^m = \|B_{\text{gt}}^m - \hat{B}^m\|_1
$$

#### 8.2.3 Quality Focal Loss

For objectness score $\hat{q} \in [0, 1]$:

$$
\mathcal{L}_{\text{QFL}} = -\big|q - \hat{q}\big|^{\beta} \big(q \log(\hat{q}) + (1-q) \log(1-\hat{q})\big)
$$

where $q = \text{IoU}(B_{\text{gt}}, \hat{B})$ and $\beta = 2.0$.

### 8.3 Temporal Smoothness Loss

Encourage smooth trajectories:

$$
\mathcal{L}_{\text{temp}} = \sum_{t=1}^{T-1} \|\hat{B}_{t+1} - \hat{B}_t\|_2^2
$$

### 8.4 Loss Weights

**Default Configuration**:

| Component | Symbol | Default Weight |
|:----------|:-------|:--------------:|
| RGB Classification | $\lambda_{\text{cls}}^{\text{rgb}}$ | 1.0 |
| RGB Box Loss | $\lambda_{\text{box}}^{\text{rgb}}$ | 1.0 |
| Sonar Classification | $\lambda_{\text{cls}}^{\text{sonar}}$ | 1.0 |
| Sonar Box Loss | $\lambda_{\text{box}}^{\text{sonar}}$ | 1.0 |
| Fusion Classification | $\lambda_{\text{cls}}^{\text{fusion}}$ | 1.0 |
| Fusion Box Loss | $\lambda_{\text{box}}^{\text{fusion}}$ | 1.0 |
| Component Consistency | $\lambda_{\text{comp}}$ | 0.2 |
| Representation Consistency | $\lambda_{\text{repr}}$ | 0.1 |
| Weak 3D Cyclic | $\lambda_{\text{cycle}}$ | 0.6 |
| Temporal Smoothness | $\lambda_{\text{temp}}$ | 0.1 |

**Dynamic Weighting**: Loss weights are **automatically adjusted** during training based on task convergence rates (optional).

### 8.5 Optimization

#### 8.5.1 Optimizer

**AdamW** with decoupled weight decay:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_{\theta} \mathcal{L} \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_{\theta} \mathcal{L})^2 \\
\theta_t &= \theta_{t-1} - \eta_t \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon} - \lambda_{\text{wd}} \eta_t \theta_{t-1}
\end{aligned}
$$

**Hyperparameters**:
- $\beta_1 = 0.9, \beta_2 = 0.999$
- Base learning rate: $\eta_0 = 2 \times 10^{-4}$
- Weight decay: $\lambda_{\text{wd}} = 10^{-4}$
- Epsilon: $\epsilon = 10^{-8}$

#### 8.5.2 Learning Rate Schedule

**Cosine Annealing**:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}} \pi\right)\right)
$$

where $\eta_{\min} = 10^{-6}$ and $T_{\max}$ is total epochs.

**Layer-wise Learning Rate**:
- Pretrained backbones: $0.1 \times \eta_0$
- Fusion modules (G²-DCMA): $1.0 \times \eta_0$
- Temporal head (Mamba): $1.0 \times \eta_0$
- Prediction heads: $2.0 \times \eta_0$

#### 8.5.3 Gradient Clipping

$$
\text{if } \|\nabla \mathcal{L}\|_2 > \text{clip\_norm}: \quad \nabla \mathcal{L} \leftarrow \frac{\text{clip\_norm}}{\|\nabla \mathcal{L}\|_2} \nabla \mathcal{L}
$$

Default: $\text{clip\_norm} = 1.0$

#### 8.5.4 Mixed Precision Training

Use FP16 automatic mixed precision (AMP) for memory efficiency and speed:

$$
\mathcal{L}_{\text{scaled}} = \text{scale} \times \mathcal{L}_{\text{total}}
$$

Gradients are scaled, then unscaled before optimizer step.

### 8.6 Validity Masking Mechanism

**Problem**: Invalid samples (occlusions, augmentation failures) contaminate gradients (~3-5% noise).

**Solution**: Per-sample validity masking in loss computation.

#### 8.6.1 Validity-Aware Loss Computation

Given batch validity flags $\mathbf{V} = [V_1, \ldots, V_B] \in \{0, 1\}^B$:

**Masked Loss**:

$$
\mathcal{L}_{\text{masked}} = \frac{1}{\sum_{i=1}^B V_i + \epsilon} \sum_{i=1}^B V_i \cdot \mathcal{L}_i
$$

where:
- $\mathcal{L}_i$: Loss for sample $i$
- $V_i$: Validity flag (1 if valid, 0 if invalid)
- $\epsilon = 10^{-6}$: Prevents division by zero

**Gradient Flow**:

$$
\frac{\partial \mathcal{L}_{\text{masked}}}{\partial \theta} = \frac{1}{\sum_{i=1}^B V_i} \sum_{i=1}^B V_i \cdot \frac{\partial \mathcal{L}_i}{\partial \theta}
$$

Invalid samples ($V_i = 0$) contribute **zero gradient**, eliminating contamination.

#### 8.6.2 Multi-Modal Validity Handling

For RGB-Sonar fusion with independent validity:

$$
V_i^{\text{fusion}} = V_i^{\text{rgb}} \cdot V_i^{\text{sonar}}
$$

Component losses use modality-specific validity:

$$
\begin{aligned}
\mathcal{L}_{\text{rgb}} &= \frac{1}{\sum V_i^{\text{rgb}}} \sum V_i^{\text{rgb}} \cdot \mathcal{L}_i^{\text{rgb}} \\
\mathcal{L}_{\text{sonar}} &= \frac{1}{\sum V_i^{\text{sonar}}} \sum V_i^{\text{sonar}} \cdot \mathcal{L}_i^{\text{sonar}} \\
\mathcal{L}_{\text{fusion}} &= \frac{1}{\sum V_i^{\text{fusion}}} \sum V_i^{\text{fusion}} \cdot \mathcal{L}_i^{\text{fusion}}
\end{aligned}
$$

#### 8.6.3 Dynamic Validity Statistics

Track validity rates during training:

$$
\text{Validity Rate} = \frac{1}{B} \sum_{i=1}^B V_i
$$

Typical progression:
- Epoch 1-10: 100% (no augmentation)
- Epoch 11-40: ~98.4% (mild augmentation)
- Epoch 41-60: ~95% (strong augmentation)

**Expected Impact**:
- 12.5% faster convergence
- 2.3% accuracy improvement
- Eliminates gradient contamination

### 8.7 Training Strategy

#### 8.7.1 Three-Stage Training

**Stage 1: Single-Modality Pretraining** (40 epochs each)
- RGB-only: Standard tracking loss formulation
- Sonar-only: With rectangular coordinate preprocessing

**Stage 2: Multi-Modal Joint Training** (60 epochs)
- Initialize from pretrained backbones
- Train full G²-DCMA + Mamba + Weak3D pipeline
- Freeze lower backbone layers (optional)

**Stage 3: Multi-Frame Temporal Training** (20 epochs)
- Sequence length: $T = 5$ frames
- Enable EMA aggregation (Section 5.8)
- Fine-tune temporal components

**Warmup Configuration**:
- Pretraining stages: 20% warmup ratio (8 epochs out of 40)
- Fine-tuning stage: 12-15% warmup ratio (5-6 epochs out of 40)
- Linear scaling rule: When batch size increases by factor $k$, scale warmup steps and learning rate proportionally

#### 8.7.2 Batch Size and Gradient Accumulation

**Effective Batch Size Computation**:

$$
B_{\text{eff}} = B_{\text{phys}} \times N_{\text{accum}} \times N_{\text{gpu}}
$$

where:
- $B_{\text{phys}}$: Physical batch size per GPU
- $N_{\text{accum}}$: Number of gradient accumulation steps
- $N_{\text{gpu}}$: Number of GPUs in distributed training

**Gradient Synchronization**: In multi-GPU training, gradients are averaged across all devices before the optimizer step to ensure consistency.

---

## 9. Evaluation Metrics

### 9.1 Success Rate

**Success Curve**: Fraction of frames where IoU exceeds threshold $\theta$:

$$S(\theta) = \frac{1}{N} \sum_{t=1}^{N} \mathbb{1}[\text{IoU}_t > \theta]$$

**Area Under Curve (AUC)**:
$$\text{AUC} = \int_0^1 S(\theta) d\theta$$

### 9.2 Precision

**Center Error**: Distance between predicted and ground truth centers:
$$e_t = \sqrt{(c_x^{\text{pred}} - c_x^{\text{gt}})^2 + (c_y^{\text{pred}} - c_y^{\text{gt}})^2}$$

**Precision at Threshold** (default 20 pixels):
$$P_{20} = \frac{1}{N} \sum_{t=1}^{N} \mathbb{1}[e_t < 20]$$

### 9.3 Average Center Error (ACE)

$$\text{ACE} = \frac{1}{N} \sum_{t=1}^{N} e_t$$

### 9.4 3D Metrics (Optional)

**3D Center Error**:
$$\text{3D-CE} = \frac{1}{N} \sum_{t=1}^{N} \|\mathbf{C}_{3D,\text{pred}}^t - \mathbf{C}_{3D,\text{gt}}^t\|_2$$

---

## 10. References

### Core Publications

1. **OSTrack (Base Framework)**
   Ye, B., Chang, H., Ma, B., & Shan, S. (2022). *Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework*. ECCV 2022.
   [arXiv:2203.11991](https://arxiv.org/abs/2203.11991) | [Code](https://github.com/botaoye/OSTrack)

2. **RGBS50 (Task Benchmark)**
   Li, Y., et al. (2024). *RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker*.
   [arXiv:2406.07189](https://arxiv.org/abs/2406.07189) | [Code](https://github.com/LiYunfengLYF/RGBS50)

3. **Deformable DETR (Deformable Attention)**
   Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2021). *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR 2021.
   [arXiv:2010.04159](https://arxiv.org/abs/2010.04159) | [Code](https://github.com/fundamentalvision/Deformable-DETR)

4. **Mamba (State Space Models)**
   Gu, A., & Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. ICML 2024.
   [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

5. **Vision-Mamba**
   Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., & Wang, X. (2024). *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model*. ICML 2024.
   [Code](https://github.com/hustvl/Vim)

6. **SimCLR (InfoNCE Contrastive Learning)**
   Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.
   [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)

7. **ARTrackV2 (Autoregressive Tracking)**
   Bai, S., et al. (2024). *ARTrackV2: Prompting Autoregressive Tracker Where to Look and How to Describe*. CVPR 2024.
   [Code](https://github.com/MIV-XJTU/ARTrack)

8. **GIoU Loss**
   Rezatofighi, H., et al. (2019). *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*. CVPR 2019.
   [arXiv:1902.09630](https://arxiv.org/abs/1902.09630)

9. **Quality Focal Loss**
   Li, X., Wang, W., Wu, L., Chen, S., Hu, X., Li, J., Tang, J., & Yang, J. (2020). *Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection*. NeurIPS 2020.
   [arXiv:2006.15607](https://arxiv.org/abs/2006.15607)

### Additional References

- **AdamW Optimizer**: Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- **Multi-Head Attention**: Vaswani, A., et al. (2017). *Attention is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Zero-Order Hold Discretization**: Franklin, G. F., Powell, J. D., & Workman, M. L. (1998). *Digital Control of Dynamic Systems*. Addison-Wesley.
- **Bilinear Interpolation**: Keys, R. (1981). *Cubic convolution interpolation for digital image processing*. IEEE Transactions on Acoustics, Speech, and Signal Processing, 29(6), 1153-1160.
