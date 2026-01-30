# AquaSOT: RGB-Sonar 双模态跟踪数学公式与算法框架

**版本**: 2.2 (与当前实现完全同步)
**最后更新**: 2026-01-31
**基于代码**: `lib/models/aquasot/`, `lib/train/`, `lib/utils/`, `lib/test/evaluation/`

---

## 目录

1. [引言与问题定义](#1-引言与问题定义)
2. [模型架构](#2-模型架构)
3. [损失函数](#3-损失函数)
4. [数据处理流程](#4-数据处理流程)
5. [模板更新策略](#5-模板更新策略)
6. [训练配置](#6-训练配置)
7. [评估指标](#7-评估指标)
8. [参考文献](#8-参考文献)

---

## 1. 引言与问题定义

### 1.1 问题描述

AquaSOT 解决水下场景中的**免标定双模态单目标跟踪**问题，其中 RGB 相机和前视声呐 (FLS) 传感器具有以下特性：
- **时间对齐**但**空间未对齐**（无已知外参标定）
- 互补的失效模式（浑浊影响 RGB，声学噪声影响声呐）

系统通过端到端训练隐式学习跨模态对应关系。

**形式化定义**:
$$
\mathcal{T}: (I_t^{rgb}, I_t^{sonar}, B_0^{rgb}, B_0^{sonar}) \rightarrow (B_t^{rgb}, B_t^{sonar})
$$

其中 $t \in \{1, 2, ..., T\}$，$B = (c_x, c_y, w, h)$ 为归一化坐标格式的边界框。

### 1.2 输入规格

| 输入 | 形状 | 描述 |
|------|------|------|
| RGB模板 | $[B, 3, 128, 128]$ | 第一帧的RGB模板区域 |
| RGB搜索区域 | $[B, 3, 256, 256]$ | 当前帧RGB搜索区域 |
| Sonar模板 | $[B, 1, 128, 128]$ | 第一帧的声呐模板区域（灰度图） |
| Sonar搜索区域 | $[B, 1, 256, 256]$ | 当前帧声呐搜索区域（灰度图） |

### 1.3 输出规格

| 输出 | 形状 | 描述 |
|------|------|------|
| RGB边界框 | $[B, 4]$ | RGB模态预测框 $(c_x, c_y, w, h)$ 归一化 |
| Sonar边界框 | $[B, 4]$ | 声呐模态预测框 $(c_x, c_y, w, h)$ 归一化 |
| RGB置信度图 | $[B, 1, 16, 16]$ | RGB目标位置热力图 |
| Sonar置信度图 | $[B, 1, 16, 16]$ | 声呐目标位置热力图 |

### 1.4 坐标系统与边界框格式

**内部格式**: cxcywh (中心点坐标 + 宽高)
$$
B_{cxcywh} = (c_x, c_y, w, h)
$$

**转换公式**:
$$
\begin{aligned}
B_{xyxy} &= (c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2) \\
B_{xywh} &= (c_x - w/2, c_y - h/2, w, h)
\end{aligned}
$$

**代码参考**: `lib/utils/box_ops.py:14-52`

---

## 2. 模型架构

### 2.1 架构概览

```
RGB模板 [B, 3, 128, 128] + RGB搜索 [B, 3, 256, 256]
   --> ViT RGB骨干网络 --> RGB特征 [B, L, 768]
                                    |
                                    v
                              DCMA融合 (2层)
                                    |
Sonar模板 [B, 1, 128, 128] + Sonar搜索 [B, 1, 256, 256]
   --> ViT Sonar骨干网络 --> Sonar特征 [B, L, 768]
                                    |
                                    v
                    +---------------+---------------+
                    v                               v
              rgb_box_head                   sonar_box_head
                    |                               |
                    v                               v
            RGB边界框 [B, 4]                 Sonar边界框 [B, 4]
```

注意: AquaSOT 使用**独立预测头** (`rgb_box_head`, `sonar_box_head`) 以防止因模态视场不对称 (RGB视场~60-70度, Sonar视场~120-160度) 导致的预测漂移。

**特征长度**: $L = N_z + N_s = (128/16)^2 + (256/16)^2 = 64 + 256 = 320$ patches

**代码参考**: `lib/models/aquasot/aquasot.py:16-177`

### 2.2 Vision Transformer 骨干网络

双流架构中，RGB和Sonar各使用独立的ViT骨干网络。两者共享相同架构但参数独立。

**Patch嵌入**:
$$
x = \text{PatchEmbed}(I) + E_{pos}
$$

其中 $I \in \mathbb{R}^{B \times C \times H \times W}$，$C=3$ (RGB) 或 $C=1$ (Sonar)

参数定义:
- $N = (H/P) \times (W/P)$: Patch数量
- $P = 16$: Patch大小
- $D = 768$: 嵌入维度

模板 ($128 \times 128$): $N_z = 64$ patches
搜索区域 ($256 \times 256$): $N_s = 256$ patches

**组合序列长度**: $L = N_z + N_s = 64 + 256 = 320$ patches

**自注意力机制**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $d_k = D/H = 768/12 = 64$

**Transformer块**:
$$
\begin{aligned}
x' &= x + \text{DropPath}(\text{Attn}(\text{LayerNorm}(x))) \\
x'' &= x' + \text{DropPath}(\text{MLP}(\text{LayerNorm}(x')))
\end{aligned}
$$

**ViT参数**:

| 参数 | 值 | 代码来源 |
|------|-----|---------|
| 嵌入维度 | 768 | `cfg.MODEL.BACKBONE.RGB.EMBED_DIM` |
| 深度 | 12 | `cfg.MODEL.BACKBONE.RGB.DEPTH` |
| 注意力头数 | 12 | `cfg.MODEL.BACKBONE.RGB.NUM_HEADS` |
| MLP比率 | 4.0 | `cfg.MODEL.BACKBONE.RGB.MLP_RATIO` |
| Patch大小 | 16 | `cfg.MODEL.BACKBONE.RGB.STRIDE` |
| DropPath率 | 0.1 | `cfg.MODEL.BACKBONE.RGB.DROP_PATH_RATE` |

**代码参考**: `lib/models/aquasot/vit.py:30-165`

### 2.3 跨模态注意力融合 (DCMA)

DCMA (Dual-stream Cross-Modal Attention) 模块使用标准双向交叉注意力实现RGB与Sonar特征的融合。

**注意**: 尽管名称中包含"D"，当前实现使用的是标准交叉注意力，而非可变形注意力。

**交叉注意力**:
$$
\begin{aligned}
\mathbf{Q} &= \mathbf{W}_Q \cdot \mathbf{x}_{query} \\
\mathbf{K} &= \mathbf{W}_K \cdot \mathbf{x}_{kv} \\
\mathbf{V} &= \mathbf{W}_V \cdot \mathbf{x}_{kv} \\
\text{Attn} &= \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} \\
\text{out} &= \mathbf{W}_O \cdot \text{Attn}
\end{aligned}
$$

**DCMA层 (双向)**:
$$
\begin{aligned}
\mathbf{F}_{rgb}' &= \mathbf{F}_{rgb} + \text{CrossAttn}(\text{LN}(\mathbf{F}_{rgb}), \mathbf{F}_{sonar}) \\
\mathbf{F}_{sonar}' &= \mathbf{F}_{sonar} + \text{CrossAttn}(\text{LN}(\mathbf{F}_{sonar}), \mathbf{F}_{rgb}) \\
\mathbf{F}_{rgb}^{out} &= \mathbf{F}_{rgb}' + \text{FFN}(\text{LN}(\mathbf{F}_{rgb}')) \\
\mathbf{F}_{sonar}^{out} &= \mathbf{F}_{sonar}' + \text{FFN}(\text{LN}(\mathbf{F}_{sonar}'))
\end{aligned}
$$

**DCMA参数**:

| 参数 | 值 | 代码来源 |
|------|-----|---------|
| 层数 | 2 | `cfg.MODEL.FUSION.NUM_LAYERS` |
| 注意力头数 | 8 | `cfg.MODEL.FUSION.NUM_HEADS` |
| 嵌入维度 | 768 | `cfg.MODEL.FUSION.EMBED_DIM` |
| Dropout率 | 0.1 | `cfg.MODEL.FUSION.DROPOUT` |
| RGB融合权重(推理) | 0.3 | `cfg.MODEL.FUSION.RGB_FUSION_WEIGHT` |
| Sonar融合权重(推理) | 0.3 | `cfg.MODEL.FUSION.SONAR_FUSION_WEIGHT` |
| RGB融合权重(训练) | 1.0 | `cfg.MODEL.FUSION.RGB_FUSION_WEIGHT_TRAIN` |
| Sonar融合权重(训练) | 1.0 | `cfg.MODEL.FUSION.SONAR_FUSION_WEIGHT_TRAIN` |

**代码参考**: `lib/models/aquasot/fusion.py:19-167`

#### 2.3.3 非对称融合权重（训练 vs 推理）

**代码参考**: `lib/models/aquasot/fusion.py:222-250`

交叉注意力残差连接使用可配置权重控制融合强度:

$$
\begin{aligned}
\mathbf{F}_{rgb}' &= \mathbf{F}_{rgb} + \alpha_{rgb} \cdot \text{CrossAttn}(\text{LN}(\mathbf{F}_{rgb}), \mathbf{F}_{sonar}) \\
\mathbf{F}_{sonar}' &= \mathbf{F}_{sonar} + \alpha_{sonar} \cdot \text{CrossAttn}(\text{LN}(\mathbf{F}_{sonar}), \mathbf{F}_{rgb})
\end{aligned}
$$

**非对称设计**:
- **训练**: $\alpha_{rgb} = \alpha_{sonar} = 1.0$（完全融合，充分训练DCMA投影）
- **推理**: $\alpha_{rgb} = \alpha_{sonar} = 0.3$（弱融合，保留模态特定信息）
- $\alpha = 0$: 无跨模态影响（消融实验，通过 `--eval_mode no_fusion` 启用）

**设计原理**: 训练时使用完全融合使DCMA投影获得充分的梯度信号。推理时使用弱融合保留模态特定信息以提高鲁棒性。消融实验建议启用BN适应以补偿特征分布差异。

#### 2.3.4 模态有效性检测与自注意力回退

**代码参考**: `lib/models/aquasot/aquasot.py:73-74`, `lib/models/aquasot/fusion.py:125-145`

当通过模态丢弃将某个模态置零时，交叉注意力变得无效。模型检测此情况并使用自注意力作为回退。

**有效性检测**:

$$
\text{valid}_{m} = \mathbb{1}\left[\sum_{i,j,k}|\mathbf{I}_{m}[i,j,k]| > 10^{-4}\right]
$$

**自注意力回退**:

当Sonar被丢弃 ($\text{valid}_{sonar} = \text{False}$) 时:

$$
\mathbf{F}_{rgb}' = \mathbf{F}_{rgb} + \alpha_{rgb} \cdot \text{SelfAttn}_{rgb \to sonar}(\text{LN}(\mathbf{F}_{rgb}), \mathbf{F}_{rgb})
$$

当RGB被丢弃 ($\text{valid}_{rgb} = \text{False}$) 时:

$$
\mathbf{F}_{sonar}' = \mathbf{F}_{sonar} + \alpha_{sonar} \cdot \text{CrossAttn}_{rgb \to sonar}(\text{LN}(\mathbf{F}_{sonar}), \mathbf{F}_{sonar})
$$

**注意**: Sonar自注意力时使用`rgb_to_sonar`模块，因为其K/V投影是为Sonar特征训练的。使用`sonar_to_rgb`会将为RGB特征训练的K/V投影应用于Sonar特征，产生错误的输出分布。

### 2.4 CenterPredictor 预测头

预测头遵循CenterNet设计，包含三个并行分支。AquaSOT使用**独立预测头**:
- `rgb_box_head`: 学习RGB特有的预测模式 (运动更快，方差更大)
- `sonar_box_head`: 学习Sonar特有的预测模式 (运动更慢，方差更稳定)

这种分离至关重要，因为共享头会导致Sonar预测向RGB运动模式漂移（训练时梯度平均效应）。

**架构**:
$$
\begin{aligned}
\mathbf{M}_{score} &= \sigma(\text{Conv}_{score}(\mathbf{F}_{search})) \in \mathbb{R}^{B \times 1 \times 16 \times 16} \\
\mathbf{M}_{size} &= \sigma(\text{Conv}_{size}(\mathbf{F}_{search})) \in \mathbb{R}^{B \times 2 \times 16 \times 16} \\
\mathbf{M}_{offset} &= \text{Conv}_{offset}(\mathbf{F}_{search}) \in \mathbb{R}^{B \times 2 \times 16 \times 16}
\end{aligned}
$$

其中 $\sigma(\cdot)$ 表示Sigmoid激活函数，并限制在 $[10^{-4}, 1-10^{-4}]$ 范围内。

**边界框解码**:
$$
\begin{aligned}
(i_x, i_y) &= \text{argmax}_{x,y}(\mathbf{M}_{score}) \\
(o_x, o_y) &= \mathbf{M}_{offset}[:, :, i_y, i_x] \\
(w, h) &= \mathbf{M}_{size}[:, :, i_y, i_x] \\
\hat{B} &= \left(\frac{i_x + o_x}{f_s}, \frac{i_y + o_y}{f_s}, w, h\right)
\end{aligned}
$$

其中 $f_s = 16$ 为特征步长。

**代码参考**: `lib/models/layers/head.py:31-165`

---

## 3. 损失函数

### 3.1 Focal Loss (分类损失)

Focal Loss解决正负样本不平衡问题，重点关注难例。

$$
\mathcal{L}_{focal} = \frac{1}{N_{pos}}\left(\sum_{pos} -(1-p)^\alpha \log(p) + \sum_{neg} -(1-y)^\beta p^\alpha \log(1-p)\right)
$$

其中:
- $p$ = 预测概率
- $y$ = 真值热力图
- $\alpha = 2$ (聚焦参数)
- $\beta = 4$ (负样本降权参数)
- $N_{pos}$ = 正样本数量

**代码参考**: `lib/utils/focal_loss.py:10-59`

### 3.2 GIoU Loss (定位损失)

广义IoU (GIoU) loss对非重叠边界框具有更好的梯度特性。

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

$$
\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
$$

$$
\mathcal{L}_{GIoU} = 1 - \text{GIoU}
$$

其中 $C$ 是包围 $A$ 和 $B$ 的最小闭合框。

**取值范围**: $\mathcal{L}_{GIoU} \in [0, 2]$

**代码参考**: `lib/utils/box_ops.py:93-136`

### 3.3 L1 Loss (坐标回归损失)

边界框坐标回归使用L1损失。

$$
\mathcal{L}_{L1} = \frac{1}{4}\sum_{i=1}^{4}\left|\hat{B}^{(i)} - B_{gt}^{(i)}\right|
$$

**代码参考**: `lib/utils/box_ops.py:139-149`

### 3.4 总损失

$$
\mathcal{L}_{total} = \lambda_{giou}(\mathcal{L}_{giou}^{rgb} + \mathcal{L}_{giou}^{sonar}) + \lambda_{l1}(\mathcal{L}_{l1}^{rgb} + \mathcal{L}_{l1}^{sonar}) + \lambda_{focal}(\mathcal{L}_{focal}^{rgb} + \mathcal{L}_{focal}^{sonar})
$$

**默认权重**:

| 损失项 | 权重 | 代码来源 |
|-------|------|---------|
| GIoU | $\lambda_{giou} = 2.0$ | `cfg.TRAIN.GIOU_WEIGHT` |
| L1 | $\lambda_{l1} = 5.0$ | `cfg.TRAIN.L1_WEIGHT` |
| Focal | $\lambda_{focal} = 1.0$ | `cfg.TRAIN.FOCAL_WEIGHT` |

**代码参考**: `lib/train/actors/aquasot.py:166-171`

---

## 4. 数据处理流程

### 4.1 模板与搜索区域裁剪

给定边界框 $B = (c_x, c_y, w, h)$，裁剪区域计算如下:

**裁剪尺寸计算**:

$$
s_{crop} = \left\lceil\sqrt{w \cdot h} \cdot f_{area}\right\rceil
$$

其中:
- $w, h$: 目标宽高
- $f_{area}$: 区域扩展因子 (模板: 2.0, 搜索区域: 4.0)

**裁剪参数**:

| 区域类型 | 尺寸 | Factor |
|---------|------|--------|
| 模板 | 128 | 2.0 |
| 搜索区域 | 256 | 4.0 |

**代码参考**: `lib/train/data/processing.py:38-120`

### 4.2 空间抖动同步

RGB和Sonar使用相同的抖动参数以保持空间一致性。

**尺度抖动**:

$$
(w', h') = (w, h) \cdot \exp(\sigma_{scale} \cdot \mathcal{N}(0, 1))
$$

**中心抖动**:

$$
\begin{aligned}
d_{max} &= \sqrt{w' \cdot h'} \cdot f_{center} \\
(c_x', c_y') &= (c_x, c_y) + d_{max} \cdot (\mathcal{U}(-0.5, 0.5), \mathcal{U}(-0.5, 0.5))
\end{aligned}
$$

**抖动参数**:

| 参数 | 模板 | 搜索区域 |
|------|------|---------|
| 中心抖动因子 ($f_{center}$) | 0.0 | 1.0 |
| 尺度抖动标准差 ($\sigma_{scale}$) | 0.0 | 0.25 |

其中:
- $(w, h)$: 原始边界框尺寸
- $(w', h')$: 抖动后的边界框尺寸

**代码参考**: `lib/train/data/processing.py:89-131`

### 4.3 数据增强

**模态独立增强**:

| 增强方法 | RGB | Sonar |
|---------|-----|-------|
| 水平翻转 | 是 (0.5) | 是 (同步) |
| 颜色抖动 | 是 | - |
| 灰度化 | 是 (0.05) | - |
| 高斯模糊 | 是 (0.3) | - |
| 散斑噪声 | - | 是 (0.5) |

**同步增强**: 水平翻转在两个模态间同步应用。

**代码参考**: `lib/train/data/processing.py:212-327`

### 4.4 高斯热力图生成

真值热力图使用自适应高斯生成。

**标准差计算**:

$$
\begin{aligned}
\sigma_x &= \max\left(\frac{w_{norm} \cdot f_s}{6.0}, 1.0\right) \\
\sigma_y &= \max\left(\frac{h_{norm} \cdot f_s}{6.0}, 1.0\right)
\end{aligned}
$$

其中:
- $w_{norm}, h_{norm} \in [0, 1]$: 相对于搜索区域的归一化目标宽高
- $f_s = 16$: 特征图尺寸 (search_size / stride = 256 / 16)

**高斯分布**:

$$
G(x, y) = \exp\left(-\frac{(x - c_x)^2}{2\sigma_x^2} - \frac{(y - c_y)^2}{2\sigma_y^2}\right)
$$

其中 $(c_x, c_y)$ 为特征图空间中的目标中心坐标。

**代码参考**: `lib/utils/heatmap_utils.py:30-76`

### 4.5 归一化

**代码参考**: `lib/config/aquasot/config.py:141-144`

| 模态 | 均值 | 标准差 | 来源 |
|------|------|--------|------|
| RGB | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | ImageNet |
| Sonar | [0.5] | [0.5] | `cfg.DATA.SONAR_*` |

### 4.6 模态丢弃（测试时消融）

**代码参考**: `lib/utils/modality_dropout.py:28-77`

消融实验可在测试时应用模态丢弃，以使用双模态训练的模型评估单模态性能:

$$
I_{dropped} = \begin{cases}
\mathbf{0} & \text{if } p_{drop} = 1.0 \\
I & \text{otherwise}
\end{cases}
$$

当模态被丢弃时:
1. 输入张量被替换为零张量
2. 模型对该模态产生低置信度分数 ($\approx 0$)
3. 目标丢失处理器检测到被丢弃的模态为LOST状态
4. 主输出完全使用活跃模态

**配置参数**:

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `ENABLED` | False | 启用模态丢弃 |
| `P_DROP_RGB` | 0.0 | 丢弃RGB的概率 (1.0 = 仅Sonar) |
| `P_DROP_SONAR` | 0.0 | 丢弃Sonar的概率 (1.0 = 仅RGB) |

**注意**: 训练始终使用双模态以学习鲁棒的双模态表示。丢弃仅在测试时用于消融研究。

### 4.7 推理时 BatchNorm 自适应

**代码参考**: `lib/utils/bn_adaptation.py:21-131`

双模态训练期间，BatchNorm层学习融合特征分布的运行统计量。单模态推理时，特征分布发生偏移（交叉注意力输出变为自注意力或零张量），导致分布不匹配，降低预测质量。

#### 4.7.1 自适应算法

自适应过程重新计算BN运行统计量：

1. **统计量重置**:

$$
\mu_{running} \leftarrow 0, \quad \sigma^2_{running} \leftarrow 1
$$

2. **前向传播自适应** (对于 $n = 1, ..., N_{adapt}$ 批次):

$$
\mu_{running} \leftarrow (1 - \gamma) \cdot \mu_{running} + \gamma \cdot \mu_{batch}
$$

$$
\sigma^2_{running} \leftarrow (1 - \gamma) \cdot \sigma^2_{running} + \gamma \cdot \sigma^2_{batch}
$$

其中:
- $\gamma = 0.2$: 自适应动量（高于训练默认值0.1）
- $N_{adapt} = 30$: 自适应批次数（可配置）

#### 4.7.2 模态特定输入生成

| 模式 | RGB输入 | Sonar输入 | 用途 |
|------|---------|-----------|------|
| `rgb` | 正常 | $\mathbf{0}$ | 仅RGB消融 |
| `sonar` | $\mathbf{0}$ | 正常 | 仅Sonar消融 |
| `dual` | 正常 | 正常 | 无融合消融 |

#### 4.7.3 配置参数

| 参数 | 默认值 | 代码来源 |
|------|--------|---------|
| `ENABLED` | False | `cfg.TEST.BN_ADAPTATION.ENABLED` |
| `NUM_BATCHES` | 30 | `cfg.TEST.BN_ADAPTATION.NUM_BATCHES` |
| `DATA_SOURCE` | 'test' | `cfg.TEST.BN_ADAPTATION.DATA_SOURCE` |

**相关工作**: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021 [arXiv:2006.10726]

---

## 5. 模板更新策略

### 5.1 MixFormer 三层模板策略

推理阶段采用三层模板管理策略：

**第一层: 首帧模板 (固定)**
- 始终保留第一帧模板
- 不更新以防止漂移

**第二层: 在线模板FIFO队列**
- 固定大小 $|Q| = 2$
- 保存最近的高置信度模板
- 按间隔更新 ($T_{interval} = 200$ 帧)

**第三层: 最佳帧缓存**
- 保存间隔内最高置信度模板
- 用于模板队列更新

### 5.2 实现细节

**更新算法**:
$$
\begin{aligned}
&\textbf{if } s_t > \tau_{score} \land s_t > s_{max}: \\
&\quad T_{best} \leftarrow \text{crop}(I_t, B_t) \\
&\quad s_{max} \leftarrow s_t \\
&\textbf{if } t \mod T_{interval} = 0: \\
&\quad \text{push}(Q_{online}, T_{best}) \\
&\quad s_{max} \leftarrow -1
\end{aligned}
$$

**更新参数**:

| 参数 | 值 | 代码来源 |
|------|-----|---------|
| 置信度阈值 | $\tau_{score} = 0.5$ | `cfg.TEST.TEMPLATE_UPDATE.SCORE_THRESHOLD` |
| 更新间隔 | $T_{interval} = 200$ | `cfg.TEST.TEMPLATE_UPDATE.UPDATE_INTERVAL` |
| 队列大小 | $|Q_{online}| = 2$ | `cfg.TEST.TEMPLATE_UPDATE.ONLINE_SIZE` |

**代码参考**: `lib/test/tracker/aquasot_tracker.py:296-370`

### 5.3 目标丢失检测与恢复

**代码参考**: `lib/test/tracker/target_loss_handler.py`

AquaSOT实现双模态独立状态机以实现鲁棒的目标跟踪。每个模态（RGB和Sonar）维护独立的跟踪状态，允许Sonar在目标离开RGB视场时继续跟踪。

#### 5.3.1 跟踪状态

| 状态 | 描述 | 输出行为 |
|------|------|---------|
| STABLE | 目标被可靠跟踪 | 使用预测，允许模板更新 |
| UNCERTAIN | 置信度较低 | 谨慎使用预测，禁止模板更新 |
| LOST | 目标丢失 | 保持最后稳定位置，禁止模板更新 |

#### 5.3.2 分数分类

置信度分数 $s$ 被分类为跟踪标志:

$$
f(s) = \begin{cases}
\texttt{NOT FOUND} & \text{if } s < \tau_{nf} \\
\texttt{UNCERTAIN} & \text{if } \tau_{nf} \leq s < \tau_{s} \\
\texttt{STABLE} & \text{if } s \geq \tau_{s}
\end{cases}
$$

其中:
- $\tau_{nf} = 0.01$: 未找到阈值 (`NOT_FOUND_THRESHOLD`)
- $\tau_{unc} = 0.05$: 不确定阈值 (`UNCERTAIN_THRESHOLD`)
- $\tau_{s} = 0.10$: 稳定阈值 (`STABLE_THRESHOLD`)

**注意**: 极低阈值 ($\tau_{nf}=0.01$, $\tau_{s}=0.10$) 意味着跟踪器除非置信度接近零，否则保持STABLE状态。此设计优先考虑持续跟踪而非过早的丢失检测。

#### 5.3.3 状态机转换

**丢失检测** (带滞后防止振荡):

$$
\text{state} = \begin{cases}
\text{UNCERTAIN} & \text{if } c_{lost} \geq 1 \text{ (从STABLE)} \\
\text{LOST} & \text{if } c_{lost} \geq N_{lost}
\end{cases}
$$

**恢复检测**:

$$
\text{state} = \begin{cases}
\text{STABLE} & \text{if } c_{found} \geq N_{quick} \text{ (从UNCERTAIN)} \\
\text{STABLE} & \text{if } c_{found} \geq N_{found} \text{ (从LOST)}
\end{cases}
$$

其中:
- $c_{lost}$: 连续`target_lost = True`的帧数
- $c_{found}$: 连续`target_lost = False`的帧数
- $N_{lost} = 2$: 确认LOST所需帧数 (`LOST_CONFIRM_FRAMES`)
- $N_{found} = 5$: 从LOST恢复所需帧数 (`FOUND_CONFIRM_FRAMES`)
- $N_{quick} = 2$: 从UNCERTAIN快速恢复所需帧数 (`QUICK_RECOVERY_FRAMES`)

#### 5.3.4 搜索区域验证

预测结果需在以前一位置为中心的搜索区域内验证:

$$
\mathit{valid} = \begin{cases}
\texttt{True} & \text{if } |c_x^{pred} - c_x^{prev}| \leq \frac{w \cdot K}{2} \land |c_y^{pred} - c_y^{prev}| \leq \frac{h \cdot K}{2} \\
\texttt{False} & \text{otherwise}
\end{cases}
$$

其中 $K = 5.0$ 为搜索区域倍率 (`SEARCH_REGION_K`)。

#### 5.3.5 主输出选择

当两个模态都处于STABLE状态时:

$$
B_{primary} = \begin{cases}
B_{rgb} & \text{if } \mathit{priority} = \texttt{rgb-first} \\
B_{sonar} & \text{if } \mathit{priority} = \texttt{sonar-first} \\
\arg\max_{m \in \{rgb, sonar\}}(s_m) & \text{if } \mathit{priority} = \texttt{higher-score}
\end{cases}
$$

当一个模态为LOST时，使用STABLE模态的输出。当两个模态都为LOST时，使用RGB最后稳定位置作为回退。

#### 5.3.6 选择性模板更新

模板仅在以下条件满足时更新:
1. 模态状态为STABLE
2. 置信度分数超过 `HIGH_CONFIDENCE_THRESHOLD` (0.7)

这可防止遮挡或目标丢失期间不可靠预测污染模板。

#### 5.3.7 配置参数

| 参数 | 值 | 代码来源 |
|------|-----|---------|
| `ENABLED` | True | `cfg.TEST.TARGET_LOSS.ENABLED` |
| $\tau_{nf}$ | 0.01 | `cfg.TEST.TARGET_LOSS.NOT_FOUND_THRESHOLD` |
| $\tau_{unc}$ | 0.05 | `cfg.TEST.TARGET_LOSS.UNCERTAIN_THRESHOLD` |
| $\tau_{s}$ | 0.10 | `cfg.TEST.TARGET_LOSS.STABLE_THRESHOLD` |
| $N_{lost}$ | 2 | `cfg.TEST.TARGET_LOSS.LOST_CONFIRM_FRAMES` |
| $N_{found}$ | 5 | `cfg.TEST.TARGET_LOSS.FOUND_CONFIRM_FRAMES` |
| $N_{quick}$ | 2 | `cfg.TEST.TARGET_LOSS.QUICK_RECOVERY_FRAMES` |
| $K$ | 5.0 | `cfg.TEST.TARGET_LOSS.SEARCH_REGION_K` |
| priority | higher_score | `cfg.TEST.TARGET_LOSS.PRIMARY_PRIORITY` |

#### 5.3.8 动态搜索因子

**代码参考**: `lib/config/aquasot/config.py:291-312`

搜索区域因子根据跟踪状态动态调整以改善目标恢复:

$$
f_{search}(state) = \begin{cases}
f_{stable} = 4.0 & \text{if } state = \text{STABLE} \\
f_{uncertain} = 5.0 & \text{if } state = \text{UNCERTAIN} \\
f_{lost} = 7.0 & \text{if } state = \text{LOST}
\end{cases}
$$

**配置参数**:

| 参数 | 值 | 代码来源 |
|------|-----|---------|
| `ENABLED` | True | `cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.ENABLED` |
| `STABLE` | 4.0 | `cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.STABLE` |
| `UNCERTAIN` | 5.0 | `cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.UNCERTAIN` |
| `LOST` | 7.0 | `cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.LOST` |

RGB和Sonar模态可通过 `RGB_STABLE`, `RGB_UNCERTAIN`, `RGB_LOST`, `SONAR_STABLE`, `SONAR_UNCERTAIN`, `SONAR_LOST` 设置独立的搜索因子。

---

## 6. 训练配置

### 6.1 优化器

使用带权重衰减的AdamW优化器，骨干网络采用差异化学习率。

$$
\eta_{backbone} = \eta_{base} \times \gamma_{backbone}
$$

其中 $\gamma_{backbone} = 0.1$ 为骨干网络学习率倍率。

**优化器参数**:

| 参数 | 值 |
|------|-----|
| 基础学习率 | $10^{-4}$ |
| 骨干网络倍率 | 0.1 |
| 权重衰减 | $10^{-4}$ |
| 梯度裁剪范数 | 0.1 |

### 6.2 学习率调度器

使用带预热的阶梯衰减策略。

$$
\text{lr}_t = \begin{cases}
\text{lr}_{warmup} + \frac{t}{T_{warmup}}(\text{lr}_{base} - \text{lr}_{warmup}) & t < T_{warmup} \\
\text{lr}_{base} \times \gamma^{\lfloor t / T_{step} \rfloor} & t \geq T_{warmup}
\end{cases}
$$

**调度器参数**:

| 参数 | 值 | 代码来源 |
|------|-----|---------|
| 预热轮数 | 5 | `cfg.TRAIN.SCHEDULER.WARMUP_EPOCHS` |
| 预热学习率 | $10^{-5}$ | `cfg.TRAIN.SCHEDULER.WARMUP_LR` |
| 学习率衰减轮数 | 80 | `cfg.TRAIN.LR_DROP_EPOCH` |
| 衰减因子 $\gamma$ | 0.1 | `cfg.TRAIN.SCHEDULER.GAMMA` |
| 最小学习率 | $10^{-6}$ | `cfg.TRAIN.SCHEDULER.MIN_LR` |

### 6.3 梯度裁剪

$$
\text{if } \|\nabla \mathcal{L}\|_2 > \tau_{clip}: \quad \nabla \mathcal{L} \leftarrow \frac{\tau_{clip}}{\|\nabla \mathcal{L}\|_2} \nabla \mathcal{L}
$$

| 参数 | 值 | 代码来源 |
|------|-----|---------|
| $\tau_{clip}$ | 0.1 | `cfg.TRAIN.GRAD_CLIP_NORM` |

### 6.4 训练超参数

| 参数 | 值 | 代码来源 |
|------|-----|---------|
| 总轮数 | 100 | `cfg.TRAIN.EPOCH` |
| 批大小 | 2 | `cfg.TRAIN.BATCH_SIZE` |
| 梯度累积步数 | 4 | `cfg.TRAIN.GRADIENT_ACCUMULATION` |
| 有效批大小 | 8 | 批大小 $\times$ 累积步数 |
| 每轮采样数 | 8000 | `cfg.DATA.TRAIN.SAMPLE_PER_EPOCH` |

**代码参考**: `lib/config/aquasot/config.py:78-131`

---

## 7. 评估指标

### 7.0 评估设置

**代码参考**: `lib/config/aquasot/config.py:314-318`, `lib/test/evaluation/experiment.py:23-57`

AquaSOT采用GOT-10k风格评估，支持可见性过滤和多模态评估。

#### 7.0.1 配置参数

| 参数 | 值 | 描述 |
|------|-----|------|
| `FILTER_VISIBILITY` | True | 过滤缺失帧 (GT=0,0,0,0) |
| `SKIP_INIT_FRAME` | True | 跳过第0帧（初始化帧） |
| `EVAL_MODE` | 'dual' | 评估模式: 'dual', 'rgb_only', 'sonar_only' |

**BatchNorm自适应**（用于消融实验）:

| 参数 | 值 | 描述 |
|------|-----|------|
| `TEST.BN_ADAPTATION.ENABLED` | False | 通过`--bn_adapt`标志启用 |
| `TEST.BN_ADAPTATION.NUM_BATCHES` | 30 | BN统计量重新计算的批次数 |
| `TEST.BN_ADAPTATION.DATA_SOURCE` | 'test' | 自适应数据来源 |

#### 7.0.2 可见性过滤

遵循GOT-10k评估模式，真值框为 $(0, 0, 0, 0)$（目标缺失/遮挡）的帧被排除在指标计算之外:

$$
\mathcal{F}_{eval} = \{t \mid B_t^{gt} \neq (0, 0, 0, 0) \land t > 0\}
$$

确保指标反映实际跟踪性能而非缺失检测能力。

**可见性掩码计算** (`lib/test/evaluation/experiment.py:249-273`):

$$
\text{visible}(t) = \neg(\text{all}(B_t^{gt} = 0)) \land (w_t > 0) \land (h_t > 0)
$$

#### 7.0.3 多模态评估

指标分别为RGB和Sonar模态计算:
- **RGB指标**: RGB预测与RGB真值的IoU、精确度、中心误差
- **Sonar指标**: Sonar预测与Sonar真值的IoU、精确度、中心误差
- **综合指标**: 按各模态可见帧数加权平均

### 7.1 成功率曲线 AUC

成功率定义为IoU大于阈值的帧比例:
$$
\text{Success}(\tau) = \frac{|\{t : \text{IoU}(B_t^{pred}, B_t^{gt}) > \tau\}|}{T}
$$

AUC计算在 $\tau \in [0, 1]$ 区间积分:
$$
\text{AUC} = \int_0^1 \text{Success}(\tau) d\tau
$$

### 7.2 精确度曲线

精确度定义为中心距离小于阈值的帧比例:
$$
\text{Precision}(\tau) = \frac{|\{t : \|c_t^{pred} - c_t^{gt}\|_2 < \tau\}|}{T}
$$

标准阈值: $\tau = 20$ 像素

### 7.3 平均中心误差

$$
\text{ACE} = \frac{1}{T}\sum_{t=1}^{T}\|c_t^{pred} - c_t^{gt}\|_2
$$

**代码参考**: `lib/test/evaluation/metrics.py`

---

## 8. 参考文献

### 基础框架
1. **OSTrack**: Ye et al., "Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework", ECCV 2022
   - GitHub: https://github.com/botaoye/OSTrack

2. **MixFormer**: Cui et al., "MixFormer: End-to-End Tracking with Iterative Mixed Attention", CVPR 2022
   - GitHub: https://github.com/MCG-NJU/MixFormer

### 损失函数
3. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
   - 论文: https://arxiv.org/abs/1708.02002

4. **GIoU Loss**: Rezatofighi et al., "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression", CVPR 2019
   - 论文: https://arxiv.org/abs/1902.09630

### 模型架构
5. **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
   - 论文: https://arxiv.org/abs/2010.11929

6. **CenterNet**: Zhou et al., "Objects as Points", 2019
   - 论文: https://arxiv.org/abs/1904.07850

### 优化
7. **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
   - 论文: https://arxiv.org/abs/1711.05101

### 数据集
8. **RGBS50** (RGB-Sonar跟踪基准)
   - Li, Y., et al. (2024). *RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker*
   - 论文: https://arxiv.org/abs/2406.07189 | GitHub: https://github.com/LiYunfengLYF/RGBS50

### 测试时自适应
9. **Tent** (测试时自适应)
   - Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). *Tent: Fully Test-Time Adaptation by Entropy Minimization*. ICLR 2021.
   - 论文: https://arxiv.org/abs/2006.10726

---

## 附录: 代码参考索引

| 章节 | 实现文件 | 行号 |
|------|----------|------|
| 模型架构 | `lib/models/aquasot/aquasot.py` | 16-177 |
| ViT骨干网络 | `lib/models/aquasot/vit.py` | 89-165 |
| 注意力模块 | `lib/models/aquasot/vit.py` | 30-59 |
| Transformer块 | `lib/models/aquasot/vit.py` | 62-86 |
| DCMA融合 | `lib/models/aquasot/fusion.py` | 128-167 |
| 交叉模态注意力 | `lib/models/aquasot/fusion.py` | 19-62 |
| DCMA层 | `lib/models/aquasot/fusion.py` | 65-125 |
| 融合权重 | `lib/models/aquasot/fusion.py` | 101-119 |
| CenterPredictor | `lib/models/layers/head.py` | 31-165 |
| Focal Loss | `lib/utils/focal_loss.py` | 10-59 |
| GIoU Loss | `lib/utils/box_ops.py` | 93-136 |
| 热力图生成 | `lib/utils/heatmap_utils.py` | 30-76 |
| 损失计算 | `lib/train/actors/aquasot.py` | 116-227 |
| 模板更新 | `lib/test/tracker/aquasot_tracker.py` | 296-370 |
| 目标丢失处理器 | `lib/test/tracker/target_loss_handler.py` | 1-403 |
| 模态丢弃 | `lib/utils/modality_dropout.py` | 28-77 |
| BN自适应 | `lib/utils/bn_adaptation.py` | 21-131 |
| 评估实验 | `lib/test/evaluation/experiment.py` | 1-766 |
| 可见性过滤 | `lib/test/evaluation/experiment.py` | 249-273 |
| 检查点迁移 | `lib/train/trainers/base_trainer.py` | 125-148 |
| 数据处理 | `lib/train/data/processing.py` | 38-378 |
| 配置 | `lib/config/aquasot/config.py` | 1-443 |
