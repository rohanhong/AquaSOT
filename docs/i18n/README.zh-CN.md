<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../.github/assets/banner-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../../.github/assets/banner-light.svg">
  <img alt="AquaSOT" src="../../.github/assets/banner-light.svg" width="480">
</picture>

<p><b>面向水下环境的端到端 RGB-声呐单目标跟踪</b></p>

<p>
  <a href="../../LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://doi.org/10.5281/zenodo.23758685"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23758685.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.3-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://agents.md"><img alt="AGENTS.md" src="https://img.shields.io/badge/AGENTS.md-ready-24292f"></a>
  <a href="https://agentskills.io"><img alt="Agent Skills" src="https://img.shields.io/badge/Agent%20Skills-open%20format-8250df"></a>
  <!-- Reserved badges: uncomment when the service exists.
  <a href="https://github.com/rohanhong/AquaSOT/actions/workflows/<workflow>"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/rohanhong/AquaSOT/<workflow>?branch=main"></a>
  -->
  <a href="https://github.com/rohanhong/AquaSOT/releases"><img alt="Release" src="https://img.shields.io/github/v/release/rohanhong/AquaSOT"></a>
  <a href="https://github.com/rohanhong/AquaSOT/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/rohanhong/AquaSOT"></a>
  <a href="https://github.com/rohanhong/AquaSOT/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/rohanhong/AquaSOT?style=flat"></a>
</p>

<p>🌐 <a href="../../README.md">English</a> | <b>简体中文</b></p>

</div>

AquaSOT 是一个端到端多模态单目标跟踪框架，融合 RGB 相机与前视声呐（FLS）数据，无需外参标定。系统采用双流 Vision Transformer 架构，并通过可变形跨模态注意力（DCMA）实现免标定的跨模态对应学习。

## ✨ 亮点

- 🧠 **双流 ViT**：RGB（3 通道）与声呐（1 通道）各用一个 ViT-Base 主干提取特征。
- 🔗 **免标定融合**：可变形跨模态注意力在空间上对齐两种模态，无需外参标定。
- 🎯 **模态独立预测头**：RGB 与声呐使用各自的预测头，避免视场不对称导致预测漂移。
- 🔬 **内置消融**：推理阶段支持 `rgb_only`、`sonar_only`、`no_fusion` 模式，配合 BatchNorm 自适应。

## 🏗️ 架构

```
RGB Input   --> ViT Backbone --+
                               +--> DCMA Fusion --> Prediction Heads --> BBox (RGB, Sonar)
Sonar Input --> ViT Backbone --+
```

## 🚀 快速开始

> [!NOTE]
> 需要 conda（例如 [Miniforge](https://conda-forge.org/download/)）和支持 CUDA 11.8 的 GPU。`environment.yml` 会创建 `aquasot_env_py38` 环境（Python 3.8）并安装 `requirements.txt` 中的依赖；不用 conda 时，可在任意 Python 3.8+ 环境中运行 `pip install -r requirements.txt`。数据集和训练产物不包含在本仓库中：请从 RGBS50 官方发布处下载这些序列，按 [数据集](#-数据集) 一节的结构放在 `data/` 下。第 3 步仅供维护者使用：它从私有的 Hugging Face 仓库下载固定版本的副本（需要访问权限）。

1. 克隆仓库并安装：

   ```bash
   git clone https://github.com/rohanhong/AquaSOT.git
   cd AquaSOT
   conda env create -f environment.yml
   conda activate aquasot_env_py38
   pip install -e .
   ```

2. 下载 MAE 预训练权重：

   ```bash
   wget -O models/pretrained/mae_pretrain_vit_base.pth \
       https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
   ```

3. 仅限维护者：从 Hugging Face 下载固定版本的数据集和训练产物（首次使用需先登录）：

   ```bash
   hf auth login
   python scripts/fetch_artifacts.py
   ```

4. 仅针对原始数据：将声呐图像从极坐标转换为笛卡尔坐标（只需一次；下载的数据已包含 `img_rect/`）：

   ```bash
   python scripts/preprocess_sonar.py --all
   ```

5. 训练，然后评估：

   ```bash
   python tracking/train.py --script aquasot --config vitb_256_dual
   python tracking/test.py --tracker aquasot --checkpoint outputs/train/checkpoints/model.pth
   ```

## 📦 数据集

AquaSOT 使用 12 个 UUV 序列（`uuv1` 至 `uuv12`）进行训练和评估，每帧为成对的 RGB 与声呐图像。这些序列来自 Li 等人发布的 **RGBS50** RGB-声呐跟踪基准（[arXiv:2406.07189](https://arxiv.org/abs/2406.07189)）。RGBS50 由原作者发布在 [LiYunfengLYF/RGBS50](https://github.com/LiYunfengLYF/RGBS50)，其 README 给出了下载地址（[百度网盘](https://pan.baidu.com/s/1mC192X-NnH4ByMxvVtYtvg?pwd=orrx)，提取码 `orrx`）。本仓库不再分发这些数据；使用时请引用 RGBS50（见 [引用](#-引用)）。目录结构如下：

```
data/
  uuv1/                    # Training
    light/                 # RGB: img/, groundtruth.txt
    sonar/                 # Sonar: img_rect/, groundtruth_rect.txt
  uuv2/ ... uuv12/
```

**划分：** 训练集（uuv1,2,4,5,6,7,8,9,10,11,12）| 验证集（uuv3）

供维护者使用：`scripts/fetch_artifacts.py` 按 `scripts/hf_artifacts.yaml` 中固定的版本下载：数据集来自私有数据集仓库 `aquasot/aquasot-data`（每个序列一个 tar，校验 sha256 后解压到 `data/`）；训练产物来自私有模型仓库 `aquasot/aquasot-models`，下载到 `outputs/`，其中包括论文使用的 checkpoint `outputs/train/aquasot/vitb_256_dual_auto/checkpoints/checkpoint_ep0100.pth`。运行 `python scripts/fetch_artifacts.py --verify_only` 可在不下载的情况下校验本地文件；如果传输卡住，请加上 `HF_HUB_DISABLE_XET=1` 重新运行。

## 🏋️ 训练

```bash
# Standard training
python tracking/train.py --script aquasot --config vitb_256_dual

# With W&B logging
python tracking/train.py --script aquasot --config vitb_256_dual --wandb
```

配置文件位于 `experiments/aquasot/`。`scripts/auto_config.py` 可根据硬件生成调优后的 `vitb_256_dual_auto.yaml`，详见 [docs/TRAINING.md](../TRAINING.md)。

## 📊 评估

```bash
# Standard dual-modal evaluation
python tracking/test.py \
    --tracker aquasot \
    --checkpoint outputs/train/checkpoints/model.pth \
    --dataset aquasot \
    --dataset_path data/

# Ablation studies (inference-time, requires --bn_adapt)
python tracking/test.py --checkpoint model.pth --eval_mode rgb_only --bn_adapt
python tracking/test.py --checkpoint model.pth --eval_mode sonar_only --bn_adapt
python tracking/test.py --checkpoint model.pth --eval_mode no_fusion --bn_adapt

# Disable visibility filtering (include absent frames)
python tracking/test.py --checkpoint model.pth --no_filter_visibility
```

<details>
<summary>评估参数</summary>

- `--eval_mode {dual,rgb_only,sonar_only,no_fusion}`：评估所用的模态模式
- `--bn_adapt`：启用 BatchNorm 自适应（消融实验必需）
- `--no_filter_visibility`：包含所有帧（忽略 GT 缺失标记）
- `--no_skip_init_frame`：评估时包含第 0 帧
- `--plot_curves`：生成成功率与精度曲线图

</details>

**指标：** Success（AUC）、Precision@20、中心误差（按模态分别统计）

## 🗂️ 项目结构

| 路径 | 内容 |
|---|---|
| `lib/models/aquasot/` | 模型：主干、融合、预测头 |
| `lib/train/` | 训练：数据集、actor、trainer |
| `lib/test/` | 评估：跟踪器、指标 |
| `lib/config/` | 默认配置 |
| `tracking/` | 入口：`train.py`、`test.py` |
| `experiments/aquasot/` | YAML 配置 |
| `scripts/` | 工具脚本 |
| `data/` | 数据集目录 |

## 📝 引用

```bibtex
@mastersthesis{hong2026aquasot,
  title={{AquaSOT}: End-to-End Dual-Modal Single Object Tracking in Underwater Environments},
  author={Hong, Luhang},
  school={Nanyang Technological University},
  address={Singapore},
  type={Master's dissertation},
  year={2026},
  url={https://hdl.handle.net/10356/213865}
}
```

如果使用了数据集，请同时引用 RGBS50：

```bibtex
@article{li2024rgbs50,
  title={{RGB}-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker},
  author={Li, Yunfeng and Wang, Bo and Sun, Jiuran and Wu, Xueyi and Li, Ye},
  journal={arXiv preprint arXiv:2406.07189},
  year={2024}
}
```

## 📚 文档

| 文档 | 内容 |
|---|---|
| [docs/TRAINING.md](../TRAINING.md) | 前置准备、自动配置、训练、监控 |
| [docs/TESTING.md](../TESTING.md) | 评估命令与输出 |
| [docs/ABLATION.md](../ABLATION.md) | 消融实验 |
| [docs/specs/](../specs/) | 方法说明、数学推导、数据集规范 |
| [AGENTS.md](../../AGENTS.md) | 面向 AI 编程助手的说明 |

## 🤝 参与贡献

欢迎贡献。提交 issue 或 pull request 前，请先阅读 [贡献指南](CONTRIBUTING.zh-CN.md)。

## 🔗 相关资源

- **OSTrack**（ECCV 2022）：[Joint Feature Learning and Relation Modeling for Tracking](https://arxiv.org/abs/2203.11991)
- **RGBS50**（2024）：[RGB-Sonar Tracking Benchmark](https://arxiv.org/abs/2406.07189)
- **Deformable DETR**（ICLR 2021）：[Deformable Transformers for Object Detection](https://arxiv.org/abs/2010.04159)

## ⚖️ 法律信息

- 许可证：[MIT](../../LICENSE)
- 第三方声明：[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
- 行为准则：[Contributor Covenant 3.0](../../CODE_OF_CONDUCT.md)
- 安全：[安全策略](../../SECURITY.md)

## ⭐ Star 历史

<a href="https://www.star-history.com/#rohanhong/AquaSOT&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=rohanhong/AquaSOT&type=Date&theme=dark">
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=rohanhong/AquaSOT&type=Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=rohanhong/AquaSOT&type=Date" width="600">
  </picture>
</a>
