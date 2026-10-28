# 为 AquaSOT 做贡献

[English](../../CONTRIBUTING.md) | 简体中文

感谢你帮助改进 AquaSOT。请遵守 [行为准则](../../CODE_OF_CONDUCT.md)。安全问题请勿提交到公开 issue，请按 [SECURITY.md](../../SECURITY.md) 中的方式报告。

## 贡献方式

- **报告缺陷**：提交 issue，附上运行的命令、所用配置（`experiments/aquasot/*.yaml`）、完整报错输出，以及 Python、PyTorch、CUDA 版本。
- **提出方案**：较大的改动（模型架构、训练流程、评估协议）请先开 issue 讨论，确定方案后再写代码。
- **Pull request**：保持改动小而集中，并清楚说明改了什么、为什么改。

## 开发环境

训练需要 Python 3.8+ 和支持 CUDA 11.8 的 GPU。

```bash
pip3 install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
python scripts/preprocess_sonar.py --all   # once, after placing the dataset under data/
pytest tests/ -v
```

## 工作流程

1. Fork 本仓库，按 [`.gitmessage`](../../.gitmessage)（Branch Naming，Variant B）命名分支：`<type>/<scope>/<short-desc>`，例如 `feature/fusion/weak-weight`。
2. 提交信息采用 [`.gitmessage`](../../.gitmessage) 定义的 `<type>(<scope>): <subject>` 格式。运行一次 `git config commit.template .gitmessage`，即可在编辑器中载入该模板。
3. 向 `main` 发起 pull request，标题同样遵循提交头格式。

如果你使用 AI 编程助手，`.agents/skills/` 中的 `git-*` 技能已遵循上述规则，详见 [AGENTS.md](../../AGENTS.md)。

## 修改规则

- 遵循现有的 OSTrack 风格代码结构，改动保持紧凑、直接。
- 新参数先加入 `lib/config/aquasot/config.py`；`experiments/aquasot/` 中的 YAML 只能覆盖已有的键。
- 行为变更需在 `tests/` 中新增或更新测试。
- 相关文档（`README.md`、`docs/`、`AGENTS.md`）在同一个 pull request 中更新；`docs/i18n/` 中的每份翻译与其英文原文同步修改。

## 提交 Pull Request 之前

- [ ] `pytest tests/ -v` 通过。
- [ ] 提交信息与分支名符合 `.gitmessage`。
- [ ] 文档与翻译已反映本次改动。
- [ ] 未提交数据集、权重或输出文件（`data/`、`outputs/`、`*.pth`）。

## 许可证

提交贡献即表示你同意以 [MIT 许可证](../../LICENSE) 授权你的贡献。
