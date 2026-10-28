# Contributing to AquaSOT

English | [简体中文](docs/i18n/CONTRIBUTING.zh-CN.md)

Thank you for helping improve AquaSOT. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems never go to public issues; report them as described in [SECURITY.md](SECURITY.md).

## Ways to Contribute

- **Bug reports**: open an issue with the command you ran, the config (`experiments/aquasot/*.yaml`), the full error output, and your Python, PyTorch, and CUDA versions.
- **Proposals**: for a larger change (model architecture, training pipeline, evaluation protocol), open an issue first so the approach can be agreed before you write code.
- **Pull requests**: small, focused changes with a clear description of what changed and why.

## Development Setup

Requires conda and a CUDA 11.8 capable GPU for training. `environment.yml` creates the `aquasot_env_py38` env (Python 3.8) and installs `requirements.txt` into it.

```bash
conda env create -f environment.yml
conda activate aquasot_env_py38
pip install -e .
python scripts/preprocess_sonar.py --all   # once, after placing the dataset under data/
pytest tests/ -v
```

## Workflow

1. Fork the repository and create a branch named per [`.gitmessage`](.gitmessage) (Branch Naming, Variant B): `<type>/<scope>/<short-desc>`, for example `feature/fusion/weak-weight`.
2. Commit with the `<type>(<scope>): <subject>` format defined in [`.gitmessage`](.gitmessage). Run `git config commit.template .gitmessage` once to load it in your editor.
3. Open a pull request against `main`; its title follows the same commit header format.

If you use an AI coding agent, the `git-*` skills in `.agents/skills/` follow these rules; see [AGENTS.md](AGENTS.md).

## Change Rules

- Follow the existing OSTrack-style code layout; keep changes compact and direct.
- New parameters go into `lib/config/aquasot/config.py` first; YAML files in `experiments/aquasot/` may only override existing keys.
- Add or update tests in `tests/` for behavior changes.
- Update the related docs (`README.md`, `docs/`, `AGENTS.md`) in the same pull request; each translation in `docs/i18n/` changes together with its English original.

## Before You Open a Pull Request

- [ ] `pytest tests/ -v` passes.
- [ ] Commit messages and the branch name follow `.gitmessage`.
- [ ] Docs and translations reflect the change.
- [ ] No datasets, checkpoints, or outputs (`data/`, `outputs/`, `*.pth`) are committed.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
