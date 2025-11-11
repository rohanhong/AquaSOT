"""
Configuration Loader for AquaSOT-3D

This module provides utilities for loading and merging YAML configuration files
with support for inheritance (_base_) and command-line overrides.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """
    Dictionary that supports attribute-style access.

    Example:
        >>> cfg = ConfigDict({'model': {'type': 'vit'}})
        >>> print(cfg.model.type)  # 'vit'
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Configuration dictionary
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")

    return config if config is not None else {}


def merge_configs(base_cfg: Dict, override_cfg: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base_cfg: Base configuration
        override_cfg: Override configuration

    Returns:
        Merged configuration
    """
    merged = deepcopy(base_cfg)

    for key, value in override_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def resolve_base_configs(config: Dict, config_path: Path) -> Dict:
    """
    Resolve _base_ inheritance in configuration.

    Args:
        config: Configuration dictionary with potential _base_ field
        config_path: Path to current config file (for resolving relative paths)

    Returns:
        Resolved configuration with base configs merged
    """
    if '_base_' not in config:
        return config

    base_path = config.pop('_base_')

    # Resolve relative path
    if not os.path.isabs(base_path):
        base_path = (config_path.parent / base_path).resolve()
    else:
        base_path = Path(base_path)

    logger.info(f"Loading base config from: {base_path}")

    # Load base config
    base_config = load_yaml(base_path)

    # Recursively resolve base config's bases
    base_config = resolve_base_configs(base_config, base_path)

    # Merge base config with current config
    merged = merge_configs(base_config, config)

    return merged


def parse_cli_overrides(overrides: List[str]) -> Dict:
    """
    Parse command-line configuration overrides.

    Args:
        overrides: List of override strings in format 'key.subkey=value'

    Returns:
        Dictionary of overrides

    Example:
        >>> parse_cli_overrides(['training.lr=0.001', 'model.type=vit'])
        {'training': {'lr': 0.001}, 'model': {'type': 'vit'}}
    """
    override_dict = {}

    for override in overrides:
        if '=' not in override:
            logger.warning(f"Invalid override format (missing '='): {override}")
            continue

        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')

        # Try to parse value as int, float, bool, or keep as string
        try:
            if value_str.lower() in ['true', 'false']:
                value = value_str.lower() == 'true'
            elif value_str.lower() == 'null' or value_str.lower() == 'none':
                value = None
            elif '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            value = value_str

        # Build nested dictionary
        current = override_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    return override_dict


def load_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
    as_dict: bool = False
) -> ConfigDict:
    """
    Load configuration from YAML file with support for inheritance and overrides.

    Args:
        config_path: Path to main configuration file
        overrides: List of command-line overrides in format 'key.subkey=value'
        as_dict: If True, return plain dict instead of ConfigDict

    Returns:
        Configuration object

    Example:
        >>> cfg = load_config('configs/train/train_uuv2.yaml',
        ...                  overrides=['training.lr=0.001'])
        >>> print(cfg.training.lr)  # 0.001
    """
    config_path = Path(config_path).resolve()

    logger.info(f"Loading configuration from: {config_path}")

    # Load main config
    config = load_yaml(config_path)

    # Resolve base configs
    config = resolve_base_configs(config, config_path)

    # Apply command-line overrides
    if overrides:
        logger.info(f"Applying {len(overrides)} override(s)")
        override_dict = parse_cli_overrides(overrides)
        config = merge_configs(config, override_dict)

    # Convert to ConfigDict for attribute access
    if not as_dict:
        config = ConfigDict(config)

    logger.info("Configuration loaded successfully")

    return config


def _config_to_dict(obj):
    """
    Recursively convert ConfigDict to plain dict for YAML serialization.

    Args:
        obj: Object to convert (ConfigDict, dict, list, or other)

    Returns:
        Plain dictionary/list/value
    """
    if isinstance(obj, ConfigDict):
        return {k: _config_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: _config_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_config_to_dict(item) for item in obj]
    else:
        return obj


def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Recursively convert ConfigDict to plain dict
    plain_config = _config_to_dict(config)

    with open(save_path, 'w') as f:
        yaml.safe_dump(plain_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to: {save_path}")


def validate_config(config: Dict) -> bool:
    """
    Validate configuration has required fields.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_fields = ['model', 'data']

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field missing in config: {field}")

    # Validate model config
    if 'type' not in config['model']:
        raise ValueError("model.type not specified in config")

    # Validate data config
    if 'train_root' not in config['data'] and 'test_root' not in config['data']:
        raise ValueError("Neither data.train_root nor data.test_root specified")

    logger.info("Configuration validation passed")
    return True


def print_config(config: Dict, indent: int = 0):
    """
    Pretty print configuration.

    Args:
        config: Configuration dictionary
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print('  ' * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print('  ' * indent + f"{key}: {value}")


# Convenience function for creating default logger
def setup_config_logging(level: str = 'INFO'):
    """
    Setup logging for config module.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
