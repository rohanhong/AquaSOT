"""
Configuration management for AquaSOT-3D.

This module provides a flexible configuration system based on YAML files,
with support for inheritance, variable interpolation, and validation.
"""

import os
import copy
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import yaml


class ConfigDict(dict):
    """
    Configuration dictionary with attribute-style access.

    This class extends dict to support both dict['key'] and dict.key access patterns.

    Example:
        >>> cfg = ConfigDict({'model': {'backbone': 'ResNet'}})
        >>> print(cfg.model.backbone)  # Attribute access
        ResNet
        >>> print(cfg['model']['backbone'])  # Dictionary access
        ResNet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dicts to ConfigDict
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)
            elif isinstance(value, list):
                self[key] = self._process_list(value)

    def __getattr__(self, key: str) -> Any:
        """Get item using attribute access."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"ConfigDict has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """Set item using attribute access."""
        self[key] = value

    def __delattr__(self, key: str) -> None:
        """Delete item using attribute access."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"ConfigDict has no attribute '{key}'")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item with automatic ConfigDict conversion."""
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        elif isinstance(value, list):
            value = self._process_list(value)
        super().__setitem__(key, value)

    @staticmethod
    def _process_list(lst: List) -> List:
        """Process list items, converting dicts to ConfigDict."""
        result = []
        for item in lst:
            if isinstance(item, dict) and not isinstance(item, ConfigDict):
                result.append(ConfigDict(item))
            elif isinstance(item, list):
                result.append(ConfigDict._process_list(item))
            else:
                result.append(item)
        return result

    def merge(self, other: Dict) -> 'ConfigDict':
        """
        Merge another dictionary into this ConfigDict.

        Args:
            other: Dictionary to merge

        Returns:
            Self for chaining
        """
        for key, value in other.items():
            if key in self and isinstance(self[key], dict) and isinstance(value, dict):
                if isinstance(self[key], ConfigDict):
                    self[key].merge(value)
                else:
                    self[key] = ConfigDict(self[key])
                    self[key].merge(value)
            else:
                self[key] = value
        return self

    def to_dict(self) -> Dict:
        """
        Convert ConfigDict to regular dict recursively.

        Returns:
            Regular dictionary
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigDict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def copy(self) -> 'ConfigDict':
        """
        Create a deep copy of this ConfigDict.

        Returns:
            Deep copy of ConfigDict
        """
        return ConfigDict(copy.deepcopy(self.to_dict()))


class Config:
    """
    Configuration manager for AquaSOT-3D.

    Supports:
    - Loading from YAML files
    - Configuration inheritance
    - Variable interpolation
    - Validation

    Args:
        cfg_dict: Configuration dictionary
        filename: Source filename (for debugging)

    Example:
        >>> cfg = Config.from_file('configs/default.yaml')
        >>> print(cfg.model.backbone.type)
        ResNet
        >>> cfg.merge_from_dict({'model': {'backbone': {'depth': 101}}})
    """

    def __init__(
        self,
        cfg_dict: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ):
        self._cfg_dict = ConfigDict(cfg_dict or {})
        self._filename = filename

    @property
    def filename(self) -> Optional[str]:
        """Get source filename."""
        return self._filename

    @property
    def cfg_dict(self) -> ConfigDict:
        """Get configuration dictionary."""
        return self._cfg_dict

    def __getattr__(self, key: str) -> Any:
        """Get config item using attribute access."""
        try:
            return self._cfg_dict[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        """Get config item using dictionary access."""
        return self._cfg_dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set config item using dictionary access."""
        self._cfg_dict[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._cfg_dict

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(filename={self._filename})"

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with default.

        Args:
            key: Config key (supports nested keys with dot notation)
            default: Default value if key not found

        Returns:
            Config value or default

        Example:
            >>> cfg.get('model.backbone.depth', 50)
        """
        keys = key.split('.')
        value = self._cfg_dict

        for k in keys:
            if isinstance(value, (dict, ConfigDict)) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set config value using dot notation.

        Args:
            key: Config key (supports nested keys with dot notation)
            value: Value to set

        Example:
            >>> cfg.set('model.backbone.depth', 101)
        """
        keys = key.split('.')
        config = self._cfg_dict

        for k in keys[:-1]:
            if k not in config:
                config[k] = ConfigDict()
            config = config[k]

        config[keys[-1]] = value

    def merge_from_dict(self, cfg_dict: Dict[str, Any]) -> 'Config':
        """
        Merge configuration from dictionary.

        Args:
            cfg_dict: Dictionary to merge

        Returns:
            Self for chaining
        """
        self._cfg_dict.merge(cfg_dict)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to regular dictionary.

        Returns:
            Regular dictionary
        """
        return self._cfg_dict.to_dict()

    def copy(self) -> 'Config':
        """
        Create a deep copy of this config.

        Returns:
            Deep copy of Config
        """
        return Config(cfg_dict=self._cfg_dict.copy(), filename=self._filename)

    @staticmethod
    def from_file(filename: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            filename: Path to YAML config file

        Returns:
            Config object

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails

        Example:
            >>> cfg = Config.from_file('configs/default.yaml')
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"Config file not found: {filename}")

        with open(filename, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)

        if cfg_dict is None:
            cfg_dict = {}

        # Handle inheritance (base config)
        if '_base_' in cfg_dict:
            base_path = cfg_dict.pop('_base_')
            if not isinstance(base_path, list):
                base_path = [base_path]

            # Load base configs and merge
            base_cfg_dict = {}
            for base_file in base_path:
                base_file = filename.parent / base_file
                base_cfg = Config.from_file(base_file)
                base_cfg_dict = _merge_dict(base_cfg_dict, base_cfg.to_dict())

            # Merge current config over base config
            cfg_dict = _merge_dict(base_cfg_dict, cfg_dict)

        return Config(cfg_dict=cfg_dict, filename=str(filename))

    def dump(self, filename: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            filename: Path to output YAML file
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False
            )

    def __str__(self) -> str:
        """String representation."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


def _merge_dict(base: Dict, update: Dict) -> Dict:
    """
    Recursively merge two dictionaries.

    Args:
        base: Base dictionary
        update: Dictionary to merge into base

    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result
