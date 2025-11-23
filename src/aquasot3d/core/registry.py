"""
Registry system for AquaSOT-3D components.

This module provides a flexible registration mechanism for model components,
allowing dynamic component instantiation based on configuration files.
Inspired by MMDetection's registry system.
"""

from typing import Dict, Any, Optional, Callable, Type
import inspect
import warnings


class Registry:
    """
    A registry to map strings to classes or functions.

    This class provides a mechanism to register and retrieve components
    (e.g., backbones, fusion modules, heads) by name.

    Args:
        name: Name of the registry
        build_func: Optional function to build components (default: direct instantiation)

    Example:
        >>> BACKBONES = Registry('backbone')
        >>> @BACKBONES.register_module()
        >>> class ResNet:
        >>>     pass
        >>> backbone = BACKBONES.build(dict(type='ResNet'))
    """

    def __init__(
        self,
        name: str,
        build_func: Optional[Callable] = None
    ):
        self._name = name
        self._module_dict: Dict[str, Type] = {}
        self._build_func = build_func

    def __len__(self) -> int:
        """Get number of registered modules."""
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        """Check if key is registered."""
        return key in self._module_dict

    def __repr__(self) -> str:
        """String representation."""
        format_str = (
            f"{self.__class__.__name__}(name={self._name}, "
            f"items={list(self._module_dict.keys())})"
        )
        return format_str

    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        """Get the registry dictionary."""
        return self._module_dict

    def get(self, key: str) -> Optional[Type]:
        """
        Get registered module by key.

        Args:
            key: Module name

        Returns:
            Registered module class or None if not found
        """
        return self._module_dict.get(key, None)

    def _register_module(
        self,
        module_class: Type,
        module_name: Optional[str] = None,
        force: bool = False
    ) -> Type:
        """
        Register a module.

        Args:
            module_class: Module class to be registered
            module_name: Name of the module. If not specified, use class name
            force: Whether to override existing module with same name

        Returns:
            Registered module class
        """
        if module_name is None:
            module_name = module_class.__name__

        if not force and module_name in self._module_dict:
            raise KeyError(
                f"{module_name} is already registered in {self._name}. "
                f"Use force=True to override."
            )

        self._module_dict[module_name] = module_class
        return module_class

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module: Optional[Type] = None
    ) -> Callable:
        """
        Register a module class.

        This method can be used as a decorator or called directly.

        Args:
            name: Module name. If not specified, use class name
            force: Whether to override existing module
            module: Module class to register (when not used as decorator)

        Returns:
            Decorator function or registered class

        Example:
            >>> BACKBONES = Registry('backbone')
            >>> # Usage 1: As decorator
            >>> @BACKBONES.register_module()
            >>> class ResNet:
            >>>     pass
            >>>
            >>> # Usage 2: As decorator with name
            >>> @BACKBONES.register_module(name='resnet50')
            >>> class ResNet50:
            >>>     pass
            >>>
            >>> # Usage 3: Direct call
            >>> BACKBONES.register_module(module=ResNet)
        """
        if module is not None:
            # Direct registration
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # Used as decorator
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register

    def register(self, name: Optional[str] = None, force: bool = False) -> Callable:
        """
        Alias for register_module() for backward compatibility.

        Args:
            name: Module name. If not specified, use class name
            force: Whether to override existing module

        Returns:
            Decorator function
        """
        return self.register_module(name=name, force=force)

    def build(self, cfg: Dict[str, Any], **kwargs) -> Any:
        """
        Build a module from configuration.

        Args:
            cfg: Configuration dictionary containing 'type' key
            **kwargs: Additional arguments passed to module constructor

        Returns:
            Instantiated module

        Example:
            >>> BACKBONES = Registry('backbone')
            >>> @BACKBONES.register_module()
            >>> class ResNet:
            >>>     def __init__(self, depth, num_classes):
            >>>         pass
            >>> cfg = dict(type='ResNet', depth=50, num_classes=1000)
            >>> model = BACKBONES.build(cfg)
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, but got {type(cfg)}")

        if 'type' not in cfg:
            raise KeyError('Config dict must contain "type" key')

        cfg = cfg.copy()
        module_type = cfg.pop('type')

        if isinstance(module_type, str):
            module_class = self.get(module_type)
            if module_class is None:
                raise KeyError(
                    f"{module_type} is not registered in {self._name}. "
                    f"Available: {list(self._module_dict.keys())}"
                )
        elif inspect.isclass(module_type):
            module_class = module_type
        else:
            raise TypeError(
                f"type must be a str or class, but got {type(module_type)}"
            )

        # Merge kwargs with cfg, kwargs have higher priority
        cfg.update(kwargs)

        # Use custom build function if provided
        if self._build_func is not None:
            return self._build_func(module_class, cfg)

        # Default: direct instantiation
        return module_class(**cfg)

    def build_from_cfg(self, cfg: Dict[str, Any], **kwargs) -> Any:
        """
        Alias for build() method for backward compatibility.

        Args:
            cfg: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Instantiated module
        """
        return self.build(cfg, **kwargs)


# ============================================================================
# Pre-defined Registry Instances
# ============================================================================

# Model component registries
BACKBONES = Registry('backbone')
FUSION_MODULES = Registry('fusion')
PREDICTION_HEADS = Registry('prediction_head')
TRACKERS = Registry('tracker')

# Data component registries
DATASETS = Registry('dataset')
TRANSFORMS = Registry('transform')
SAMPLERS = Registry('sampler')
AUGMENTATIONS = Registry('augmentation')

# Training component registries
OPTIMIZERS = Registry('optimizer')
SCHEDULERS = Registry('scheduler')
LOSSES = Registry('loss')

# Evaluation component registries
METRICS = Registry('metric')
EVALUATORS = Registry('evaluator')

# Aliases for backward compatibility (using _REGISTRY naming convention)
BACKBONE_REGISTRY = BACKBONES
FUSION_REGISTRY = FUSION_MODULES
PREDICTION_HEAD_REGISTRY = PREDICTION_HEADS
TRACKER_REGISTRY = TRACKERS
DATASET_REGISTRY = DATASETS
TRANSFORM_REGISTRY = TRANSFORMS
SAMPLER_REGISTRY = SAMPLERS
AUGMENTATION_REGISTRY = AUGMENTATIONS
OPTIMIZER_REGISTRY = OPTIMIZERS
SCHEDULER_REGISTRY = SCHEDULERS
LOSS_REGISTRY = LOSSES
METRIC_REGISTRY = METRICS
EVALUATOR_REGISTRY = EVALUATORS


def build_from_cfg(
    cfg: Dict[str, Any],
    registry: Registry,
    default_args: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Build a module from config dict.

    Args:
        cfg: Config dict containing 'type' key
        registry: Registry to build from
        default_args: Default arguments merged with cfg

    Returns:
        Instantiated module

    Example:
        >>> BACKBONES = Registry('backbone')
        >>> @BACKBONES.register_module()
        >>> class ResNet:
        >>>     def __init__(self, depth):
        >>>         pass
        >>> cfg = dict(type='ResNet', depth=50)
        >>> model = build_from_cfg(cfg, BACKBONES)
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")

    if not isinstance(registry, Registry):
        raise TypeError(
            f"registry must be a Registry instance, but got {type(registry)}"
        )

    # Merge default args
    if default_args is not None:
        cfg = {**default_args, **cfg}

    return registry.build(cfg)
