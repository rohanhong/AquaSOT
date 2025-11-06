"""
Base module classes for AquaSOT-3D components.

This module defines abstract base classes that all model components should inherit from.
It ensures consistent interfaces across different implementations and facilitates
component registration and factory pattern usage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import logging


class BaseModule(nn.Module, ABC):
    """
    Abstract base class for all AquaSOT-3D modules.

    All model components (backbones, fusion modules, heads, etc.) should inherit
    from this class to ensure consistent interface and behavior.

    Args:
        config: Configuration dictionary for the module
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self._is_initialized = False

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the module.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Forward method must be implemented by subclass")

    def init_weights(self) -> None:
        """
        Initialize module weights.

        Override this method to implement custom weight initialization.
        Default implementation uses PyTorch's default initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self._is_initialized = True

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary of this module.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def extra_repr(self) -> str:
        """
        Extra representation string for printing.

        Returns:
            String representation of key configuration parameters
        """
        return f"config_keys={list(self.config.keys())}"


class BaseBackbone(BaseModule):
    """
    Abstract base class for backbone networks.

    Backbones extract multi-scale features from input images (RGB or Sonar).
    All backbone implementations should inherit from this class.

    Args:
        config: Configuration dictionary for the backbone
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_dims = []  # List of feature dimensions at each scale
        self.output_strides = []  # List of output strides at each scale

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from input image.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dictionary containing:
                - 'features': List of feature tensors at different scales
                - 'feature_dims': List of feature dimensions
                - 'output_strides': List of output strides
        """
        raise NotImplementedError

    def get_feature_dims(self) -> list:
        """
        Get the feature dimensions at each scale.

        Returns:
            List of feature dimensions
        """
        return self.feature_dims

    def get_output_strides(self) -> list:
        """
        Get the output strides at each scale.

        Returns:
            List of output strides
        """
        return self.output_strides


class BaseFusion(BaseModule):
    """
    Abstract base class for fusion modules.

    Fusion modules combine features from multiple modalities (RGB and Sonar).
    All fusion implementations should inherit from this class.

    Args:
        config: Configuration dictionary for the fusion module
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @abstractmethod
    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse RGB and Sonar features.

        Args:
            rgb_features: RGB feature tensor of shape (B, C_rgb, H, W)
            sonar_features: Sonar feature tensor of shape (B, C_sonar, H, W)
            **kwargs: Additional arguments (e.g., geometric priors, quality scores)

        Returns:
            Dictionary containing:
                - 'fused_features': Fused feature tensor
                - 'fusion_weights': Attention weights (optional)
                - 'aux_outputs': Auxiliary outputs for loss computation (optional)
        """
        raise NotImplementedError


class BaseTemporalHead(BaseModule):
    """
    Abstract base class for temporal modeling heads.

    Temporal heads aggregate features across time for robust tracking.
    All temporal head implementations should inherit from this class.

    Args:
        config: Configuration dictionary for the temporal head
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        temporal_state: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Perform temporal feature aggregation.

        Args:
            features: Current frame features of shape (B, C, H, W) or (B, T, C, H, W)
            temporal_state: Previous temporal state (hidden state, memory, etc.)

        Returns:
            Tuple containing:
                - aggregated_features: Temporally aggregated features
                - new_temporal_state: Updated temporal state
        """
        raise NotImplementedError

    def reset_state(self) -> None:
        """
        Reset temporal state (e.g., for new sequence).

        Override this method if the temporal head maintains internal state.
        """
        pass


class BaseHead(BaseModule):
    """
    Abstract base class for prediction heads.

    Prediction heads output final predictions (bounding boxes, 3D centers, etc.).
    All prediction head implementations should inherit from this class.

    Args:
        config: Configuration dictionary for the prediction head
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Predict outputs from features.

        Args:
            features: Input feature tensor
            **kwargs: Additional arguments

        Returns:
            Dictionary containing predictions (format depends on head type)
        """
        raise NotImplementedError


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for loss functions.

    All custom loss functions should inherit from this class.

    Args:
        weight: Weight for this loss component
        **kwargs: Additional loss-specific arguments
    """

    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__()
        self.weight = weight
        self.kwargs = kwargs

    @abstractmethod
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss value.

        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of ground truth targets
            **kwargs: Additional arguments

        Returns:
            Scalar loss tensor
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute weighted loss.

        Returns:
            Weighted scalar loss tensor
        """
        loss = self.forward(*args, **kwargs)
        return self.weight * loss


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    All evaluation metrics should inherit from this class.
    """

    def __init__(self):
        self.reset()

    @abstractmethod
    def update(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> None:
        """
        Update metric with new predictions and targets.

        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of ground truth targets
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """
        Compute final metric values.

        Returns:
            Dictionary of metric names and values
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset metric state.
        """
        raise NotImplementedError
