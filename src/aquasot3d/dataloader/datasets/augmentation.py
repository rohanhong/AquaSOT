"""
Data Augmentation Pipelines

This module provides unified data augmentation pipelines that combine
RGB and Sonar transforms for multi-modal training. It handles:
- Coordinated augmentation (same random seed for paired modalities)
- Bbox-aware transforms (geometric transforms update bboxes accordingly)
- Training vs. Evaluation mode switching
- Configurable augmentation strategies

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from typing import Dict, Optional, Tuple, Union
import random
import numpy as np
import torch

from ..transforms.rgb_transforms import (
    ColorJitter,
    RandomBlur,
    RandomOcclusion,
    RandomHorizontalFlip,
    Normalize,
    ToTensor,
)

from ..transforms.sonar_transforms import (
    SpeckleNoise,
    RadialScalingJitter,
    SectorOcclusion,
    IntensityNormalization,
    FanMaskApplication,
)

from ...core.registry import AUGMENTATION_REGISTRY


class BaseAugmentation:
    """
    Base class for data augmentation pipelines.

    Args:
        is_train: Training mode flag (enables augmentation)
        seed: Random seed for reproducibility (None = random)
    """

    def __init__(self, is_train: bool = True, seed: Optional[int] = None):
        self.is_train = is_train
        self.seed = seed

    def __call__(self, data: Dict) -> Dict:
        """
        Apply augmentation to data dictionary.

        Args:
            data: Dictionary containing images and metadata

        Returns:
            Augmented data dictionary
        """
        raise NotImplementedError("Subclass must implement __call__ method")

    def set_random_seed(self):
        """Set random seed if specified."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)


@AUGMENTATION_REGISTRY.register("TrackingAugmentation")
class TrackingAugmentation(BaseAugmentation):
    """
    Augmentation pipeline for tracking data (frame-by-frame).

    This pipeline applies coordinated augmentation to RGB and Sonar images
    while ensuring bounding box consistency.

    Args:
        is_train: Enable training augmentations
        rgb_config: Configuration dict for RGB transforms
        sonar_config: Configuration dict for Sonar transforms
        seed: Random seed

    Example:
        >>> aug = TrackingAugmentation(
        ...     is_train=True,
        ...     rgb_config={'enable_color_jitter': True, 'enable_blur': True},
        ...     sonar_config={'enable_speckle': True}
        ... )
        >>> data = {
        ...     'rgb_img': np.array(...),
        ...     'sonar_img': np.array(...),
        ...     'rgb_bbox': np.array([x, y, w, h]),
        ...     'sonar_bbox': np.array([x, y, w, h]),
        ... }
        >>> augmented = aug(data)
    """

    def __init__(
        self,
        is_train: bool = True,
        rgb_config: Optional[Dict] = None,
        sonar_config: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(is_train, seed)

        if rgb_config is None:
            rgb_config = {}
        if sonar_config is None:
            sonar_config = {}

        # Initialize RGB transforms
        self.rgb_transforms = self._init_rgb_transforms(rgb_config)

        # Initialize Sonar transforms
        self.sonar_transforms = self._init_sonar_transforms(sonar_config)

        # Always apply normalization and tensor conversion
        self.rgb_normalize = Normalize()
        self.sonar_normalize = IntensityNormalization(mode='adaptive')
        self.to_tensor = ToTensor()

    def _init_rgb_transforms(self, config: Dict):
        """Initialize RGB transform pipeline."""
        transforms = []

        if self.is_train:
            if config.get('enable_color_jitter', True):
                transforms.append(ColorJitter(
                    brightness=config.get('brightness', 0.2),
                    contrast=config.get('contrast', 0.2),
                    saturation=config.get('saturation', 0.2),
                    hue=config.get('hue', 0.1),
                    prob=config.get('color_jitter_prob', 0.5),
                ))

            if config.get('enable_blur', True):
                transforms.append(RandomBlur(
                    prob=config.get('blur_prob', 0.3),
                ))

        return transforms

    def _init_sonar_transforms(self, config: Dict):
        """Initialize Sonar transform pipeline."""
        transforms = []

        if self.is_train:
            if config.get('enable_speckle', True):
                transforms.append(SpeckleNoise(
                    noise_level=config.get('speckle_level', 0.1),
                    prob=config.get('speckle_prob', 0.5),
                ))

            if config.get('enable_radial_jitter', True):
                transforms.append(RadialScalingJitter(
                    scale_range=config.get('radial_scale_range', (0.9, 1.1)),
                    prob=config.get('radial_jitter_prob', 0.3),
                ))

        return transforms

    def __call__(self, data: Dict) -> Dict:
        """
        Apply augmentation pipeline.

        Args:
            data: Dictionary with keys:
                - rgb_img: RGB image (H, W, 3)
                - sonar_img: Sonar image (H, W, 3)
                - rgb_bbox: RGB bounding box [x, y, w, h]
                - sonar_bbox: Sonar bounding box [x, y, w, h]
                - (optional) is_rgb_valid, is_sonar_valid

        Returns:
            Augmented data dictionary
        """
        self.set_random_seed()

        rgb_img = data['rgb_img'].copy() if data['rgb_img'] is not None else None
        sonar_img = data['sonar_img'].copy() if data['sonar_img'] is not None else None
        rgb_bbox = data['rgb_bbox'].copy()
        sonar_bbox = data['sonar_bbox'].copy()

        # Apply RGB transforms
        if rgb_img is not None:
            for transform in self.rgb_transforms:
                rgb_img = transform(rgb_img)

            # Apply normalization
            rgb_img = self.rgb_normalize(rgb_img)

            # Convert to tensor
            rgb_img = self.to_tensor(rgb_img)

        # Apply Sonar transforms
        if sonar_img is not None:
            for transform in self.sonar_transforms:
                sonar_img = transform(sonar_img)

            # Apply normalization
            sonar_img = self.sonar_normalize(sonar_img)

            # Apply fan mask
            fan_mask = FanMaskApplication()
            sonar_img = fan_mask(sonar_img)

            # Convert to tensor
            if not isinstance(sonar_img, torch.Tensor):
                sonar_img_float = sonar_img.astype(np.float32) / 255.0
                if sonar_img_float.ndim == 2:
                    sonar_img_float = sonar_img_float[np.newaxis, :, :]
                else:
                    sonar_img_float = sonar_img_float.transpose(2, 0, 1)
                sonar_img = torch.from_numpy(sonar_img_float.copy())

        # Update data dictionary
        data['rgb_img'] = rgb_img
        data['sonar_img'] = sonar_img
        data['rgb_bbox'] = torch.from_numpy(rgb_bbox) if isinstance(rgb_bbox, np.ndarray) else rgb_bbox
        data['sonar_bbox'] = torch.from_numpy(sonar_bbox) if isinstance(sonar_bbox, np.ndarray) else sonar_bbox

        return data


@AUGMENTATION_REGISTRY.register("TemplateSearchAugmentation")
class TemplateSearchAugmentation(BaseAugmentation):
    """
    Augmentation pipeline for template-search pairs (OSTrack-style).

    This pipeline applies different augmentation strategies to template
    and search regions, with coordinated geometric transforms.

    Args:
        is_train: Enable training augmentations
        template_config: Configuration for template augmentation
        search_config: Configuration for search augmentation
        enable_flip: Enable horizontal flip (applies to both template and search)
        seed: Random seed

    Example:
        >>> aug = TemplateSearchAugmentation(
        ...     is_train=True,
        ...     template_config={'enable_color_jitter': False},  # Less augmentation
        ...     search_config={'enable_color_jitter': True, 'enable_blur': True},
        ...     enable_flip=True
        ... )
        >>> data = {
        ...     'template_rgb': np.array(...),
        ...     'search_rgb': np.array(...),
        ...     'template_bbox_rgb': np.array([x, y, w, h]),
        ...     'search_bbox_rgb': np.array([x, y, w, h]),
        ...     ...
        ... }
        >>> augmented = aug(data)
    """

    def __init__(
        self,
        is_train: bool = True,
        template_config: Optional[Dict] = None,
        search_config: Optional[Dict] = None,
        enable_flip: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(is_train, seed)

        if template_config is None:
            template_config = {}
        if search_config is None:
            search_config = {}

        self.enable_flip = enable_flip and is_train

        # Initialize template augmentation (usually less aggressive)
        self.template_rgb_aug = TrackingAugmentation(
            is_train=is_train,
            rgb_config=template_config.get('rgb', {}),
            sonar_config=template_config.get('sonar', {}),
        )

        # Initialize search augmentation (more aggressive)
        self.search_rgb_aug = TrackingAugmentation(
            is_train=is_train,
            rgb_config=search_config.get('rgb', {}),
            sonar_config=search_config.get('sonar', {}),
        )

        # Geometric transforms (applied to both)
        if self.enable_flip:
            self.flip = RandomHorizontalFlip(prob=0.5)

    def __call__(self, data: Dict) -> Dict:
        """
        Apply augmentation to template-search pair.

        Args:
            data: Dictionary with keys:
                - template_rgb, template_sonar
                - search_rgb, search_sonar
                - template_bbox_rgb, template_bbox_sonar
                - search_bbox_rgb, search_bbox_sonar

        Returns:
            Augmented data dictionary
        """
        self.set_random_seed()

        # Prepare template data
        template_data = {
            'rgb_img': data.get('template_rgb'),
            'sonar_img': data.get('template_sonar'),
            'rgb_bbox': data.get('template_bbox_rgb'),
            'sonar_bbox': data.get('template_bbox_sonar'),
        }

        # Prepare search data
        search_data = {
            'rgb_img': data.get('search_rgb'),
            'sonar_img': data.get('search_sonar'),
            'rgb_bbox': data.get('search_bbox_rgb'),
            'sonar_bbox': data.get('search_bbox_sonar'),
        }

        # Apply flip to both (coordinated)
        if self.enable_flip:
            # Use same random state for coordinated flip
            flip_state = random.getstate()

            # Flip template
            if template_data['rgb_img'] is not None:
                random.setstate(flip_state)
                template_data['rgb_img'], template_data['rgb_bbox'] = self.flip(
                    template_data['rgb_img'], template_data['rgb_bbox']
                )

            if template_data['sonar_img'] is not None:
                random.setstate(flip_state)
                template_data['sonar_img'], template_data['sonar_bbox'] = self.flip(
                    template_data['sonar_img'], template_data['sonar_bbox']
                )

            # Flip search
            if search_data['rgb_img'] is not None:
                random.setstate(flip_state)
                search_data['rgb_img'], search_data['rgb_bbox'] = self.flip(
                    search_data['rgb_img'], search_data['rgb_bbox']
                )

            if search_data['sonar_img'] is not None:
                random.setstate(flip_state)
                search_data['sonar_img'], search_data['sonar_bbox'] = self.flip(
                    search_data['sonar_img'], search_data['sonar_bbox']
                )

        # Apply template augmentation
        template_aug = self.template_rgb_aug(template_data)

        # Apply search augmentation
        search_aug = self.search_rgb_aug(search_data)

        # Update original data dictionary
        data['template_rgb'] = template_aug['rgb_img']
        data['template_sonar'] = template_aug['sonar_img']
        data['template_bbox_rgb'] = template_aug['rgb_bbox']
        data['template_bbox_sonar'] = template_aug['sonar_bbox']

        data['search_rgb'] = search_aug['rgb_img']
        data['search_sonar'] = search_aug['sonar_img']
        data['search_bbox_rgb'] = search_aug['rgb_bbox']
        data['search_bbox_sonar'] = search_aug['sonar_bbox']

        return data


@AUGMENTATION_REGISTRY.register("NoAugmentation")
class NoAugmentation(BaseAugmentation):
    """
    No augmentation pipeline (evaluation mode).

    Only applies normalization and tensor conversion.

    Example:
        >>> aug = NoAugmentation()
        >>> data = aug(data)
    """

    def __init__(self):
        super().__init__(is_train=False)

        self.rgb_normalize = Normalize()
        self.sonar_normalize = IntensityNormalization(mode='adaptive')
        self.to_tensor = ToTensor()
        self.fan_mask = FanMaskApplication()

    def __call__(self, data: Dict) -> Dict:
        """Apply only normalization and tensor conversion."""
        rgb_img = data.get('rgb_img')
        sonar_img = data.get('sonar_img')

        # Process RGB
        if rgb_img is not None:
            rgb_img = self.rgb_normalize(rgb_img)
            rgb_img = self.to_tensor(rgb_img)

        # Process Sonar
        if sonar_img is not None:
            sonar_img = self.sonar_normalize(sonar_img)
            sonar_img = self.fan_mask(sonar_img)

            if not isinstance(sonar_img, torch.Tensor):
                sonar_img_float = sonar_img.astype(np.float32) / 255.0
                if sonar_img_float.ndim == 2:
                    sonar_img_float = sonar_img_float[np.newaxis, :, :]
                else:
                    sonar_img_float = sonar_img_float.transpose(2, 0, 1)
                sonar_img = torch.from_numpy(sonar_img_float.copy())

        data['rgb_img'] = rgb_img
        data['sonar_img'] = sonar_img

        # Convert bboxes to tensors
        for key in ['rgb_bbox', 'sonar_bbox', 'template_bbox_rgb', 'template_bbox_sonar',
                    'search_bbox_rgb', 'search_bbox_sonar']:
            if key in data and data[key] is not None:
                if isinstance(data[key], np.ndarray):
                    data[key] = torch.from_numpy(data[key])

        return data


def build_augmentation(cfg: Dict, is_train: bool = True) -> BaseAugmentation:
    """
    Build augmentation pipeline from configuration.

    Args:
        cfg: Configuration dictionary with structure:
            {
                'type': 'TrackingAugmentation' | 'TemplateSearchAugmentation' | 'NoAugmentation',
                'rgb_config': {...},
                'sonar_config': {...},
                ...
            }
        is_train: Training mode flag

    Returns:
        Augmentation pipeline instance

    Example:
        >>> cfg = {
        ...     'type': 'TemplateSearchAugmentation',
        ...     'template_config': {'rgb': {'enable_color_jitter': False}},
        ...     'search_config': {'rgb': {'enable_color_jitter': True}},
        ...     'enable_flip': True,
        ... }
        >>> aug = build_augmentation(cfg, is_train=True)
    """
    aug_type = cfg.get('type', 'TrackingAugmentation')

    if not is_train or aug_type == 'NoAugmentation':
        return NoAugmentation()

    if aug_type == 'TrackingAugmentation':
        return TrackingAugmentation(
            is_train=is_train,
            rgb_config=cfg.get('rgb_config', {}),
            sonar_config=cfg.get('sonar_config', {}),
            seed=cfg.get('seed'),
        )

    elif aug_type == 'TemplateSearchAugmentation':
        return TemplateSearchAugmentation(
            is_train=is_train,
            template_config=cfg.get('template_config', {}),
            search_config=cfg.get('search_config', {}),
            enable_flip=cfg.get('enable_flip', True),
            seed=cfg.get('seed'),
        )

    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
