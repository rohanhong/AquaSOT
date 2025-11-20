"""
Bbox-Aware Data Augmentation Pipelines

Updated augmentation pipelines using bbox-aware transforms from
Audit 2025111801 refactoring. Maintains backward compatibility with
existing configs while enabling proper bbox-image synchronization.

This module replaces augmentation.py for training with bbox-aware transforms,
solving the coordinate desynchronization bugs that caused training failures.

Author: AquaSOT-3D Team
Date: 2025-11-18
"""

from typing import Dict, Optional
import random
import numpy as np
import torch

from ..transforms.rgb_transforms import (
    ToTensor,  # Keep ToTensor as utility
    ColorJitterBboxAware,
    RandomBlurBboxAware,
    NormalizeBboxAware,
    RandomHorizontalFlipBboxAware,
    RandomOcclusionBboxAware,
)

from ..transforms.sonar_transforms import (
    SpeckleNoiseBboxAware,
    IntensityNormalizationBboxAware,
)

from ..transforms.base import (
    ComposeTransforms,
)

from ...core.registry import AUGMENTATION_REGISTRY


class BaseAugmentation:
    """
    Base class for bbox-aware data augmentation pipelines.

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
    Bbox-aware augmentation pipeline for tracking data.

    CRITICAL FIX (Audit 2025111801):
    This pipeline uses bbox-aware transforms to maintain coordinate
    synchronization between images and bboxes, solving the training
    failure caused by desynchronized annotations.

    Args:
        is_train: Enable training augmentations
        rgb_config: Configuration dict for RGB transforms
        sonar_config: Configuration dict for Sonar transforms
        seed: Random seed

    Example:
        >>> aug = TrackingAugmentationBboxAware(
        ...     is_train=True,
        ...     rgb_config={'enable_color_jitter': True},
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
        **kwargs  # Accept but ignore legacy logpolar params for backward compatibility
    ):
        super().__init__(is_train, seed)

        if rgb_config is None:
            rgb_config = {}
        if sonar_config is None:
            sonar_config = {}

        # Initialize RGB transforms
        self.rgb_transforms = self._init_rgb_transforms(rgb_config)

        # Initialize Sonar bbox-aware transforms
        self.sonar_transforms = self._init_sonar_transforms(sonar_config)

        # Always apply normalization and tensor conversion
        self.rgb_normalize = NormalizeBboxAware(bbox_format='xywh')
        self.to_tensor = ToTensor()

    def _init_rgb_transforms(self, config: Dict):
        """
        Initialize RGB bbox-aware transform pipeline.

        CRITICAL UPDATE (Audit 2025111901 Phase 2):
        Now uses native bbox-aware transforms instead of legacy wrapped transforms.
        """
        transforms = []

        if self.is_train:
            # Photometric transforms (bbox-preserving)
            if config.get('enable_color_jitter', True):
                transforms.append(ColorJitterBboxAware(
                    brightness=config.get('brightness', 0.2),
                    contrast=config.get('contrast', 0.2),
                    saturation=config.get('saturation', 0.2),
                    hue=config.get('hue', 0.1),
                    prob=config.get('color_jitter_prob', 0.5),
                    bbox_format='xywh'
                ))

            if config.get('enable_blur', True):
                transforms.append(RandomBlurBboxAware(
                    kernel_sizes=config.get('blur_kernel_sizes', [3, 5, 7]),
                    sigma_range=config.get('blur_sigma_range', (0.1, 2.0)),
                    prob=config.get('blur_prob', 0.3),
                    bbox_format='xywh'
                ))

            # Geometric transforms (bbox-aware)
            if config.get('enable_horizontal_flip', False):
                transforms.append(RandomHorizontalFlipBboxAware(
                    prob=config.get('horizontal_flip_prob', 0.5),
                    bbox_format='xywh'
                ))

            if config.get('enable_occlusion', False):
                transforms.append(RandomOcclusionBboxAware(
                    min_size=config.get('occlusion_min_size', 0.05),
                    max_size=config.get('occlusion_max_size', 0.3),
                    prob=config.get('occlusion_prob', 0.2),
                    fill_mode=config.get('occlusion_fill_mode', 'zero'),
                    occlusion_threshold=config.get('occlusion_threshold', 0.5),
                    bbox_format='xywh'
                ))

        return transforms

    def _init_sonar_transforms(self, config: Dict):
        """Initialize Sonar bbox-aware transform pipeline."""
        transforms = []

        if self.is_train:
            # Speckle noise (photometric, bbox-preserving)
            if config.get('enable_speckle', True):
                transforms.append(SpeckleNoiseBboxAware(
                    noise_level=config.get('speckle_level', 0.1),
                    prob=config.get('speckle_prob', 0.5),
                    bbox_format='xywh'
                ))

        # Always apply intensity normalization
        # NOTE: Fan mask removed for rectangular coordinates (not applicable)
        transforms.append(IntensityNormalizationBboxAware(
            mode=config.get('normalization_mode', 'global'),
            bbox_format='xywh'
        ))

        return transforms

    def __call__(self, data: Dict) -> Dict:
        """
        Apply bbox-aware augmentation pipeline with validity tracking.

        CRITICAL FIX (Audit 2025111903 Phase 2): Propagate validity flags through
        transform chain, updating when transforms filter bboxes.

        Args:
            data: Dictionary with keys:
                - rgb_img: RGB image (H, W, 3)
                - sonar_img: Sonar image (H, W, C)
                - rgb_bbox: RGB bounding box [x, y, w, h]
                - sonar_bbox: Sonar bounding box [x, y, w, h]
                - is_valid_rgb: Boolean indicating RGB validity (optional, default True)
                - is_valid_sonar: Boolean indicating Sonar validity (optional, default True)
                - (optional) metadata dicts for padding info

        Returns:
            Augmented data dictionary with synchronized images, bboxes, and updated validity
        """
        self.set_random_seed()

        # Extract initial validity flags
        is_valid_rgb = data.get('is_valid_rgb', True)
        is_valid_sonar = data.get('is_valid_sonar', True)

        rgb_img = data['rgb_img'].copy() if data['rgb_img'] is not None else None
        sonar_img = data['sonar_img'].copy() if data['sonar_img'] is not None else None
        rgb_bbox = data['rgb_bbox'].copy()
        sonar_bbox = data['sonar_bbox'].copy()

        # Apply RGB bbox-aware transforms
        if rgb_img is not None:
            current_rgb = rgb_img
            # Ensure bbox is 2D (1, 4) format
            current_rgb_bbox = rgb_bbox.reshape(1, 4) if rgb_bbox.ndim == 1 else rgb_bbox

            # Only apply transforms to valid samples
            if is_valid_rgb:
                for transform in self.rgb_transforms:
                    result = transform(image=current_rgb, bboxes=current_rgb_bbox)
                    current_rgb = result['image']
                    current_rgb_bbox = result['bboxes']

                    # CRITICAL FIX (Audit 2025111903 Phase 2): Track validity when bbox filtered
                    if current_rgb_bbox.shape[0] == 0:
                        is_valid_rgb = False  # Mark as invalid
                        h, w = current_rgb.shape[:2]
                        dummy_bbox = np.array([[w // 2, h // 2, 1, 1]], dtype=np.float32)
                        current_rgb_bbox = dummy_bbox

            # Apply normalization (bbox-aware)
            result = self.rgb_normalize(image=current_rgb, bboxes=current_rgb_bbox)
            current_rgb = result['image']
            current_rgb_bbox = result['bboxes']

            # Convert to tensor (legacy utility, operates on image only)
            rgb_img = self.to_tensor(current_rgb)
            # Flatten bbox back to (4,) if it was originally 1D
            rgb_bbox = current_rgb_bbox.flatten() if current_rgb_bbox.shape[0] == 1 else current_rgb_bbox

        # Apply Sonar bbox-aware transforms
        if sonar_img is not None:
            current_sonar = sonar_img
            # Ensure bbox is 2D (1, 4) format
            current_sonar_bbox = sonar_bbox.reshape(1, 4) if sonar_bbox.ndim == 1 else sonar_bbox

            # Only apply transforms to valid samples
            if is_valid_sonar:
                # Apply augmentation transforms
                for transform in self.sonar_transforms:
                    result = transform(
                        image=current_sonar,
                        bboxes=current_sonar_bbox,
                        metadata=data.get('sonar_search_metadata')  # Pass padding info
                    )
                    current_sonar = result['image']
                    current_sonar_bbox = result['bboxes']

                    # CRITICAL FIX (Audit 2025111903 Phase 2): Track validity when bbox filtered
                    if current_sonar_bbox.shape[0] == 0:
                        is_valid_sonar = False  # Mark as invalid
                        h, w = current_sonar.shape[:2]
                        dummy_bbox = np.array([[w // 2, h // 2, 1, 1]], dtype=np.float32)
                        current_sonar_bbox = dummy_bbox


            # Convert to tensor
            if not isinstance(current_sonar, torch.Tensor):
                sonar_img_float = current_sonar.astype(np.float32) / 255.0
                if sonar_img_float.ndim == 2:
                    sonar_img_float = sonar_img_float[np.newaxis, :, :]
                else:
                    sonar_img_float = sonar_img_float.transpose(2, 0, 1)
                sonar_img = torch.from_numpy(sonar_img_float.copy())
            else:
                sonar_img = current_sonar

            # Flatten bbox back to (4,) if it was originally 1D
            sonar_bbox = current_sonar_bbox.flatten() if current_sonar_bbox.shape[0] == 1 else current_sonar_bbox

        # Update data dictionary
        data['rgb_img'] = rgb_img
        data['sonar_img'] = sonar_img
        data['rgb_bbox'] = torch.from_numpy(rgb_bbox) if isinstance(rgb_bbox, np.ndarray) else rgb_bbox
        data['sonar_bbox'] = torch.from_numpy(sonar_bbox) if isinstance(sonar_bbox, np.ndarray) else sonar_bbox

        # CRITICAL FIX (Audit 2025111903 Phase 2): Include updated validity flags
        data['is_valid_rgb'] = is_valid_rgb
        data['is_valid_sonar'] = is_valid_sonar
        data['is_valid'] = is_valid_rgb and is_valid_sonar

        return data


@AUGMENTATION_REGISTRY.register("TemplateSearchAugmentation")
class TemplateSearchAugmentation(BaseAugmentation):
    """
    Bbox-aware augmentation pipeline for template-search pairs (OSTrack-style).

    This pipeline applies different augmentation strategies to template and search
    regions, with coordinated geometric transforms to maintain synchronization.

    CRITICAL FIX (Audit 2025111801):
    Uses bbox-aware transforms to maintain coordinate synchronization for both
    template and search regions, preventing bbox desynchronization bugs.

    Args:
        is_train: Enable training augmentations
        template_config: Configuration dict for template augmentation
        search_config: Configuration dict for search augmentation
        enable_flip: Enable coordinated horizontal flip (default True)
        seed: Random seed for reproducibility

    Example:
        >>> aug = TemplateSearchAugmentation(
        ...     is_train=True,
        ...     template_config={
        ...         'rgb': {'enable_color_jitter': False},  # Less aggressive
        ...         'sonar': {'enable_speckle': False}
        ...     },
        ...     search_config={
        ...         'rgb': {'enable_color_jitter': True, 'enable_blur': True},
        ...         'sonar': {'enable_speckle': True}
        ...     },
        ...     enable_flip=True
        ... )
        >>> data = {
        ...     'template_rgb': np.array(...),
        ...     'search_rgb': np.array(...),
        ...     'template_bbox_rgb': np.array([x, y, w, h]),
        ...     'search_bbox_rgb': np.array([x, y, w, h]),
        ...     'template_sonar': np.array(...),
        ...     'search_sonar': np.array(...),
        ...     'template_bbox_sonar': np.array([x, y, w, h]),
        ...     'search_bbox_sonar': np.array([x, y, w, h]),
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
        **kwargs  # Accept but ignore legacy logpolar params
    ):
        super().__init__(is_train, seed)

        if template_config is None:
            template_config = {}
        if search_config is None:
            search_config = {}

        self.enable_flip = enable_flip and is_train

        # Initialize template augmentation (usually less aggressive)
        self.template_aug = TrackingAugmentationBboxAware(
            is_train=is_train,
            rgb_config=template_config.get('rgb', {}),
            sonar_config=template_config.get('sonar', {}),
            seed=seed,
        )

        # Initialize search augmentation (more aggressive)
        self.search_aug = TrackingAugmentationBboxAware(
            is_train=is_train,
            rgb_config=search_config.get('rgb', {}),
            sonar_config=search_config.get('sonar', {}),
            seed=seed,
        )

        # Geometric transforms for coordinated flip
        if self.enable_flip:
            self.rgb_flip = RandomHorizontalFlipBboxAware(prob=0.5, bbox_format='xywh')
            self.sonar_flip = RandomHorizontalFlipBboxAware(prob=0.5, bbox_format='xywh')

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

        # Apply coordinated flip if enabled (same random state for template and search)
        if self.enable_flip:
            # Save random state for coordinated flip
            flip_state = random.getstate()

            # Flip template RGB
            if 'template_rgb' in data and data['template_rgb'] is not None:
                random.setstate(flip_state)
                template_rgb_bbox = data.get('template_bbox_rgb')
                if template_rgb_bbox is not None:
                    template_rgb_bbox = template_rgb_bbox.reshape(1, 4) if template_rgb_bbox.ndim == 1 else template_rgb_bbox
                    result = self.rgb_flip(image=data['template_rgb'], bboxes=template_rgb_bbox)
                    data['template_rgb'] = result['image']
                    data['template_bbox_rgb'] = result['bboxes'].flatten() if result['bboxes'].shape[0] == 1 else result['bboxes']

            # Flip search RGB (same flip state)
            if 'search_rgb' in data and data['search_rgb'] is not None:
                random.setstate(flip_state)
                search_rgb_bbox = data.get('search_bbox_rgb')
                if search_rgb_bbox is not None:
                    search_rgb_bbox = search_rgb_bbox.reshape(1, 4) if search_rgb_bbox.ndim == 1 else search_rgb_bbox
                    result = self.rgb_flip(image=data['search_rgb'], bboxes=search_rgb_bbox)
                    data['search_rgb'] = result['image']
                    data['search_bbox_rgb'] = result['bboxes'].flatten() if result['bboxes'].shape[0] == 1 else result['bboxes']

            # Flip template Sonar (same flip state)
            if 'template_sonar' in data and data['template_sonar'] is not None:
                random.setstate(flip_state)
                template_sonar_bbox = data.get('template_bbox_sonar')
                if template_sonar_bbox is not None:
                    template_sonar_bbox = template_sonar_bbox.reshape(1, 4) if template_sonar_bbox.ndim == 1 else template_sonar_bbox
                    result = self.sonar_flip(image=data['template_sonar'], bboxes=template_sonar_bbox)
                    data['template_sonar'] = result['image']
                    data['template_bbox_sonar'] = result['bboxes'].flatten() if result['bboxes'].shape[0] == 1 else result['bboxes']

            # Flip search Sonar (same flip state)
            if 'search_sonar' in data and data['search_sonar'] is not None:
                random.setstate(flip_state)
                search_sonar_bbox = data.get('search_bbox_sonar')
                if search_sonar_bbox is not None:
                    search_sonar_bbox = search_sonar_bbox.reshape(1, 4) if search_sonar_bbox.ndim == 1 else search_sonar_bbox
                    result = self.sonar_flip(image=data['search_sonar'], bboxes=search_sonar_bbox)
                    data['search_sonar'] = result['image']
                    data['search_bbox_sonar'] = result['bboxes'].flatten() if result['bboxes'].shape[0] == 1 else result['bboxes']

        # Prepare template data for augmentation
        template_data = {
            'rgb_img': data.get('template_rgb'),
            'sonar_img': data.get('template_sonar'),
            'rgb_bbox': data.get('template_bbox_rgb'),
            'sonar_bbox': data.get('template_bbox_sonar'),
        }

        # Prepare search data for augmentation
        search_data = {
            'rgb_img': data.get('search_rgb'),
            'sonar_img': data.get('search_sonar'),
            'rgb_bbox': data.get('search_bbox_rgb'),
            'sonar_bbox': data.get('search_bbox_sonar'),
        }

        # Apply template-specific augmentation
        template_aug = self.template_aug(template_data)

        # Apply search-specific augmentation
        search_aug = self.search_aug(search_data)

        # Update original data dictionary
        data['template_rgb'] = template_aug['rgb_img']
        data['template_sonar'] = template_aug['sonar_img']
        data['template_bbox_rgb'] = template_aug['rgb_bbox']
        data['template_bbox_sonar'] = template_aug['sonar_bbox']

        data['search_rgb'] = search_aug['rgb_img']
        data['search_sonar'] = search_aug['sonar_img']
        data['search_bbox_rgb'] = search_aug['rgb_bbox']
        data['search_bbox_sonar'] = search_aug['sonar_bbox']

        # CRITICAL FIX (Audit 2025112001): Propagate validity flags from augmentation
        # TrackingAugmentationBboxAware returns is_valid_rgb, is_valid_sonar, is_valid
        # These must be propagated to support loss masking for invalid samples
        data['template_is_valid_rgb'] = template_aug.get('is_valid_rgb', True)
        data['template_is_valid_sonar'] = template_aug.get('is_valid_sonar', True)
        data['template_is_valid'] = template_aug.get('is_valid', True)

        data['search_is_valid_rgb'] = search_aug.get('is_valid_rgb', True)
        data['search_is_valid_sonar'] = search_aug.get('is_valid_sonar', True)
        data['search_is_valid'] = search_aug.get('is_valid', True)

        return data


@AUGMENTATION_REGISTRY.register("NoAugmentation")
class NoAugmentation(BaseAugmentation):
    """
    Bbox-aware no-augmentation pipeline (evaluation mode).

    Only applies normalization with proper bbox synchronization.

    Example:
        >>> aug = NoAugmentationBboxAware()
        >>> data = aug(data)
    """

    def __init__(self, **kwargs):
        """No augmentation - only normalization."""
        super().__init__(is_train=False)

        self.rgb_normalize = NormalizeBboxAware(bbox_format='xywh')
        self.to_tensor = ToTensor()

        # Sonar transforms (minimal)
        # NOTE: Fan mask removed for rectangular coordinates (not applicable)
        self.sonar_normalize = IntensityNormalizationBboxAware(
            mode='adaptive', bbox_format='xywh'
        )

    def __call__(self, data: Dict) -> Dict:
        """Apply only normalization and tensor conversion."""
        rgb_img = data.get('rgb_img')
        sonar_img = data.get('sonar_img')
        rgb_bbox = data.get('rgb_bbox')
        sonar_bbox = data.get('sonar_bbox')

        # Process RGB with bbox awareness
        if rgb_img is not None:
            # Ensure bbox is 2D format if present
            current_rgb_bbox = rgb_bbox.reshape(1, 4) if rgb_bbox is not None and rgb_bbox.ndim == 1 else rgb_bbox

            # Normalize (bbox-aware)
            result = self.rgb_normalize(image=rgb_img, bboxes=current_rgb_bbox)
            rgb_img = result['image']
            if rgb_bbox is not None:
                rgb_bbox = result['bboxes'].flatten() if result['bboxes'].shape[0] == 1 else result['bboxes']

            # Convert to tensor
            rgb_img = self.to_tensor(rgb_img)

        # Process Sonar with bbox awareness
        if sonar_img is not None:
            # Normalize (fan mask removed for rectangular coordinates)
            result = self.sonar_normalize(image=sonar_img, bboxes=sonar_bbox)
            sonar_img = result['image']
            sonar_bbox = result['bboxes']

            # Convert to tensor
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
        for key in ['rgb_bbox', 'sonar_bbox']:
            if key in data and data[key] is not None:
                if isinstance(data[key], np.ndarray):
                    data[key] = torch.from_numpy(data[key])

        return data


def build_augmentation_bbox_aware(
    cfg: Dict,
    is_train: bool = True
) -> BaseAugmentation:
    """
    Build bbox-aware augmentation pipeline from configuration.

    Args:
        cfg: Configuration dictionary
        is_train: Training mode flag

    Returns:
        Bbox-aware augmentation pipeline instance

    Example:
        >>> cfg = {
        ...     'type': 'TrackingAugmentationBboxAware',
        ...     'sonar_config': {'enable_speckle': True}
        ... }
        >>> aug = build_augmentation_bbox_aware(cfg, is_train=True)
    """
    aug_type = cfg.get('type', 'TrackingAugmentationBboxAware')

    if not is_train or aug_type == 'NoAugmentationBboxAware':
        return NoAugmentationBboxAware()

    if aug_type == 'TrackingAugmentationBboxAware':
        return TrackingAugmentationBboxAware(
            is_train=is_train,
            rgb_config=cfg.get('rgb', {}),
            sonar_config=cfg.get('sonar', {}),
            seed=cfg.get('seed'),
        )

    elif aug_type == 'TemplateSearchAugmentationBboxAware' or aug_type == 'TemplateSearchAugmentation':
        return TemplateSearchAugmentation(
            is_train=is_train,
            template_config=cfg.get('template_config', {}),
            search_config=cfg.get('search_config', {}),
            enable_flip=cfg.get('enable_flip', True),
            seed=cfg.get('seed'),
        )

    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


# ==============================================================================
# Backward Compatibility Aliases
# ==============================================================================

# Legacy names with BboxAware suffix for backward compatibility with old imports
BaseAugmentationBboxAware = BaseAugmentation
TrackingAugmentationBboxAware = TrackingAugmentation
TemplateSearchAugmentationBboxAware = TemplateSearchAugmentation
NoAugmentationBboxAware = NoAugmentation
build_augmentation = build_augmentation_bbox_aware  # Alias for backward compatibility

# Register legacy names in registry for config file backward compatibility
# This allows configs to use both "TrackingAugmentation" and "TrackingAugmentationBboxAware"
AUGMENTATION_REGISTRY.register("TrackingAugmentationBboxAware", force=True)(TrackingAugmentation)
AUGMENTATION_REGISTRY.register("TemplateSearchAugmentationBboxAware", force=True)(TemplateSearchAugmentation)
AUGMENTATION_REGISTRY.register("NoAugmentationBboxAware", force=True)(NoAugmentation)
