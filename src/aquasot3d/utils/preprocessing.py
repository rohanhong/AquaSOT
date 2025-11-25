"""
Shared preprocessing utilities for AquaSOT-3D training and inference.

CRITICAL FIX (Audit 2025112508 Finding 1 & 3):
This module provides unified letterbox preprocessing implementation to ensure
consistency between training and inference pipelines. Previously, online inference
used a separate implementation with zero padding and missing metadata, breaking
geometry-gated fusion.

Key features:
- Letterbox resize with aspect ratio preservation (YOLOv5/OSTrack style)
- Correct padding values: RGB=(114,114,114), Sonar=127
- Returns transformation metadata for geometry-aware processing
- Supports both RGB and single-channel sonar images

Author: AquaSOT-3D Team
Date: 2025-11-25
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Union, Optional


def letterbox_with_metadata(
    img: np.ndarray,
    target_size: int,
    is_rgb: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Letterbox preprocessing with metadata generation.

    CRITICAL FIX (Audit 2025112508 Finding 1):
    Shared implementation for training and inference to ensure consistency.

    This function:
    1. Resizes image to fit within target_size while preserving aspect ratio
    2. Pads to square with appropriate padding values (RGB=114, Sonar=127)
    3. Returns transformation metadata for coordinate mapping

    Args:
        img: Input image [H, W] for grayscale or [H, W, C] for RGB
        target_size: Target square size (e.g., 256 for template, 512 for search)
        is_rgb: True for RGB images (3 channels), False for sonar (1 channel)

    Returns:
        Tuple of (letterboxed_image, metadata):
            - letterboxed_image: [target_size, target_size, C] or [target_size, target_size]
            - metadata: Dict with keys:
                - 'scale': float, resize scale factor
                - 'pad_left': int, left padding in pixels
                - 'pad_top': int, top padding in pixels
                - 'original_size': Tuple[int, int], (h, w) before transform
                - 'new_size': Tuple[int, int], (new_h, new_w) after resize
                - 'original_height': int, original image height (for sonar radial bins)

    Example:
        >>> rgb_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> letterboxed, metadata = letterbox_with_metadata(rgb_img, 512, is_rgb=True)
        >>> print(letterboxed.shape)  # (512, 512, 3)
        >>> print(metadata['scale'])  # ~0.474 (512/1080)
    """
    h, w = img.shape[:2]

    # Track if input is single-channel with explicit dimension (H, W, 1)
    # cv2.resize removes the last dimension for (H, W, 1) inputs
    is_single_channel = img.ndim == 3 and img.shape[2] == 1

    # Compute scale factor to fit within target_size
    # Use min to ensure both dimensions fit (letterbox behavior)
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with preserved aspect ratio
    # CRITICAL FIX: Squeeze (H, W, 1) to (H, W) before resize to avoid cv2 issues
    if is_single_channel:
        img_to_resize = img.squeeze(-1)
    else:
        img_to_resize = img

    img_resized = cv2.resize(img_to_resize, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding to center the image
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # CRITICAL FIX (Audit 2025112508 Finding 3):
    # Use correct padding values (NOT zeros)
    # - RGB: (114, 114, 114) - standard YOLO/OSTrack padding
    # - Sonar: 127 - mid-gray for single-channel
    if img_resized.ndim == 2:  # Grayscale or squeezed single-channel
        pad_value = 127
    else:  # RGB
        pad_value = (114, 114, 114)

    img_letterboxed = cv2.copyMakeBorder(
        img_resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=pad_value
    )

    # CRITICAL FIX (Audit 2025112508 Finding 1):
    # Return transformation metadata for:
    # 1. Bbox coordinate transformation (scale, padding offsets)
    # 2. Geometry-gated attention (sonar sensor origin compensation)
    metadata = {
        'scale': scale,
        'pad_left': left,
        'pad_top': top,
        'original_size': (h, w),
        'new_size': (new_h, new_w),
        'original_height': h  # Needed for sonar radial bins origin computation
    }

    return img_letterboxed, metadata


def preprocess_for_inference(
    img: np.ndarray,
    target_size: int,
    is_rgb: bool = True,
    rgb_mean: Optional[Union[list, tuple, np.ndarray]] = None,
    rgb_std: Optional[Union[list, tuple, np.ndarray]] = None,
    sonar_mean: float = 0.5,
    sonar_std: float = 0.5,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, Dict]:
    """
    Complete preprocessing pipeline for inference.

    CRITICAL FIX (Audit 2025112508 Finding 3):
    Unified preprocessing that matches training pipeline exactly:
    1. Letterbox with correct padding (RGB=114, Sonar=127)
    2. Normalization with config-driven parameters (not hardcoded)
    3. Returns metadata for geometry-gated fusion

    Args:
        img: Input image (uint8 numpy array)
        target_size: Target size (256 for template, 512 for search)
        is_rgb: True for RGB, False for sonar
        rgb_mean: RGB normalization mean [R, G, B] (from config)
        rgb_std: RGB normalization std [R, G, B] (from config)
        sonar_mean: Sonar normalization mean (from config)
        sonar_std: Sonar normalization std (from config)
        device: Target device ('cuda' or 'cpu')

    Returns:
        Tuple of (preprocessed_tensor, metadata):
            - preprocessed_tensor: [1, C, H, W] normalized tensor
            - metadata: Letterbox transformation metadata

    Example:
        >>> cfg = load_config('configs/train.yaml')
        >>> tensor, metadata = preprocess_for_inference(
        ...     img, 512, is_rgb=True,
        ...     rgb_mean=cfg.data.normalize.rgb_mean,
        ...     rgb_std=cfg.data.normalize.rgb_std
        ... )
    """
    # Channel validation
    if is_rgb:
        if img.ndim == 2:
            # Convert grayscale to RGB by replicating
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] != 3:
            raise ValueError(f"RGB image must have 3 channels, got {img.shape[-1]}")
    else:
        # Sonar should be single-channel
        if img.ndim == 3:
            img = img[:, :, 0]  # Take first channel if multi-channel

    # Step 1: Letterbox with metadata
    img_letterboxed, metadata = letterbox_with_metadata(img, target_size, is_rgb)

    # Step 2: Normalize to [0, 1]
    img_float = img_letterboxed.astype(np.float32) / 255.0

    # Step 3: Apply normalization (config-driven, not hardcoded)
    if is_rgb:
        # Use config-provided normalization or default to ImageNet stats
        if rgb_mean is None:
            rgb_mean = [0.485, 0.456, 0.406]
        if rgb_std is None:
            rgb_std = [0.229, 0.224, 0.225]

        mean = np.array(rgb_mean).reshape(1, 1, 3)
        std = np.array(rgb_std).reshape(1, 1, 3)
        img_normalized = (img_float - mean) / std

        # HWC to CHW
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    else:
        # Sonar single-channel
        img_normalized = (img_float - sonar_mean) / sonar_std

        # HW to 1HW (add channel dim)
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)

    # Move to device
    img_tensor = img_tensor.to(device)

    return img_tensor, metadata


def normalize_bbox(
    bbox: np.ndarray,
    img_size: int,
    metadata: Optional[Dict] = None
) -> np.ndarray:
    """
    Normalize bounding box to [0, 1] range with optional letterbox compensation.

    Args:
        bbox: Bounding box in pixel coordinates [x, y, w, h]
        img_size: Image size (after letterbox)
        metadata: Optional letterbox metadata for coordinate transformation

    Returns:
        Normalized bbox [cx, cy, w, h] in [0, 1] range
    """
    if metadata is not None:
        # Compensate for letterbox transformation
        scale = metadata['scale']
        pad_left = metadata['pad_left']
        pad_top = metadata['pad_top']

        # Transform from original to letterbox coordinates
        x, y, w, h = bbox
        x = x * scale + pad_left
        y = y * scale + pad_top
        w = w * scale
        h = h * scale
        bbox = np.array([x, y, w, h])

    # Convert to center format and normalize
    x, y, w, h = bbox
    cx = (x + w / 2) / img_size
    cy = (y + h / 2) / img_size
    w_norm = w / img_size
    h_norm = h / img_size

    return np.array([cx, cy, w_norm, h_norm])
