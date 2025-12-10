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


def crop_metadata_to_letterbox(
    crop_meta: Dict,
    search_size: int = 256
) -> Dict:
    """
    Convert crop-based metadata to letterbox-equivalent format for fusion module.

    This function unifies metadata format across training/validation/online evaluation.
    The fusion module expects letterbox-style metadata with specific keys.
    Training pipeline produces crop-style metadata. This function bridges the gap.

    Crop metadata format (from TemplateSearchDataset/OnlineEvaluator):
        - crop_x1, crop_y1: Top-left corner of crop in original image
        - crop_sz: Size of crop region before resize
        - resize_factor: output_size / crop_sz
        - pad_left, pad_top, pad_right, pad_bottom: Padding added for boundary handling
        - original_size: (H, W) of source image

    Letterbox metadata format (expected by fusion module):
        - scale: Resize factor from crop to output
        - scale_x, scale_y: Directional scale factors
        - pad_top, pad_left, pad_x, pad_y: Padding in output pixel space
        - original_height, original_width: Original image dimensions
        - new_size: (new_h, new_w) after scaling

    Semantic mapping:
        - For target-centered crops (OSTrack style), the "original" in letterbox terms
          is the crop region, not the full image
        - scale = resize_factor (crop_sz -> output_size)
        - pad_top = y1_pad * resize_factor (padding in output space)
        - original_height = crop_sz (the "original" is the crop)

    Args:
        crop_meta: Crop-style metadata dict from _crop_region/_crop_search_region
        search_size: Output size (typically 256)

    Returns:
        Letterbox-style metadata dict compatible with fusion module

    Example:
        >>> crop_meta = {'crop_sz': 200, 'resize_factor': 1.28, 'pad_top': 10, ...}
        >>> lb_meta = crop_metadata_to_letterbox(crop_meta, search_size=256)
        >>> # Pass lb_meta to fusion module
    """
    # Extract crop parameters with safe defaults
    crop_sz = crop_meta.get('crop_sz', search_size)
    resize_factor = crop_meta.get('resize_factor', 1.0)

    # Padding in crop space (before resize)
    pad_left_crop = crop_meta.get('pad_left', 0)
    pad_top_crop = crop_meta.get('pad_top', 0)
    pad_right_crop = crop_meta.get('pad_right', 0)
    pad_bottom_crop = crop_meta.get('pad_bottom', 0)

    # Original image size (for reference only)
    original_size = crop_meta.get('original_size', (crop_sz, crop_sz))
    if isinstance(original_size, (list, tuple)):
        orig_h, orig_w = original_size
    else:
        orig_h = orig_w = original_size

    # Convert padding to output space (after resize)
    pad_left_out = pad_left_crop * resize_factor
    pad_top_out = pad_top_crop * resize_factor
    pad_right_out = pad_right_crop * resize_factor
    pad_bottom_out = pad_bottom_crop * resize_factor

    # Compute effective content size in output space
    # new_size = (crop_sz - padding) * resize_factor, but clamped to output_size
    content_h = crop_sz - pad_top_crop - pad_bottom_crop
    content_w = crop_sz - pad_left_crop - pad_right_crop
    new_h = int(content_h * resize_factor)
    new_w = int(content_w * resize_factor)

    # Crop position for inverse transform
    crop_x1 = crop_meta.get('crop_x1', 0)
    crop_y1 = crop_meta.get('crop_y1', 0)

    # Build letterbox-format metadata
    letterbox_meta = {
        # Core scale factors (crop to output)
        'scale': resize_factor,
        'scale_x': resize_factor,
        'scale_y': resize_factor,

        # Padding in output pixel space
        'pad_top': pad_top_out,
        'pad_left': pad_left_out,
        'pad_bottom': pad_bottom_out,
        'pad_right': pad_right_out,
        'pad_x': (pad_left_out + pad_right_out) / 2,  # Average horizontal padding
        'pad_y': (pad_top_out + pad_bottom_out) / 2,  # Average vertical padding

        # Source image dimensions (TRUE source height/width)
        'original_height': orig_h,  # TRUE source height (NOT content_h)
        'original_width': orig_w,   # TRUE source width (NOT content_w)

        # Content dimensions (for reference)
        'content_height': content_h,
        'content_width': content_w,

        # Full image original size (alias for clarity)
        'source_height': orig_h,
        'source_width': orig_w,

        # New size after scaling (content area in output space)
        'new_size': (new_h, new_w),

        # Output size for convenience
        'output_size': search_size,

        # Pass through crop metadata for inverse transform if needed
        'crop_x1': crop_x1,
        'crop_y1': crop_y1,
        'crop_sz': crop_sz,
        'resize_factor': resize_factor,
    }

    return letterbox_meta


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


def target_centered_crop(
    img: np.ndarray,
    bbox: np.ndarray,
    output_size: int = 256,
    search_area_factor: float = 4.0,
    is_rgb: bool = True
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Target-centered crop for online inference.

    CRITICAL FIX (Audit 2025121001 Issue 5):
    Implements target-centered cropping for OnlineTracker to match training distribution.
    Training uses target-centered crops with search_area_factor, but inference was using
    full-frame letterbox preprocessing - causing train/test distribution mismatch.

    This function:
    1. Computes search region centered on target bbox
    2. Handles boundary cases with symmetric padding
    3. Resizes to output_size
    4. Returns metadata and attention mask for padded regions

    Args:
        img: Input image [H, W] for grayscale or [H, W, C] for RGB
        bbox: Target bounding box in CXCYWH normalized format [cx, cy, w, h] in [0, 1]
        output_size: Output size (typically 256 for OSTrack-256)
        search_area_factor: Factor to expand target bbox for search region (default: 4.0)
        is_rgb: True for RGB images, False for sonar

    Returns:
        Tuple of (cropped_image, metadata, attention_mask):
            - cropped_image: [output_size, output_size, C] or [output_size, output_size]
            - metadata: Dict with crop transformation parameters
            - attention_mask: [output_size, output_size] binary mask, 1=valid, 0=padded

    Example:
        >>> img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> bbox = np.array([0.5, 0.5, 0.1, 0.1])  # Centered target
        >>> cropped, meta, mask = target_centered_crop(img, bbox, 256, 4.0, is_rgb=True)
        >>> print(cropped.shape)  # (256, 256, 3)
    """
    h, w = img.shape[:2]

    # Convert normalized bbox to pixel coordinates
    cx, cy, bw, bh = bbox
    cx_px = cx * w
    cy_px = cy * h
    bw_px = bw * w
    bh_px = bh * h

    # Compute search region size (based on target size and search_area_factor)
    # Following OSTrack/MixFormer convention: search_size = sqrt(target_area * search_area_factor)
    target_area = bw_px * bh_px
    search_side = np.sqrt(target_area * search_area_factor)

    # Ensure minimum search size
    search_side = max(search_side, 32)

    # Crop region (square, centered on target)
    crop_sz = int(search_side)
    x1 = int(cx_px - crop_sz / 2)
    y1 = int(cy_px - crop_sz / 2)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    # Compute padding for boundary handling
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    # Clamp crop coordinates to image bounds
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(w, x2)
    y2_clamped = min(h, y2)

    # Extract crop from image
    if is_rgb and img.ndim == 3:
        crop = img[y1_clamped:y2_clamped, x1_clamped:x2_clamped, :]
    else:
        crop = img[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

    # Create attention mask (1 for valid pixels, 0 for padded)
    # Initially all zeros, will mark valid region after padding
    mask_crop = np.ones((y2_clamped - y1_clamped, x2_clamped - x1_clamped), dtype=np.float32)

    # Apply padding if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        # Determine padding value
        if is_rgb:
            pad_value = (114, 114, 114)
        else:
            pad_value = 127

        # Pad the crop
        crop = cv2.copyMakeBorder(
            crop,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )

        # Expand mask with zeros for padded regions
        mask_crop = cv2.copyMakeBorder(
            mask_crop,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0
        )

    # Resize to output size
    resize_factor = output_size / crop_sz
    crop_resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_crop, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    # Build metadata (crop-style, can be converted to letterbox format if needed)
    metadata = {
        # Crop parameters
        'crop_x1': x1,
        'crop_y1': y1,
        'crop_sz': crop_sz,
        'resize_factor': resize_factor,

        # Padding in crop space (before resize)
        'pad_left': pad_left,
        'pad_top': pad_top,
        'pad_right': pad_right,
        'pad_bottom': pad_bottom,

        # Original image size
        'original_size': (h, w),
        'output_size': output_size,

        # For coordinate inverse transform
        'scale': resize_factor,
        'original_height': h,
        'original_width': w,

        # Target info for reference
        'target_bbox': bbox.tolist(),
        'search_area_factor': search_area_factor,
    }

    return crop_resized, metadata, mask_resized


def preprocess_target_centered(
    img: np.ndarray,
    bbox: np.ndarray,
    output_size: int = 256,
    search_area_factor: float = 4.0,
    is_rgb: bool = True,
    rgb_mean: Optional[Union[list, tuple, np.ndarray]] = None,
    rgb_std: Optional[Union[list, tuple, np.ndarray]] = None,
    sonar_mean: float = 0.5,
    sonar_std: float = 0.5,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    """
    Complete target-centered preprocessing pipeline for online inference.

    CRITICAL FIX (Audit 2025121001 Issue 5):
    Unified target-centered preprocessing that matches training pipeline:
    1. Target-centered crop with search_area_factor
    2. Normalization with config-driven parameters
    3. Returns metadata and attention mask

    Args:
        img: Input image (uint8 numpy array) [H, W, C] or [H, W]
        bbox: Target bbox in CXCYWH normalized format [cx, cy, w, h]
        output_size: Output size (256 for template, 256 for search)
        search_area_factor: Factor to expand target for search region
        is_rgb: True for RGB, False for sonar
        rgb_mean: RGB normalization mean [R, G, B]
        rgb_std: RGB normalization std [R, G, B]
        sonar_mean: Sonar normalization mean
        sonar_std: Sonar normalization std
        device: Target device ('cuda' or 'cpu')

    Returns:
        Tuple of (preprocessed_tensor, metadata, attention_mask_tensor):
            - preprocessed_tensor: [1, C, H, W] normalized tensor
            - metadata: Crop transformation metadata
            - attention_mask_tensor: [1, H, W] binary mask tensor

    Example:
        >>> tensor, meta, mask = preprocess_target_centered(
        ...     img, bbox, 256, 4.0, is_rgb=True,
        ...     rgb_mean=[0.485, 0.456, 0.406],
        ...     rgb_std=[0.229, 0.224, 0.225]
        ... )
    """
    # Channel validation
    if is_rgb:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] != 3:
            raise ValueError(f"RGB image must have 3 channels, got {img.shape[-1]}")
    else:
        if img.ndim == 3:
            img = img[:, :, 0]

    # Step 1: Target-centered crop with attention mask
    crop, metadata, mask = target_centered_crop(
        img, bbox, output_size, search_area_factor, is_rgb
    )

    # Step 2: Normalize to [0, 1]
    crop_float = crop.astype(np.float32) / 255.0

    # Step 3: Apply normalization
    if is_rgb:
        if rgb_mean is None:
            rgb_mean = [0.485, 0.456, 0.406]
        if rgb_std is None:
            rgb_std = [0.229, 0.224, 0.225]

        mean = np.array(rgb_mean).reshape(1, 1, 3)
        std = np.array(rgb_std).reshape(1, 1, 3)
        crop_normalized = (crop_float - mean) / std

        # HWC to CHW
        img_tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1).unsqueeze(0)
    else:
        crop_normalized = (crop_float - sonar_mean) / sonar_std

        # HW to 1HW
        img_tensor = torch.from_numpy(crop_normalized).unsqueeze(0).unsqueeze(0)

    # Convert mask to tensor
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

    # Move to device
    img_tensor = img_tensor.float().to(device)
    mask_tensor = mask_tensor.float().to(device)

    return img_tensor, metadata, mask_tensor
