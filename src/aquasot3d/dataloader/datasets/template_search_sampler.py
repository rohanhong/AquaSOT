"""
Template-Search Pair Sampler (OSTrack-style)

This module implements the template-search paradigm for object tracking,
following the OSTrack framework. It generates paired template-search samples
for training and inference, handling:
- Template selection (first frame or historical frames)
- Search region extraction with context margin
- Coordinate normalization to search region
- Occlusion handling and last-valid-bbox tracking
- Multi-modal synchronization (RGB + Sonar)

Reference:
    OSTrack: Joint Feature Learning and Relation Modeling for Tracking
    https://github.com/botaoye/OSTrack

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from typing import Dict, List, Optional, Tuple
import random
import numpy as np
import cv2

from ...core.registry import SAMPLER_REGISTRY
from ..transforms.sonar_transforms import letterbox_resize


def expand_bbox(
    bbox: np.ndarray,
    scale: float = 2.0,
    img_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Expand bounding box by a scale factor with optional boundary clipping.

    Args:
        bbox: Bounding box [x, y, w, h]
        scale: Expansion scale (2.0 = double the area)
        img_shape: Image shape (height, width) for boundary clipping

    Returns:
        Expanded bounding box [x, y, w, h]
    """
    x, y, w, h = bbox

    # Calculate center
    cx = x + w / 2
    cy = y + h / 2

    # Calculate context margin (scale - 1) * 0.5 on each side
    context_w = w * (scale - 1) / 2
    context_h = h * (scale - 1) / 2

    # Expand bbox
    new_w = w + 2 * context_w
    new_h = h + 2 * context_h
    new_x = cx - new_w / 2
    new_y = cy - new_h / 2

    # Clip to image boundaries if provided
    if img_shape is not None:
        img_h, img_w = img_shape
        new_x = max(0, min(new_x, img_w - new_w))
        new_y = max(0, min(new_y, img_h - new_h))
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)

    return np.array([new_x, new_y, new_w, new_h], dtype=np.float32)


def crop_image(
    img: np.ndarray,
    bbox: np.ndarray,
    pad_value: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Crop image according to bounding box with padding if necessary.

    Args:
        img: Input image (H, W, C)
        bbox: Bounding box [x, y, w, h]
        pad_value: Padding value for out-of-bounds regions

    Returns:
        Cropped image region
    """
    x, y, w, h = bbox.astype(int)
    img_h, img_w = img.shape[:2]

    # Calculate crop region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    # Crop valid region
    cropped = img[y1:y2, x1:x2]

    # Handle out-of-bounds padding
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        # Need padding
        padded = np.full((int(h), int(w), img.shape[2]), pad_value, dtype=img.dtype)

        # Calculate paste position
        paste_x = max(0, -x)
        paste_y = max(0, -y)

        # Paste cropped region
        paste_h = cropped.shape[0]
        paste_w = cropped.shape[1]
        padded[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = cropped

        return padded

    return cropped


def normalize_bbox_to_crop(
    bbox: np.ndarray,
    crop_bbox: np.ndarray,
    target_size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Normalize bounding box coordinates to crop region and optionally scale to target size.

    CRITICAL FIX (Audit 2025111405 Finding #3): After crop, images are resized but
    bboxes were never scaled. This caused normalized coordinates to be wrong when
    crop size != target size.

    Args:
        bbox: Original bounding box [x, y, w, h] in image coordinates
        crop_bbox: Crop region [x, y, w, h] in image coordinates
        target_size: Optional (H, W) to scale bbox to after crop (for resize)

    Returns:
        Normalized (and optionally scaled) bounding box [x, y, w, h] in crop coordinates
    """
    crop_x, crop_y, crop_w, crop_h = crop_bbox

    # Translate bbox to crop-relative coordinates
    norm_x = bbox[0] - crop_x
    norm_y = bbox[1] - crop_y
    norm_w = bbox[2]
    norm_h = bbox[3]

    # If target_size is provided, scale bbox to match resized image
    if target_size is not None:
        target_h, target_w = target_size
        scale_x = target_w / crop_w
        scale_y = target_h / crop_h

        norm_x = norm_x * scale_x
        norm_y = norm_y * scale_y
        norm_w = norm_w * scale_x
        norm_h = norm_h * scale_y

    return np.array([norm_x, norm_y, norm_w, norm_h], dtype=np.float32)


@SAMPLER_REGISTRY.register("TemplateSearchSampler")
class TemplateSearchSampler:
    """
    Template-Search pair sampler for OSTrack-style tracking.

    This sampler generates paired template-search samples for training:
    - Template: Reference frame (typically first frame or random historical)
    - Search: Current frame cropped around previous bbox with context

    Args:
        dataset: RGBSonarDataset instance
        template_mode: Template selection mode ("first", "random_history")
        max_history_frames: Maximum history frames for random template (N)
        search_scale: Search region expansion scale (default 2.0 = 2x context)
        search_size: Target search region size after resize (H, W)
        template_size: Target template size after resize (H, W)
        enable_occlusion_handling: Whether to use last-valid-bbox for occluded frames

    Example:
        >>> sampler = TemplateSearchSampler(
        ...     dataset=dataset,
        ...     template_mode="first",
        ...     search_scale=2.0,
        ...     search_size=(256, 256),
        ...     template_size=(128, 128)
        ... )
        >>> sample = sampler.sample(frame_idx=10)
        >>> print(sample.keys())
        dict_keys(['template_rgb', 'template_sonar', 'search_rgb', 'search_sonar',
                   'template_bbox_rgb', 'template_bbox_sonar',
                   'search_bbox_rgb', 'search_bbox_sonar', ...])
    """

    def __init__(
        self,
        dataset,
        template_mode: str = "first",
        max_history_frames: int = 100,
        search_scale: float = 2.0,
        search_size: Tuple[int, int] = (256, 256),
        template_size: Tuple[int, int] = (128, 128),
        enable_occlusion_handling: bool = True,
    ):
        self.dataset = dataset
        self.search_scale = search_scale
        self.search_size = search_size
        self.template_size = template_size
        self.enable_occlusion_handling = enable_occlusion_handling

        if template_mode not in ["first", "random_history"]:
            raise ValueError(
                f"Invalid template_mode: {template_mode}. "
                "Must be 'first' or 'random_history'"
            )

        self.template_mode = template_mode
        self.max_history_frames = max_history_frames

    def _is_occluded(self, bbox: np.ndarray) -> bool:
        """Check if bbox indicates occlusion (0,0,0,0)."""
        return np.all(bbox == 0)

    def _select_template_idx(self, search_idx: int) -> int:
        """
        Select template frame index based on template_mode.

        Args:
            search_idx: Current search frame index (0-based)

        Returns:
            Template frame index (0-based)
        """
        if self.template_mode == "first":
            # Use first frame as template
            template_idx = 0

            # Ensure first frame is not occluded
            if self.enable_occlusion_handling:
                rgb_bbox = self.dataset.rgb_bboxes[template_idx]
                sonar_bbox = self.dataset.sonar_bboxes[template_idx]

                # If first frame is occluded, search for next valid frame
                while (self._is_occluded(rgb_bbox) or self._is_occluded(sonar_bbox)) \
                        and template_idx < search_idx:
                    template_idx += 1
                    rgb_bbox = self.dataset.rgb_bboxes[template_idx]
                    sonar_bbox = self.dataset.sonar_bboxes[template_idx]

            return template_idx

        else:  # random_history
            # Select random frame from recent history
            start_idx = max(0, search_idx - self.max_history_frames)
            end_idx = max(0, search_idx - 1)

            if end_idx < start_idx:
                # Not enough history, use first frame
                return 0

            # Collect valid candidate frames
            candidates = []
            for idx in range(start_idx, end_idx + 1):
                rgb_bbox = self.dataset.rgb_bboxes[idx]
                sonar_bbox = self.dataset.sonar_bboxes[idx]

                if not (self._is_occluded(rgb_bbox) or self._is_occluded(sonar_bbox)):
                    candidates.append(idx)

            if len(candidates) == 0:
                # No valid candidates, use first frame
                return 0

            # Random selection
            return random.choice(candidates)

    def _find_last_valid_bbox(
        self,
        frame_idx: int,
        modality: str = "rgb"
    ) -> Optional[np.ndarray]:
        """
        Find last valid (non-occluded) bbox before frame_idx.

        Args:
            frame_idx: Current frame index (0-based)
            modality: Which modality ("rgb" or "sonar")

        Returns:
            Last valid bbox or None if not found
        """
        bboxes = self.dataset.rgb_bboxes if modality == "rgb" else self.dataset.sonar_bboxes

        for idx in range(frame_idx - 1, -1, -1):
            bbox = bboxes[idx]
            if not self._is_occluded(bbox):
                return bbox.copy()

        return None

    def sample(self, search_idx: int) -> Optional[Dict]:
        """
        Generate template-search pair for given search frame index.

        Args:
            search_idx: Search frame index (0-based)

        Returns:
            Dictionary containing template-search pair data, or None if invalid

        Sample dict structure:
            {
                'template_rgb': np.ndarray (H_t, W_t, 3),
                'template_sonar': np.ndarray (H_t, W_t, 3),
                'template_bbox_rgb': np.ndarray [x, y, w, h],
                'template_bbox_sonar': np.ndarray [x, y, w, h],

                'search_rgb': np.ndarray (H_s, W_s, 3),
                'search_sonar': np.ndarray (H_s, W_s, 3),
                'search_bbox_rgb': np.ndarray [x, y, w, h] (normalized to search region),
                'search_bbox_sonar': np.ndarray [x, y, w, h] (normalized to search region),

                'template_idx': int (0-based),
                'search_idx': int (0-based),
                'is_valid': bool (both modalities valid),
            }
        """
        # Check search frame is within bounds
        if search_idx < 0 or search_idx >= self.dataset.num_frames:
            return None

        # Get current search frame bboxes
        rgb_bbox_search = self.dataset.rgb_bboxes[search_idx].copy()
        sonar_bbox_search = self.dataset.sonar_bboxes[search_idx].copy()

        # Check if current frame is occluded
        rgb_occluded = self._is_occluded(rgb_bbox_search)
        sonar_occluded = self._is_occluded(sonar_bbox_search)

        if (rgb_occluded or sonar_occluded) and not self.enable_occlusion_handling:
            # Skip occluded frames if occlusion handling is disabled
            return None

        # Select template frame
        template_idx = self._select_template_idx(search_idx)

        # Get template frame data
        template_data = self.dataset._get_frame_data(template_idx)
        template_rgb = template_data['rgb_img']
        template_sonar = template_data['sonar_img']
        template_bbox_rgb = template_data['rgb_bbox']
        template_bbox_sonar = template_data['sonar_bbox']

        if template_rgb is None or template_sonar is None:
            return None

        # Get search frame data
        search_data = self.dataset._get_frame_data(search_idx)
        search_rgb = search_data['rgb_img']
        search_sonar = search_data['sonar_img']

        if search_rgb is None or search_sonar is None:
            return None

        # Find previous bbox for search region (last valid if current is occluded)
        if search_idx == 0:
            prev_bbox_rgb = template_bbox_rgb
            prev_bbox_sonar = template_bbox_sonar
        else:
            if rgb_occluded:
                prev_bbox_rgb = self._find_last_valid_bbox(search_idx, "rgb")
                if prev_bbox_rgb is None:
                    prev_bbox_rgb = template_bbox_rgb
            else:
                prev_idx = search_idx - 1
                prev_bbox_rgb = self.dataset.rgb_bboxes[prev_idx]
                if self._is_occluded(prev_bbox_rgb):
                    prev_bbox_rgb = self._find_last_valid_bbox(search_idx, "rgb")
                    if prev_bbox_rgb is None:
                        prev_bbox_rgb = template_bbox_rgb

            if sonar_occluded:
                prev_bbox_sonar = self._find_last_valid_bbox(search_idx, "sonar")
                if prev_bbox_sonar is None:
                    prev_bbox_sonar = template_bbox_sonar
            else:
                prev_idx = search_idx - 1
                prev_bbox_sonar = self.dataset.sonar_bboxes[prev_idx]
                if self._is_occluded(prev_bbox_sonar):
                    prev_bbox_sonar = self._find_last_valid_bbox(search_idx, "sonar")
                    if prev_bbox_sonar is None:
                        prev_bbox_sonar = template_bbox_sonar

        # Calculate search regions (2x context around previous bbox)
        search_region_rgb = expand_bbox(
            prev_bbox_rgb, scale=self.search_scale, img_shape=search_rgb.shape[:2]
        )

        # CRITICAL FIX (Audit 2025111408 Finding 1): Do NOT crop sonar search frames
        # Sonar needs full fan to preserve sensor origin at (cx=W/2, cy=H-1) for log-polar transform
        # Only crop RGB search frame
        search_crop_rgb = crop_image(search_rgb, search_region_rgb)
        search_crop_sonar = search_sonar  # Keep full sonar frame (no crop)

        # Normalize GT bboxes to search region coordinates and scale to target size
        # CRITICAL FIX (Audit 2025111405): Pass target_size to scale bboxes after crop+resize
        if not rgb_occluded:
            search_bbox_normalized_rgb = normalize_bbox_to_crop(
                rgb_bbox_search, search_region_rgb, target_size=self.search_size
            )
        else:
            # Use zero bbox for occluded frames
            search_bbox_normalized_rgb = np.zeros(4, dtype=np.float32)

        # Sonar bbox: no crop normalization needed since we keep full frame
        # Just use original bbox (will be adjusted for letterbox padding later)
        if not sonar_occluded:
            search_bbox_normalized_sonar = sonar_bbox_search.copy()
        else:
            search_bbox_normalized_sonar = np.zeros(4, dtype=np.float32)

        # Resize to target sizes
        # RGB: direct resize (no geometric constraints)
        template_rgb_resized = cv2.resize(
            template_rgb, self.template_size[::-1], interpolation=cv2.INTER_LINEAR
        )
        search_rgb_resized = cv2.resize(
            search_crop_rgb, self.search_size[::-1], interpolation=cv2.INTER_LINEAR
        )

        # Sonar: letterbox padding to preserve aspect ratio (Audit 2025111408 Finding 2)
        # This prevents log-polar transform distortion from anisotropic warping
        template_sonar_resized, (t_sonar_scale, t_sonar_pad_y, t_sonar_pad_x) = letterbox_resize(
            template_sonar, self.template_size, fill_value=0
        )
        search_sonar_resized, (s_sonar_scale, s_sonar_pad_y, s_sonar_pad_x) = letterbox_resize(
            search_crop_sonar, self.search_size, fill_value=0
        )

        # Adjust sonar template bbox for letterbox padding
        # Original bbox [x, y, w, h] -> scaled + padded
        template_bbox_sonar_adjusted = template_bbox_sonar.copy()
        template_bbox_sonar_adjusted[0] = template_bbox_sonar[0] * t_sonar_scale + t_sonar_pad_x  # x
        template_bbox_sonar_adjusted[1] = template_bbox_sonar[1] * t_sonar_scale + t_sonar_pad_y  # y
        template_bbox_sonar_adjusted[2] = template_bbox_sonar[2] * t_sonar_scale  # w
        template_bbox_sonar_adjusted[3] = template_bbox_sonar[3] * t_sonar_scale  # h

        # Adjust sonar search bbox for letterbox padding
        # search_bbox_normalized_sonar is already in search crop coordinates
        if not sonar_occluded:
            search_bbox_normalized_sonar[0] = search_bbox_normalized_sonar[0] * s_sonar_scale + s_sonar_pad_x
            search_bbox_normalized_sonar[1] = search_bbox_normalized_sonar[1] * s_sonar_scale + s_sonar_pad_y
            search_bbox_normalized_sonar[2] = search_bbox_normalized_sonar[2] * s_sonar_scale
            search_bbox_normalized_sonar[3] = search_bbox_normalized_sonar[3] * s_sonar_scale

        # Build sample dictionary
        sample = {
            'template_rgb': template_rgb_resized,
            'template_sonar': template_sonar_resized,
            'template_bbox_rgb': template_bbox_rgb,
            'template_bbox_sonar': template_bbox_sonar_adjusted,  # Use letterbox-adjusted bbox

            'search_rgb': search_rgb_resized,
            'search_sonar': search_sonar_resized,
            'search_bbox_rgb': search_bbox_normalized_rgb,
            'search_bbox_sonar': search_bbox_normalized_sonar,  # Already adjusted above

            'search_region_rgb': search_region_rgb,
            # Note: search_region_sonar not included - sonar uses full frame (no crop)

            'template_idx': template_idx,
            'search_idx': search_idx,
            'is_rgb_valid': not rgb_occluded,
            'is_sonar_valid': not sonar_occluded,
            'is_valid': not (rgb_occluded or sonar_occluded),
        }

        return sample

    def __call__(self, search_idx: int) -> Optional[Dict]:
        """Callable interface for sampling."""
        return self.sample(search_idx)