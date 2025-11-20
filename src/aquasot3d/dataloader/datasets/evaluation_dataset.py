"""
Evaluation Dataset for Sequential Tracking

This module implements sequential evaluation datasets that iterate through
sequences frame-by-frame following standard tracking evaluation protocols
(e.g., RGBS50, VOT). Unlike training datasets that use random pair sampling,
evaluation datasets maintain temporal order for reproducible benchmarking.

Author: AquaSOT-3D Team
Date: 2025-01-07
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Optional, List
from torch.utils.data import Dataset

from .rgb_sonar_dataset import RGBSonarDataset
# Note: letterbox_resize removed - unified with RGB strategy (2025-11-19)


class EvaluationSequenceDataset(Dataset):
    """
    Sequential evaluation dataset for tracking benchmarks.

    This dataset implements the standard tracking evaluation protocol:
    - Use first valid frame as template (initialization)
    - Process all subsequent frames in sequential order
    - Maintain sequence identity and frame numbers for reproducible metrics

    Unlike the frame sequence dataset used for training (which samples
    continuous temporal windows), this dataset enforces temporal order and
    deterministic frame selection for evaluation.

    Args:
        base_dataset: RGBSonarDataset instance
        template_size: Template image size (default: 128)
        search_size: Search image size (default: 256)
        template_frame: Template frame mode:
            - 'first': Use first valid frame as template (default, RGBS50 protocol)
            - 'init': Alias for 'first'
            - int: Use specific frame ID (1-based) as template
        enable_update: Whether to update template during tracking (default: False)
        update_interval: Frames between template updates if enabled (default: 10)

    Attributes:
        sequence_name: Name of the sequence being evaluated
        num_frames: Total number of frames
        template_idx: Index of template frame (0-based)
        valid_frames: List of valid (non-occluded) frame indices

    Example:
        >>> base_ds = RGBSonarDataset(
        ...     data_root='data',
        ...     sequence_name='uuv3',
        ...     return_occluded=True
        ... )
        >>> eval_ds = EvaluationSequenceDataset(
        ...     base_dataset=base_ds,
        ...     template_frame='first'
        ... )
        >>> sample = eval_ds[0]
        >>> print(sample['sequence_name'], sample['frame_id'])
        uuv3 1
    """

    def __init__(
        self,
        base_dataset: RGBSonarDataset,
        template_size: int = 128,
        search_size: int = 256,
        template_frame: str = 'first',
        enable_update: bool = False,
        update_interval: int = 10,
    ):
        self.base_dataset = base_dataset
        self.template_size = template_size
        self.search_size = search_size
        self.enable_update = enable_update
        self.update_interval = update_interval

        # Get sequence metadata
        seq_info = base_dataset.get_sequence_info()
        self.sequence_name = seq_info['sequence_name']
        self.num_frames = base_dataset.num_frames

        # Determine template frame index (0-based)
        if isinstance(template_frame, int):
            # Explicit frame ID (convert 1-based to 0-based)
            self.template_idx = template_frame - 1
            if self.template_idx < 0 or self.template_idx >= self.num_frames:
                raise ValueError(
                    f"Template frame {template_frame} out of range [1, {self.num_frames}]"
                )
        elif template_frame in ['first', 'init']:
            # Use first valid frame
            valid_indices = base_dataset.valid_indices
            if len(valid_indices) == 0:
                raise ValueError(f"No valid frames found in sequence {self.sequence_name}")
            self.template_idx = valid_indices[0]
        else:
            raise ValueError(
                f"Invalid template_frame: {template_frame}. "
                f"Expected 'first', 'init', or integer frame ID."
            )

        # Store valid frame indices for tracking
        self.valid_frames = base_dataset.valid_indices

        # Load template once (remains fixed unless update is enabled)
        self._template_data = None
        self._load_template()

    def _load_template(self):
        """Load and cache template frame data with transformation metadata."""
        template_raw = self.base_dataset._get_frame_data(self.template_idx)

        # Prepare template images and get transformation metadata
        rgb_template, rgb_metadata = self._prepare_image(
            template_raw['rgb_img'],
            self.template_size,
            is_sonar=False
        )
        sonar_template, sonar_metadata = self._prepare_image(
            template_raw['sonar_img'],
            self.template_size,
            is_sonar=True
        )

        # Store template data with metadata
        self._template_data = {
            'rgb_template': rgb_template,
            'sonar_template': sonar_template,
            'template_bbox_rgb': template_raw['rgb_bbox'],
            'template_bbox_sonar': template_raw['sonar_bbox'],
            'template_frame_id': template_raw['frame_id'],
            'template_rgb_metadata': rgb_metadata,
            'template_sonar_metadata': sonar_metadata,
        }

    def __len__(self) -> int:
        """Return total number of frames in sequence."""
        return self.num_frames

    def __getitem__(self, index: int) -> Dict:
        """
        Get frame for evaluation.

        Args:
            index: Frame index (0-based, corresponds to position in sequence)

        Returns:
            Dictionary with keys:
                - sequence_name: Sequence identifier
                - frame_id: Frame ID (1-based)
                - rgb_template: Template RGB image (C, H, W)
                - rgb_search: Search RGB image (C, H, W)
                - sonar_template: Template sonar image (C, H, W)
                - sonar_search: Search sonar image (C, H, W)
                - template_bbox_rgb: Template RGB bbox [x, y, w, h]
                - template_bbox_sonar: Template sonar bbox [x, y, w, h]
                - search_bbox_rgb: Search RGB bbox [x, y, w, h] (ground truth)
                - search_bbox_sonar: Search sonar bbox [x, y, w, h] (ground truth)
                - rgb_boxes: Alias for search_bbox_rgb (for trainer compatibility)
                - sonar_boxes: Alias for search_bbox_sonar
                - is_valid: Whether frame has valid annotations
                - template_idx: Template frame index (0-based)
                - search_idx: Search frame index (0-based)
        """
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Index {index} out of range [0, {self.num_frames})")

        # Update template if enabled and interval reached
        if self.enable_update and index > 0 and index % self.update_interval == 0:
            # Check if previous frame was valid
            prev_data = self.base_dataset._get_frame_data(index - 1)
            if prev_data['is_valid']:
                # Update template to previous frame
                self.template_idx = index - 1
                self._load_template()

        # Get current (search) frame
        search_data = self.base_dataset._get_frame_data(index)

        # Prepare search images and get transformation metadata
        rgb_search, rgb_search_metadata = self._prepare_image(
            search_data['rgb_img'],
            self.search_size,
            is_sonar=False
        )
        sonar_search, sonar_search_metadata = self._prepare_image(
            search_data['sonar_img'],
            self.search_size,
            is_sonar=True
        )

        # Build sample dictionary with complete metadata
        sample = {
            # Sequence metadata (CRITICAL for evaluation)
            'sequence_name': self.sequence_name,
            'frame_id': search_data['frame_id'],

            # Template images and boxes (fixed or updated)
            'rgb_template': self._template_data['rgb_template'],
            'sonar_template': self._template_data['sonar_template'],
            'template_bbox_rgb': self._template_data['template_bbox_rgb'],
            'template_bbox_sonar': self._template_data['template_bbox_sonar'],

            # Search images and ground truth boxes
            'rgb_search': rgb_search,
            'sonar_search': sonar_search,
            'search_bbox_rgb': search_data['rgb_bbox'],
            'search_bbox_sonar': search_data['sonar_bbox'],

            # Aliases for trainer/evaluator compatibility
            'rgb_boxes': search_data['rgb_bbox'],
            'sonar_boxes': search_data['sonar_bbox'],

            # Validity flags
            'is_valid': search_data['is_valid'],
            'is_rgb_valid': search_data['is_rgb_valid'],
            'is_sonar_valid': search_data['is_sonar_valid'],

            # Frame indices for tracking state
            'template_idx': self.template_idx,
            'search_idx': index,

            # CRITICAL FIX (Audit 2025111409 Finding 1):
            # Transformation metadata for physical coordinate reconstruction
            # These allow the evaluator to convert predictions from the resized
            # 255x256 plane back to original image coordinates for scoring
            'rgb_search_metadata': rgb_search_metadata,
            'sonar_search_metadata': sonar_search_metadata,
            'template_rgb_metadata': self._template_data['template_rgb_metadata'],
            'template_sonar_metadata': self._template_data['template_sonar_metadata'],
        }

        return sample

    def _prepare_image(self, img, target_size: int, is_sonar: bool = False):
        """
        Prepare image for inference and return transformation metadata.

        CRITICAL FIX (Audit 2025111409 Finding 1):
        Now returns scale/padding metadata needed to reconstruct physical pixel
        coordinates during evaluation. This allows metrics to be computed in the
        original image resolution (1920x1080 for RGB, 682x376 for sonar) instead
        of the resized 255x256 plane.

        Args:
            img: Input image (numpy array, tensor, or None)
            target_size: Target image size (square)
            is_sonar: If True, prepare as 1-channel grayscale; else 3-channel RGB

        Returns:
            Tuple of (img_tensor, metadata) where:
                - img_tensor: Tensor of shape (C, target_size, target_size)
                - metadata: Dict with keys:
                    - 'original_height': Original image height
                    - 'original_width': Original image width
                    - 'scale_x': Horizontal scaling factor
                    - 'scale_y': Vertical scaling factor (may differ from scale_x for letterbox)
                    - 'pad_x': Left padding (pixels)
                    - 'pad_y': Top padding (pixels)

        Note:
            For sonar images, uses letterbox padding to preserve aspect ratio
            and maintain sensor origin at cy=H-1 (Audit 2025111408 Finding 2).
        """
        if img is None:
            # Return zeros for None images (occluded frames)
            num_channels = 1 if is_sonar else 3
            dummy_metadata = {
                'original_height': target_size,
                'original_width': target_size,
                'scale_x': 1.0,
                'scale_y': 1.0,
                'pad_x': 0,
                'pad_y': 0,
            }
            return torch.zeros(num_channels, target_size, target_size), dummy_metadata

        # Convert to numpy array if tensor
        if isinstance(img, torch.Tensor):
            # Convert CHW to HWC for letterbox_resize
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0).cpu().numpy()
            elif img.ndim == 2:
                img = img.cpu().numpy()
            # Denormalize if in [0, 1] range
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

        # Ensure numpy array
        if not isinstance(img, np.ndarray):
            raise ValueError(f"Expected numpy array or tensor, got {type(img)}")

        # Store original dimensions
        original_height, original_width = img.shape[:2]

        # Unified resize strategy for both RGB and Sonar
        # Note: Sonar now uses rectangular coordinates (682×376), no special handling needed
        import cv2
        img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Compute per-axis scaling factors
        scale_x = target_size / original_width
        scale_y = target_size / original_height
        metadata = {
            'original_height': original_height,
            'original_width': original_width,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'pad_x': 0,
            'pad_y': 0,
        }

        # Convert to tensor and normalize
        # Handle different channel formats
        if img_resized.ndim == 2:
            # Grayscale (H, W) -> (1, H, W)
            img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
        elif img_resized.ndim == 3:
            # Color (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unexpected image shape: {img_resized.shape}")

        # Normalize to [0, 1] if in [0, 255] range
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        return img_tensor, metadata

    def get_sequence_info(self) -> Dict:
        """
        Get evaluation sequence metadata.

        Returns:
            Dictionary with sequence information
        """
        return {
            'sequence_name': self.sequence_name,
            'num_frames': self.num_frames,
            'num_valid_frames': len(self.valid_frames),
            'template_frame_id': self.template_idx + 1,  # 1-based
            'template_update_enabled': self.enable_update,
            'template_update_interval': self.update_interval if self.enable_update else None,
        }