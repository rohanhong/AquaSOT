"""
Tracking Dataset for Template-Search Pair Sampling

This module wraps RGBSonarDataset to create template-search pairs for
tracking training following Siamese network paradigm.

Author: AquaSOT-3D Team
Date: 2025-11-11
"""

import random
from typing import Dict, Optional
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from .rgb_sonar_dataset import RGBSonarDataset, MultiSequenceRGBSonarDataset


class TrackingDataset(Dataset):
    """
    Dataset wrapper that samples template-search pairs from sequences.

    For tracking, we need pairs of frames:
    - Template: Initial target appearance (frame t)
    - Search: Frame to find target in (frame t+Δt)

    This dataset supports both single-sequence (RGBSonarDataset) and multi-sequence
    (MultiSequenceRGBSonarDataset) base datasets. For multi-sequence datasets, it
    enforces intra-sequence temporal constraints to prevent template/search pairs
    from crossing sequence boundaries.

    Args:
        base_dataset: RGBSonarDataset or MultiSequenceRGBSonarDataset instance
        template_search_gap: Maximum frame gap between template and search (default: 30)
        min_gap: Minimum frame gap (default: 1)
        search_first_prob: Probability of using first frame as template (default: 0.2)
        template_size: Template image size (default: 127)
        search_size: Search image size (default: 255)
    """

    def __init__(
        self,
        base_dataset,
        template_search_gap: int = 30,
        min_gap: int = 1,
        search_first_prob: float = 0.2,
        template_size: int = 127,
        search_size: int = 255,
        augmentation=None,
    ):
        self.base_dataset = base_dataset
        self.template_search_gap = template_search_gap
        self.min_gap = min_gap
        self.search_first_prob = search_first_prob
        self.template_size = template_size
        self.search_size = search_size
        self.augmentation = augmentation

        # Check if base_dataset is multi-sequence
        self.is_multi_sequence = isinstance(base_dataset, MultiSequenceRGBSonarDataset)

        # Build sequence-aware index mapping
        if self.is_multi_sequence:
            self._build_sequence_index_mapping()
        else:
            # Single sequence case
            self.valid_indices = base_dataset.valid_indices
            self.num_frames = len(self.valid_indices)
            self.sequence_ranges = None

    def _build_sequence_index_mapping(self):
        """
        Build mapping for multi-sequence dataset to ensure temporal constraints.

        Creates a mapping of global indices to their sequence boundaries,
        preventing template/search pairs from crossing sequence boundaries.
        """
        self.sequence_ranges = {}
        self.num_frames = len(self.base_dataset)

        for seq_metadata in self.base_dataset.sequences:
            seq_name = seq_metadata.sequence_name
            self.sequence_ranges[seq_name] = {
                'start': seq_metadata.start_idx,
                'end': seq_metadata.end_idx,
                'num_frames': seq_metadata.end_idx - seq_metadata.start_idx,
            }

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index: int) -> Dict:
        """
        Sample a template-search pair.

        For multi-sequence datasets, ensures template and search frames
        come from the same sequence to maintain temporal consistency.

        Returns:
            Dictionary with keys:
                - rgb_template: Template RGB image, shape (C, H, W)
                - rgb_search: Search RGB image, shape (C, H, W)
                - sonar_template: Template sonar image, shape (C, H, W)
                - sonar_search: Search sonar image, shape (C, H, W)
                - template_bbox_rgb: Template RGB bbox [x, y, w, h]
                - template_bbox_sonar: Template sonar bbox [x, y, w, h]
                - search_bbox_rgb: Search RGB bbox [x, y, w, h]
                - search_bbox_sonar: Search sonar bbox [x, y, w, h]
                - template_idx: Template frame index
                - search_idx: Search frame index
                - sequence_name: Name of the sequence
                - template_frame_id: 1-based template frame ID
                - search_frame_id: 1-based search frame ID
        """
        if self.is_multi_sequence:
            # Multi-sequence case: enforce intra-sequence temporal constraints
            template_idx, search_idx = self._sample_intra_sequence_pair(index)
        else:
            # Single-sequence case: use original logic
            template_idx, search_idx = self._sample_single_sequence_pair(index)

        # Get frame data
        template_data = self.base_dataset[template_idx]
        search_data = self.base_dataset[search_idx]

        # Verify sequence consistency for multi-sequence
        if self.is_multi_sequence:
            template_seq = template_data.get('sequence_name')
            search_seq = search_data.get('sequence_name')
            if template_seq != search_seq:
                raise RuntimeError(
                    f"Template-search sequence mismatch: {template_seq} != {search_seq}. "
                    "This indicates a bug in _sample_intra_sequence_pair."
                )

        # Apply augmentation if provided
        if self.augmentation is not None:
            # Prepare data for augmentation (expects dict with rgb_img, sonar_img, rgb_bbox, sonar_bbox)
            template_aug_input = {
                'rgb_img': template_data['rgb_img'],
                'sonar_img': template_data['sonar_img'],
                'rgb_bbox': template_data['rgb_bbox'],
                'sonar_bbox': template_data['sonar_bbox'],
            }
            search_aug_input = {
                'rgb_img': search_data['rgb_img'],
                'sonar_img': search_data['sonar_img'],
                'rgb_bbox': search_data['rgb_bbox'],
                'sonar_bbox': search_data['sonar_bbox'],
            }

            # Apply augmentation
            template_aug_output = self.augmentation(template_aug_input)
            search_aug_output = self.augmentation(search_aug_input)

            # Update template and search data with augmented results
            template_data['rgb_img'] = template_aug_output.get('rgb_img', template_data['rgb_img'])
            template_data['sonar_img'] = template_aug_output.get('sonar_img', template_data['sonar_img'])
            template_data['rgb_bbox'] = template_aug_output.get('rgb_bbox', template_data['rgb_bbox'])
            template_data['sonar_bbox'] = template_aug_output.get('sonar_bbox', template_data['sonar_bbox'])

            search_data['rgb_img'] = search_aug_output.get('rgb_img', search_data['rgb_img'])
            search_data['sonar_img'] = search_aug_output.get('sonar_img', search_data['sonar_img'])
            search_data['rgb_bbox'] = search_aug_output.get('rgb_bbox', search_data['rgb_bbox'])
            search_data['sonar_bbox'] = search_aug_output.get('sonar_bbox', search_data['sonar_bbox'])

        # Get sequence metadata from base dataset
        if self.is_multi_sequence:
            sequence_name = template_data.get('sequence_name')
        else:
            seq_info = self.base_dataset.get_sequence_info()
            sequence_name = seq_info['sequence_name']

        # Prepare images and get per-axis scale factors
        rgb_template, rgb_template_scale_x, rgb_template_scale_y = self._prepare_image(template_data['rgb_img'], self.template_size, is_sonar=False)
        rgb_search, rgb_search_scale_x, rgb_search_scale_y = self._prepare_image(search_data['rgb_img'], self.search_size, is_sonar=False)
        sonar_template, sonar_template_scale_x, sonar_template_scale_y = self._prepare_image(template_data['sonar_img'], self.template_size, is_sonar=True)
        sonar_search, sonar_search_scale_x, sonar_search_scale_y = self._prepare_image(search_data['sonar_img'], self.search_size, is_sonar=True)

        # Scale bounding boxes to match resized images with per-axis scaling
        # CRITICAL FIX (Audit 2025111405 Requirement #1): Use separate scale_x and scale_y
        template_bbox_rgb_scaled = self._scale_bbox(template_data['rgb_bbox'], rgb_template_scale_x, rgb_template_scale_y)
        template_bbox_sonar_scaled = self._scale_bbox(template_data['sonar_bbox'], sonar_template_scale_x, sonar_template_scale_y)
        search_bbox_rgb_scaled = self._scale_bbox(search_data['rgb_bbox'], rgb_search_scale_x, rgb_search_scale_y)
        search_bbox_sonar_scaled = self._scale_bbox(search_data['sonar_bbox'], sonar_search_scale_x, sonar_search_scale_y)

        # Build tracking pair
        pair = {
            # Images (resized tensors)
            'rgb_template': rgb_template,
            'rgb_search': rgb_search,
            'sonar_template': sonar_template,
            'sonar_search': sonar_search,

            # Bounding boxes (scaled to match resized images)
            'template_bbox_rgb': template_bbox_rgb_scaled,
            'template_bbox_sonar': template_bbox_sonar_scaled,
            'search_bbox_rgb': search_bbox_rgb_scaled,
            'search_bbox_sonar': search_bbox_sonar_scaled,

            # Bounding boxes (trainer-expected keys for loss computation)
            'rgb_boxes': search_bbox_rgb_scaled,  # Alias for search_bbox_rgb
            'sonar_boxes': search_bbox_sonar_scaled,  # Alias for search_bbox_sonar

            # Sequence metadata (required for evaluation)
            'sequence_name': sequence_name,
            'template_frame_id': template_data['frame_id'],
            'search_frame_id': search_data['frame_id'],
            'frame_id': search_data['frame_id'],  # Alias for search_frame_id

            # Metadata
            'template_idx': template_idx,
            'search_idx': search_idx,
            'is_valid': template_data['is_valid'] and search_data['is_valid'],
        }

        return pair

    def _sample_single_sequence_pair(self, index: int):
        """
        Sample template-search pair for single-sequence dataset.

        Args:
            index: Current sample index

        Returns:
            Tuple of (template_idx, search_idx)
        """
        # Decide template frame
        if random.random() < self.search_first_prob:
            # Use first valid frame as template
            template_idx = 0
        else:
            # Random template from current index vicinity
            template_idx = index

        # Sample search frame with gap
        max_search_idx = min(
            template_idx + self.template_search_gap,
            self.num_frames - 1
        )
        min_search_idx = min(
            template_idx + self.min_gap,
            self.num_frames - 1
        )

        # Ensure valid range
        if min_search_idx > max_search_idx:
            search_idx = template_idx
        else:
            search_idx = random.randint(min_search_idx, max_search_idx)

        return template_idx, search_idx

    def _sample_intra_sequence_pair(self, index: int):
        """
        Sample template-search pair for multi-sequence dataset with intra-sequence constraints.

        Ensures both template and search frames come from the same sequence.

        Args:
            index: Global frame index

        Returns:
            Tuple of (template_idx, search_idx) where both are in the same sequence
        """
        # Get sequence name for this index
        sequence_name = self.base_dataset.get_frame_sequence(index)
        seq_range = self.sequence_ranges[sequence_name]

        # Convert to local index within sequence
        local_index = index - seq_range['start']

        # Decide template frame (local)
        if random.random() < self.search_first_prob:
            # Use first frame of this sequence as template
            local_template_idx = 0
        else:
            # Use current frame as template
            local_template_idx = local_index

        # Sample search frame with gap (ensuring it stays within sequence)
        local_max_search_idx = min(
            local_template_idx + self.template_search_gap,
            seq_range['num_frames'] - 1
        )
        local_min_search_idx = min(
            local_template_idx + self.min_gap,
            seq_range['num_frames'] - 1
        )

        # Ensure valid range
        if local_min_search_idx > local_max_search_idx:
            local_search_idx = local_template_idx
        else:
            local_search_idx = random.randint(local_min_search_idx, local_max_search_idx)

        # Convert back to global indices
        template_idx = seq_range['start'] + local_template_idx
        search_idx = seq_range['start'] + local_search_idx

        # Sanity check: ensure indices are within sequence boundaries
        assert seq_range['start'] <= template_idx < seq_range['end'], \
            f"Template index {template_idx} out of sequence range [{seq_range['start']}, {seq_range['end']})"
        assert seq_range['start'] <= search_idx < seq_range['end'], \
            f"Search index {search_idx} out of sequence range [{seq_range['start']}, {seq_range['end']})"

        return template_idx, search_idx

    def _prepare_image(self, img, target_size, is_sonar=False):
        """
        Prepare image for training.

        Converts HWC numpy array to CHW tensor and resizes to target_size.

        Args:
            img: Input image (numpy array or tensor)
            target_size: Target image size (square)
            is_sonar: If True, prepare as 1-channel grayscale; if False, as 3-channel RGB

        Returns:
            Tuple of (resized_tensor, scale_x, scale_y):
                - resized_tensor: Tensor of shape (C, target_size, target_size) where C=1 for sonar, C=3 for RGB
                - scale_x: float, x-axis scale factor (target_size / original_width)
                - scale_y: float, y-axis scale factor (target_size / original_height)

        Note:
            CRITICAL FIX (Audit 2025111405 Requirement #1): Returns separate per-axis scale factors
            instead of averaged scale_factor. This prevents geometry distortion when resizing
            non-square images (e.g., 1920x1080 -> 255x255). Using averaged scale would cause:
            - x-coords over-scaled by ~39% (0.1845/0.1328)
            - y-coords under-scaled by ~22% (0.1845/0.2361)
            leading to incorrect normalized bbox labels.
        """
        if img is None:
            # Return zeros for None images (occluded frames)
            num_channels = 1 if is_sonar else 3
            return torch.zeros(num_channels, target_size, target_size), 1.0, 1.0

        # Store original size for scale factor calculation
        if isinstance(img, np.ndarray):
            original_h, original_w = img.shape[:2]
            # Convert HWC to CHW
            if img.ndim == 3 and img.shape[2] in [1, 3]:
                img = img.transpose(2, 0, 1)
            elif img.ndim == 2:
                # Handle 2D grayscale images (H, W) -> (1, H, W)
                img = img[np.newaxis, :, :]
            # Convert to tensor
            img = torch.from_numpy(img).float()
        elif isinstance(img, torch.Tensor):
            # Already tensor, ensure CHW format
            if img.ndim == 2:
                # Handle 2D grayscale tensors (H, W) -> (1, H, W)
                img = img.unsqueeze(0)
                original_h, original_w = img.shape[1], img.shape[2]
            elif img.ndim == 3 and img.shape[0] not in [1, 3]:
                original_h, original_w = img.shape[0], img.shape[1]
                img = img.permute(2, 0, 1)
            else:
                original_h, original_w = img.shape[1], img.shape[2]

        # Normalize to [0, 1] if in [0, 255] range
        if img.max() > 1.0:
            img = img / 255.0

        # Resize to target size using bilinear interpolation
        # Add batch dimension, resize, remove batch dimension
        img = F.interpolate(
            img.unsqueeze(0),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Calculate per-axis scale factors (CRITICAL: separate x and y scales)
        # For non-square images resized to square (e.g., 1920x1080 -> 255x255):
        # scale_x = 255/1920 = 0.1328, scale_y = 255/1080 = 0.2361
        scale_x = target_size / original_w
        scale_y = target_size / original_h

        return img, scale_x, scale_y

    def _scale_bbox(self, bbox, scale_x, scale_y):
        """
        Scale bounding box coordinates after image resize with per-axis scaling.

        CRITICAL FIX (Audit 2025111405 Requirement #1): Uses separate scale factors
        for x and y axes to maintain correct geometry when resizing non-square images
        to square targets. Previous implementation used averaged scale_factor which
        caused ~28% distortion in both axes.

        Args:
            bbox: Bounding box in [x, y, w, h] format (pixel coordinates)
            scale_x: X-axis scale factor (target_width / original_width)
            scale_y: Y-axis scale factor (target_height / original_height)

        Returns:
            Scaled bounding box in [x, y, w, h] format (resized pixel coordinates)

        Example:
            >>> # Original 1920x1080 image with bbox [1000, 500, 200, 100]
            >>> # Resized to 255x255
            >>> scale_x = 255 / 1920  # 0.1328
            >>> scale_y = 255 / 1080  # 0.2361
            >>> scaled = _scale_bbox([1000, 500, 200, 100], scale_x, scale_y)
            >>> # Result: [132.8, 118.0, 26.6, 23.6] (correct geometry)
            >>> # vs. averaged scale (0.1845): [184.5, 92.3, 36.9, 18.5] (distorted)
        """
        if bbox is None or len(bbox) != 4:
            return bbox

        # Apply per-axis scaling to maintain correct geometry
        x, y, w, h = bbox
        return np.array([
            x * scale_x,  # Scale x-coordinate by x-axis factor
            y * scale_y,  # Scale y-coordinate by y-axis factor
            w * scale_x,  # Scale width by x-axis factor
            h * scale_y   # Scale height by y-axis factor
        ], dtype=np.float32)