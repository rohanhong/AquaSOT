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

from .rgb_sonar_dataset import RGBSonarDataset


class TrackingDataset(Dataset):
    """
    Dataset wrapper that samples template-search pairs from sequences.

    For tracking, we need pairs of frames:
    - Template: Initial target appearance (frame t)
    - Search: Frame to find target in (frame t+Δt)

    Args:
        base_dataset: RGBSonarDataset instance
        template_search_gap: Maximum frame gap between template and search (default: 30)
        min_gap: Minimum frame gap (default: 1)
        search_first_prob: Probability of using first frame as template (default: 0.2)
        template_size: Template image size (default: 127)
        search_size: Search image size (default: 255)
    """

    def __init__(
        self,
        base_dataset: RGBSonarDataset,
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

        # Get valid frame indices
        self.valid_indices = base_dataset.valid_indices
        self.num_frames = len(self.valid_indices)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index: int) -> Dict:
        """
        Sample a template-search pair.

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

        # Get frame data
        template_data = self.base_dataset[template_idx]
        search_data = self.base_dataset[search_idx]

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

        # Build tracking pair
        pair = {
            # Images (convert HWC to CHW if needed and resize)
            'rgb_template': self._prepare_image(template_data['rgb_img'], self.template_size, is_sonar=False),
            'rgb_search': self._prepare_image(search_data['rgb_img'], self.search_size, is_sonar=False),
            'sonar_template': self._prepare_image(template_data['sonar_img'], self.template_size, is_sonar=True),
            'sonar_search': self._prepare_image(search_data['sonar_img'], self.search_size, is_sonar=True),

            # Bounding boxes (original keys)
            'template_bbox_rgb': template_data['rgb_bbox'],
            'template_bbox_sonar': template_data['sonar_bbox'],
            'search_bbox_rgb': search_data['rgb_bbox'],
            'search_bbox_sonar': search_data['sonar_bbox'],

            # Bounding boxes (trainer-expected keys for loss computation)
            'rgb_boxes': search_data['rgb_bbox'],  # Alias for search_bbox_rgb
            'sonar_boxes': search_data['sonar_bbox'],  # Alias for search_bbox_sonar

            # Metadata
            'template_idx': template_idx,
            'search_idx': search_idx,
            'is_valid': template_data['is_valid'] and search_data['is_valid'],
        }

        return pair

    def _prepare_image(self, img, target_size, is_sonar=False):
        """
        Prepare image for training.

        Converts HWC numpy array to CHW tensor and resizes to target_size.

        Args:
            img: Input image (numpy array or tensor)
            target_size: Target image size (square)
            is_sonar: If True, prepare as 1-channel grayscale; if False, as 3-channel RGB

        Returns:
            Tensor of shape (C, target_size, target_size) where C=1 for sonar, C=3 for RGB
        """
        if img is None:
            # Return zeros for None images (occluded frames)
            num_channels = 1 if is_sonar else 3
            return torch.zeros(num_channels, target_size, target_size)

        if isinstance(img, np.ndarray):
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
            elif img.ndim == 3 and img.shape[0] not in [1, 3]:
                img = img.permute(2, 0, 1)

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

        return img