"""
Evaluation Dataset for Sequential Tracking

This module implements sequential evaluation datasets that iterate through
sequences frame-by-frame following standard tracking evaluation protocols
(e.g., RGBS50, VOT). Unlike training datasets that use random pair sampling,
evaluation datasets maintain temporal order for reproducible benchmarking.

CRITICAL FIX (Audit 2025112602): Updated for target-centered search cropping
consistency with training. Both training and evaluation now use OSTrack-style
target-centered cropping with search_area_factor.

Author: AquaSOT-3D Team
Date: 2025-01-07
Updated: 2025-11-26 (Audit 2025112602 - Target-centered search cropping)
"""

import math
import torch
import numpy as np
import cv2
import logging
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from torch.utils.data import Dataset

from .rgb_sonar_dataset import RGBSonarDataset
# Note: letterbox_resize removed - unified with RGB strategy (2025-11-19)

logger = logging.getLogger(__name__)


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
        template_size: Template image size (default: 128, OSTrack-256 standard)
        search_size: Search image size (default: 256, OSTrack-256 standard)
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
        template_size: int = 128,  # OSTrack-256 standard
        search_size: int = 256,    # OSTrack-256 standard
        template_frame: str = 'first',
        enable_update: bool = False,
        update_interval: int = 10,
        normalization_config: Optional[Dict] = None,
        template_visibility_policy: str = 'any',
        context_factor: float = 2.0,
        # CRITICAL FIX (Audit 2025112602): Target-centered search cropping
        search_area_factor: float = 4.0,      # Search = sqrt(w*h) * 4.0
        template_area_factor: float = 2.0,    # Template = sqrt(w*h) * 2.0
        use_attention_mask: bool = True,
    ):
        self.base_dataset = base_dataset
        self.template_size = template_size
        self.search_size = search_size
        self.enable_update = enable_update
        self.update_interval = update_interval
        self.context_factor = context_factor

        # CRITICAL FIX (Audit 2025112602): Target-centered cropping parameters
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.use_attention_mask = use_attention_mask

        # CRITICAL FIX (Audit 2025112402 Finding 3):
        # Load normalization parameters from config instead of hardcoding
        if normalization_config is None:
            normalization_config = {
                'rgb_mean': [0.485, 0.456, 0.406],
                'rgb_std': [0.229, 0.224, 0.225],
                'sonar_mean': [0.5],
                'sonar_std': [0.5]
            }

        self.rgb_mean = torch.tensor(normalization_config.get('rgb_mean', [0.485, 0.456, 0.406])).view(3, 1, 1)
        self.rgb_std = torch.tensor(normalization_config.get('rgb_std', [0.229, 0.224, 0.225])).view(3, 1, 1)
        self.sonar_mean = torch.tensor(normalization_config.get('sonar_mean', [0.5])).view(1, 1, 1)
        self.sonar_std = torch.tensor(normalization_config.get('sonar_std', [0.5])).view(1, 1, 1)

        # Get sequence metadata
        seq_info = base_dataset.get_sequence_info()
        self.sequence_name = seq_info['sequence_name']
        self.num_frames = base_dataset.num_frames

        # CRITICAL FIX (Audit 2025112504 Finding 2):
        # Build visibility-aware template candidate list to prevent 0000 template selection
        # Even if base_dataset uses visibility_policy='all' (includes all frames for evaluation),
        # template selection requires at least one modality visible
        self.template_visibility_policy = template_visibility_policy
        self.template_candidates = self._build_template_candidates(
            base_dataset, template_visibility_policy
        )

        if len(self.template_candidates) == 0:
            raise ValueError(
                f"No valid template candidates found in sequence {self.sequence_name} "
                f"with template_visibility_policy='{template_visibility_policy}'. "
                f"This sequence may start with zero-modality frames (both RGB and Sonar 0000)."
            )

        # Determine template frame index (0-based)
        if isinstance(template_frame, int):
            # Explicit frame ID (convert 1-based to 0-based)
            requested_idx = template_frame - 1
            if requested_idx < 0 or requested_idx >= self.num_frames:
                raise ValueError(
                    f"Template frame {template_frame} out of range [1, {self.num_frames}]"
                )

            # Validate that requested frame satisfies visibility policy
            if requested_idx not in self.template_candidates:
                logger.warning(
                    f"Template frame {template_frame} does not satisfy "
                    f"template_visibility_policy='{template_visibility_policy}'. "
                    f"Using first valid candidate (frame {self.template_candidates[0] + 1}) instead."
                )
                self.template_idx = self.template_candidates[0]
            else:
                self.template_idx = requested_idx
        elif template_frame in ['first', 'init']:
            # Use first valid template candidate (guaranteed to have at least one modality visible)
            self.template_idx = self.template_candidates[0]
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

    def _build_template_candidates(
        self,
        base_dataset: 'RGBSonarDataset',
        policy: str
    ) -> List[int]:
        """
        Build list of valid template candidate frame indices.

        CRITICAL FIX (Audit 2025112504 Finding 2):
        Template frames must satisfy visibility requirements to prevent 0000 templates.
        Even if base_dataset uses 'all' policy for evaluation, template selection
        requires at least one modality visible.

        Args:
            base_dataset: Base RGBSonarDataset
            policy: Visibility policy ('any', 'both', 'all')
                - 'any': At least one modality visible (recommended for template selection)
                - 'both': Both modalities visible (strict dual-modality requirement)
                - 'all': No restriction (not recommended for template selection)

        Returns:
            List of frame indices (0-based) that satisfy visibility policy
        """
        from .rgb_sonar_dataset import is_occluded

        candidates = []

        for idx in range(base_dataset.num_frames):
            rgb_bbox = base_dataset.rgb_bboxes[idx]
            rgb_valid = not is_occluded(rgb_bbox)

            sonar_bbox = base_dataset.sonar_bboxes[idx]
            sonar_valid = not is_occluded(sonar_bbox)

            if policy == 'all':
                # Include all frames (not recommended for template selection)
                include = True
            elif policy == 'any':
                # At least one modality visible (recommended for template selection)
                include = rgb_valid or sonar_valid
            elif policy == 'both':
                # Both modalities visible (strict requirement)
                include = rgb_valid and sonar_valid
            else:
                raise ValueError(
                    f"Invalid template_visibility_policy: '{policy}'. "
                    f"Must be 'all', 'any', or 'both'."
                )

            if include:
                candidates.append(idx)

        return candidates

    def _load_template(self):
        """
        Load and cache template frame data with bbox-based cropping.

        CRITICAL FIX (Audit 2025112601):
        Template must be cropped around target bbox (same as training),
        NOT full-frame letterbox resize. This follows mainstream SOT methods
        (OSTrack, TransT, MixFormer) where template is a target-centered crop.
        """
        template_raw = self.base_dataset._get_frame_data(self.template_idx)

        # CRITICAL FIX: Use bbox-based cropping (same as training)
        # Crop template around target with context factor
        rgb_template, rgb_metadata = self._crop_template(
            template_raw['rgb_img'],
            template_raw['rgb_bbox'],
            self.template_size,
            is_sonar=False
        )
        sonar_template, sonar_metadata = self._crop_template(
            template_raw['sonar_img'],
            template_raw['sonar_bbox'],
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

    def _crop_template(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        crop_size: int,
        is_sonar: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Crop template region around target bbox using OSTrack methodology.

        CRITICAL FIX (Audit 2025112602): Consistent with training dataset.
        Uses sqrt(w*h) * template_area_factor instead of max(w,h) * context_factor.
        Uses boundary padding instead of clamping.

        Args:
            image: Input image [H, W, C] or [H, W] numpy array
            bbox: [x, y, w, h] in XYWH pixel format
            crop_size: Target crop size (e.g., 256)
            is_sonar: If True, process as single-channel sonar

        Returns:
            Tuple of (crop_tensor, crop_metadata)
        """
        # Handle None images (occluded frames)
        if image is None:
            num_channels = 1 if is_sonar else 3
            dummy_metadata = {
                'crop_x1': 0, 'crop_y1': 0,
                'crop_sz': crop_size, 'resize_factor': 1.0,
                'pad_left': 0, 'pad_top': 0,
                # Legacy keys for backward compatibility
                'crop_width': crop_size, 'crop_height': crop_size,
                'letterbox_scale': 1.0,
                'letterbox_pad_left': 0, 'letterbox_pad_top': 0,
                'original_height': crop_size, 'original_width': crop_size,
            }
            return torch.zeros(num_channels, crop_size, crop_size), dummy_metadata

        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0).cpu().numpy()
            elif image.dim() == 2:
                image = image.cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

        image_np = image if isinstance(image, np.ndarray) else np.array(image)

        # Ensure uint8 format
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        H, W = image_np.shape[:2]
        x, y, w, h = bbox

        # Handle degenerate bbox
        if w <= 0 or h <= 0:
            w = max(w, 10)
            h = max(h, 10)

        # Compute bbox center
        cx = x + w / 2
        cy = y + h / 2

        # CRITICAL FIX (Audit 2025112602):
        # Use OSTrack formula: sqrt(w*h) * template_area_factor
        # NOT: max(w,h) * context_factor (inconsistent with search)
        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * self.template_area_factor))
        crop_sz = max(crop_sz, 1)

        # Compute crop boundaries (may extend beyond image)
        x1 = int(round(cx - crop_sz / 2))
        y1 = int(round(cy - crop_sz / 2))
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        # CRITICAL FIX: Compute padding amounts for boundary handling (NOT clamping!)
        x1_pad = max(0, -x1)
        x2_pad = max(0, x2 - W)
        y1_pad = max(0, -y1)
        y2_pad = max(0, y2 - H)

        # Clamp to valid image region
        x1_valid = max(0, x1)
        x2_valid = min(W, x2)
        y1_valid = max(0, y1)
        y2_valid = min(H, y2)

        # Extract valid region and pad
        if x2_valid <= x1_valid or y2_valid <= y1_valid:
            # Completely out of bounds
            if image_np.ndim == 2:
                crop_padded = np.full((crop_sz, crop_sz), 127, dtype=np.uint8)
            else:
                avg_color = image_np.mean(axis=(0, 1)).astype(np.uint8)
                crop_padded = np.full((crop_sz, crop_sz, image_np.shape[2]), avg_color, dtype=np.uint8)
        else:
            crop = image_np[y1_valid:y2_valid, x1_valid:x2_valid]

            # Compute padding color (image mean)
            if image_np.ndim == 2:
                pad_value = int(image_np.mean())
            else:
                pad_value = tuple(int(c) for c in image_np.mean(axis=(0, 1)))

            # Pad to full crop size
            crop_padded = cv2.copyMakeBorder(
                crop,
                y1_pad, y2_pad, x1_pad, x2_pad,
                cv2.BORDER_CONSTANT,
                value=pad_value
            )

        # Resize to crop_size
        resize_factor = crop_size / crop_sz
        crop_resized = cv2.resize(
            crop_padded,
            (crop_size, crop_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Store metadata (consistent with training)
        crop_metadata = {
            'crop_x1': x1,
            'crop_y1': y1,
            'crop_sz': crop_sz,
            'resize_factor': resize_factor,
            'pad_left': x1_pad,
            'pad_top': y1_pad,
            # Legacy keys for backward compatibility
            'crop_width': crop_sz,
            'crop_height': crop_sz,
            'letterbox_scale': resize_factor,
            'letterbox_pad_left': int(x1_pad * resize_factor),
            'letterbox_pad_top': int(y1_pad * resize_factor),
            'original_height': H,
            'original_width': W,
        }

        # Convert to tensor and normalize
        if crop_resized.ndim == 2:
            crop_tensor = torch.from_numpy(crop_resized).unsqueeze(0).float() / 255.0
            crop_tensor = (crop_tensor - self.sonar_mean) / self.sonar_std
        else:
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            crop_tensor = (crop_tensor - self.rgb_mean) / self.rgb_std

        return crop_tensor, crop_metadata

    def _letterbox(self, img: np.ndarray, target_size: int) -> Tuple[np.ndarray, Dict]:
        """
        Letterbox preprocessing: resize to fit within target_size and pad to square.

        Args:
            img: Input image [H, W] or [H, W, C]
            target_size: Target square size

        Returns:
            Tuple of (letterboxed_image, metadata)
        """
        h, w = img.shape[:2]
        is_single_channel = img.ndim == 3 and img.shape[2] == 1

        # Compute scale factor
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        if is_single_channel:
            img_to_resize = img.squeeze(-1)
        else:
            img_to_resize = img

        img_resized = cv2.resize(img_to_resize, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Pad with gray
        if img_resized.ndim == 2:
            pad_value = 127
        else:
            pad_value = (114, 114, 114)

        img_letterboxed = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_value
        )

        metadata = {
            'scale': scale,
            'pad_left': left,
            'pad_top': top,
            'original_size': (h, w),
            'new_size': (new_h, new_w)
        }

        return img_letterboxed, metadata

    def _crop_search_region(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        output_size: int,
        area_factor: float,
        is_sonar: bool = False
    ) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        """
        Crop search region centered on target bbox following OSTrack methodology.

        CRITICAL FIX (Audit 2025112602): Consistent with training dataset.
        For evaluation, we use GT bbox as crop center (offline evaluation mode).
        For online tracking, use OnlineTracker which uses predicted bbox.

        Args:
            image: Input image [H, W, C] or [H, W] numpy array
            bbox: Target bbox [x, y, w, h] in XYWH pixel format
            output_size: Output crop size (e.g., 256)
            area_factor: Crop size multiplier (e.g., 4.0 for search)
            is_sonar: Whether this is a sonar image

        Returns:
            Tuple of (crop_tensor, attention_mask, metadata)
        """
        if image is None:
            num_channels = 1 if is_sonar else 3
            crop_tensor = torch.zeros(num_channels, output_size, output_size)
            # CRITICAL FIX (Audit 2025112603): None image should have all-ZERO mask
            # (all invalid), not all-ones. This prevents model from attending to fake pixels.
            att_mask = np.zeros((output_size, output_size), dtype=np.float32)
            metadata = {
                'crop_center': (0, 0), 'crop_sz': output_size,
                'resize_factor': 1.0, 'crop_x1': 0, 'crop_y1': 0,
                'pad_left': 0, 'pad_top': 0,
                'original_size': (output_size, output_size),
                'output_size': output_size
            }
            return crop_tensor, att_mask, metadata

        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0).cpu().numpy()
            elif image.dim() == 2:
                image = image.cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

        H, W = image.shape[:2]
        x, y, w, h = bbox

        # Handle degenerate bbox
        if w <= 0 or h <= 0:
            w = max(w, 10)
            h = max(h, 10)

        # Compute bbox center
        cx = x + w / 2
        cy = y + h / 2

        # Compute crop size: sqrt(w*h) * area_factor
        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * area_factor))
        crop_sz = max(crop_sz, 1)

        # Compute crop boundaries
        x1 = int(round(cx - crop_sz / 2))
        y1 = int(round(cy - crop_sz / 2))
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        # Compute padding amounts
        x1_pad = max(0, -x1)
        x2_pad = max(0, x2 - W)
        y1_pad = max(0, -y1)
        y2_pad = max(0, y2 - H)

        # Clamp to valid region
        x1_valid = max(0, x1)
        x2_valid = min(W, x2)
        y1_valid = max(0, y1)
        y2_valid = min(H, y2)

        # Extract valid region
        if x2_valid <= x1_valid or y2_valid <= y1_valid:
            if image.ndim == 2:
                crop_padded = np.full((crop_sz, crop_sz), 127, dtype=np.uint8)
            else:
                avg_color = image.mean(axis=(0, 1)).astype(np.uint8)
                crop_padded = np.full((crop_sz, crop_sz, image.shape[2]), avg_color, dtype=np.uint8)
        else:
            crop = image[y1_valid:y2_valid, x1_valid:x2_valid]

            if image.ndim == 2:
                pad_value = int(image.mean())
            else:
                pad_value = tuple(int(c) for c in image.mean(axis=(0, 1)))

            crop_padded = cv2.copyMakeBorder(
                crop, y1_pad, y2_pad, x1_pad, x2_pad,
                cv2.BORDER_CONSTANT, value=pad_value
            )

        # Create attention mask
        att_mask_full = np.ones((crop_sz, crop_sz), dtype=np.float32)
        if y1_pad > 0:
            att_mask_full[:y1_pad, :] = 0
        if y2_pad > 0:
            att_mask_full[-y2_pad:, :] = 0
        if x1_pad > 0:
            att_mask_full[:, :x1_pad] = 0
        if x2_pad > 0:
            att_mask_full[:, -x2_pad:] = 0

        # Resize
        resize_factor = output_size / crop_sz
        crop_resized = cv2.resize(
            crop_padded, (output_size, output_size),
            interpolation=cv2.INTER_LINEAR
        )
        att_mask_resized = cv2.resize(
            att_mask_full, (output_size, output_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert to tensor and normalize
        if crop_resized.ndim == 2:
            crop_tensor = torch.from_numpy(crop_resized).unsqueeze(0).float() / 255.0
            crop_tensor = (crop_tensor - self.sonar_mean) / self.sonar_std
        else:
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            crop_tensor = (crop_tensor - self.rgb_mean) / self.rgb_std

        metadata = {
            'crop_center': (cx, cy),
            'crop_sz': crop_sz,
            'resize_factor': resize_factor,
            'crop_x1': x1,
            'crop_y1': y1,
            'pad_left': x1_pad,
            'pad_top': y1_pad,
            'original_size': (H, W),
            'output_size': output_size,
        }

        return crop_tensor, att_mask_resized, metadata

    def _transform_bbox_to_crop(
        self,
        bbox_xywh: np.ndarray,
        crop_metadata: Dict
    ) -> np.ndarray:
        """Transform bbox from original image to crop coordinates."""
        x, y, w, h = bbox_xywh
        crop_x1 = crop_metadata['crop_x1']
        crop_y1 = crop_metadata['crop_y1']
        resize_factor = crop_metadata['resize_factor']

        x_rel = x - crop_x1
        y_rel = y - crop_y1

        x_out = x_rel * resize_factor
        y_out = y_rel * resize_factor
        w_out = w * resize_factor
        h_out = h * resize_factor

        return np.array([x_out, y_out, w_out, h_out], dtype=np.float32)

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

        # CRITICAL FIX (Audit 2025112602): Use target-centered cropping for search
        # Consistent with training dataset - crop centered on GT bbox
        rgb_search, rgb_att_mask, rgb_search_metadata = self._crop_search_region(
            search_data['rgb_img'],
            search_data['rgb_bbox'],
            output_size=self.search_size,
            area_factor=self.search_area_factor,
            is_sonar=False
        )
        sonar_search, sonar_att_mask, sonar_search_metadata = self._crop_search_region(
            search_data['sonar_img'],
            search_data['sonar_bbox'],
            output_size=self.search_size,
            area_factor=self.search_area_factor,
            is_sonar=True
        )

        # CRITICAL FIX (Audit 2025112602): Transform GT bboxes to crop coordinates
        search_bbox_rgb_pixels = self._transform_bbox_to_crop(
            search_data['rgb_bbox'],
            rgb_search_metadata
        )
        search_bbox_sonar_pixels = self._transform_bbox_to_crop(
            search_data['sonar_bbox'],
            sonar_search_metadata
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

            # CRITICAL FIX (Audit 2025112602):
            # Pixel-space boxes in crop coordinates for evaluation
            'search_bbox_rgb_pixels': search_bbox_rgb_pixels,
            'search_bbox_sonar_pixels': search_bbox_sonar_pixels,

            # CRITICAL FIX (Audit 2025112602): Attention masks for padded regions
            'rgb_att_mask': rgb_att_mask,
            'sonar_att_mask': sonar_att_mask,

            # Aliases for trainer/evaluator compatibility
            'rgb_boxes': search_data['rgb_bbox'],
            'sonar_boxes': search_data['sonar_bbox'],
            'rgb_boxes_pixels': search_bbox_rgb_pixels,
            'sonar_boxes_pixels': search_bbox_sonar_pixels,

            # Validity flags (CRITICAL FIX Audit 2025112504 Finding 1)
            # These flags enable ConditionalG2DCMA to route single/zero modality scenarios
            # CRITICAL FIX (Audit 2025112601): Use is_valid_rgb naming for consistency with
            # collate_fn_template_search and TrackingEvaluator expectations
            'is_valid': search_data['is_valid'],
            'is_valid_rgb': search_data['is_rgb_valid'],
            'is_valid_sonar': search_data['is_sonar_valid'],
            # has_object flags distinguish "no object" from "object present"
            # Needed for confidence head training and zero-modality handling
            'has_object_rgb': search_data['has_object_rgb'],
            'has_object_sonar': search_data['has_object_sonar'],
            'has_object': search_data['has_object'],

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

        # CRITICAL FIX (Audit 2025112401 Finding 5):
        # Use letterbox resize (same as training) instead of plain resize
        # to preserve aspect ratio and maintain consistency
        import cv2

        # Letterbox preprocessing (from template_search_dataset.py)
        h, w = img.shape[:2]
        is_single_channel = img.ndim == 3 and img.shape[2] == 1

        # Compute scale factor to fit within target_size
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize with preserved aspect ratio
        if is_single_channel:
            img_to_resize = img.squeeze(-1)
        else:
            img_to_resize = img

        img_resized = cv2.resize(img_to_resize, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Pad with gray (114 for RGB, 127 for grayscale)
        if img_resized.ndim == 2:
            pad_value = 127
        else:
            pad_value = (114, 114, 114)

        img_letterboxed = cv2.copyMakeBorder(
            img_resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )

        # Store metadata (consistent with training)
        metadata = {
            'scale': scale,
            'pad_left': left,
            'pad_top': top,
            'original_size': (h, w),
            'new_size': (new_h, new_w),
            # Legacy keys for backward compatibility
            'original_height': h,
            'original_width': w,
            'scale_x': scale,
            'scale_y': scale,
            'pad_x': left,
            'pad_y': top,
        }

        # Convert to tensor and normalize
        if img_letterboxed.ndim == 2:
            # Grayscale (H, W) -> (1, H, W)
            img_tensor = torch.from_numpy(img_letterboxed).unsqueeze(0).float()
        elif img_letterboxed.ndim == 3:
            # Color (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_letterboxed).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unexpected image shape: {img_letterboxed.shape}")

        # CRITICAL FIX (Audit 2025112401 Finding 4):
        # Apply ImageNet normalization (same as training)
        # CRITICAL FIX (Audit 2025112402 Finding 3):
        # Use config-loaded normalization parameters instead of hardcoding
        img_tensor = img_tensor / 255.0

        if is_sonar:
            # Sonar normalization from config
            img_tensor = (img_tensor - self.sonar_mean) / self.sonar_std
        else:
            # RGB normalization from config
            img_tensor = (img_tensor - self.rgb_mean) / self.rgb_std

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