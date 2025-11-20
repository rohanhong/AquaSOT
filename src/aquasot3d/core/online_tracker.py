"""
Online Tracker for Template-Search Inference

This module provides online tracking interface with template update strategies
following mainstream SOT methods (OSTrack/TransT/ARTrack).

Implements class-agnostic tracking with dynamic template refresh.

Author: AquaSOT-3D Team
Date: 2025-11-21
"""

import torch
import numpy as np
from typing import Dict, Literal, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TemplateBuffer:
    """Buffer for storing template frames and metadata."""
    rgb_template: torch.Tensor  # [1, 3, H, W]
    sonar_template: torch.Tensor  # [1, 1, H, W]
    template_bbox_rgb: np.ndarray  # [4] in CXCYWH normalized format
    template_bbox_sonar: np.ndarray  # [4] in CXCYWH normalized format
    frame_id_rgb: int  # Frame ID when RGB template was last updated
    frame_id_sonar: int  # Frame ID when Sonar template was last updated
    quality_score_rgb: Optional[float] = None  # RGB quality score [0, 1]
    quality_score_sonar: Optional[float] = None  # Sonar quality score [0, 1]


class OnlineTracker:
    """
    Online tracking with template update strategies.

    This class wraps AquaSOT3DTracker to provide online inference interface
    with per-frame template management, following OSTrack/TransT paradigm.

    Template Update Strategies:
    - first_frame: Always use first frame as template (baseline, no updates)
    - recent: Update template every N frames with latest prediction
    - keyframe: Update template based on quality/motion thresholds
    - mixformer: MixFormer-style quality-driven updates with independent RGB/Sonar updating

    Args:
        model: AquaSOT3DTracker instance
        template_size: Template image size (default: 256)
        search_size: Search image size (default: 512)
        update_strategy: Template update strategy (default: 'first_frame')
        update_interval: Frames between updates for 'recent' strategy (default: 10)
        quality_threshold_rgb: RGB quality threshold for 'mixformer'/'keyframe' (default: 0.7)
        quality_threshold_sonar: Sonar quality threshold for 'mixformer'/'keyframe' (default: 0.7)
        device: Device for inference (default: 'cuda')

    Example:
        >>> tracker = OnlineTracker(model, update_strategy='first_frame')
        >>> tracker.init(first_frame_rgb, first_frame_sonar, init_bbox_rgb, init_bbox_sonar)
        >>> for frame_rgb, frame_sonar in video:
        >>>     bbox_rgb, bbox_sonar, bbox_fusion = tracker.track(frame_rgb, frame_sonar)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        template_size: int = 256,
        search_size: int = 512,
        update_strategy: Literal['first_frame', 'recent', 'keyframe', 'mixformer'] = 'first_frame',
        update_interval: int = 10,
        quality_threshold_rgb: float = 0.7,
        quality_threshold_sonar: float = 0.7,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()

        self.template_size = template_size
        self.search_size = search_size
        self.update_strategy = update_strategy
        self.update_interval = update_interval
        self.quality_threshold_rgb = quality_threshold_rgb
        self.quality_threshold_sonar = quality_threshold_sonar
        self.device = device

        # Tracking state
        self.template_buffer: Optional[TemplateBuffer] = None
        self.frame_count = 0
        self.is_initialized = False

    def init(
        self,
        rgb_frame: np.ndarray,
        sonar_frame: np.ndarray,
        bbox_rgb: np.ndarray,
        bbox_sonar: np.ndarray
    ):
        """
        Initialize tracker with first frame.

        Args:
            rgb_frame: RGB image [H, W, 3] in uint8 format
            sonar_frame: Sonar image [H, W] in uint8 format
            bbox_rgb: Initial RGB bbox [4] in CXCYWH normalized format
            bbox_sonar: Initial sonar bbox [4] in CXCYWH normalized format
        """
        # Preprocess frames
        rgb_template = self._preprocess_image(rgb_frame, is_rgb=True, target_size=self.template_size)
        sonar_template = self._preprocess_image(sonar_frame, is_rgb=False, target_size=self.template_size)

        # Store initial template
        self.template_buffer = TemplateBuffer(
            rgb_template=rgb_template,
            sonar_template=sonar_template,
            template_bbox_rgb=bbox_rgb.copy(),
            template_bbox_sonar=bbox_sonar.copy(),
            frame_id_rgb=0,
            frame_id_sonar=0,
            quality_score_rgb=1.0,
            quality_score_sonar=1.0
        )

        self.frame_count = 0
        self.is_initialized = True

    def track(
        self,
        rgb_frame: np.ndarray,
        sonar_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track object in current frame.

        Args:
            rgb_frame: RGB image [H, W, 3] in uint8 format
            sonar_frame: Sonar image [H, W] in uint8 format

        Returns:
            Tuple of (bbox_rgb, bbox_sonar, bbox_fusion) in CXCYWH normalized format
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")

        self.frame_count += 1

        # Preprocess search frames
        rgb_search = self._preprocess_image(rgb_frame, is_rgb=True, target_size=self.search_size)
        sonar_search = self._preprocess_image(sonar_frame, is_rgb=False, target_size=self.search_size)

        # Run tracker forward pass
        with torch.no_grad():
            outputs = self.model(
                rgb_template=self.template_buffer.rgb_template,
                rgb_search=rgb_search,
                sonar_template=self.template_buffer.sonar_template,
                sonar_search=sonar_search,
                return_losses=False
            )

        # Extract predictions (CXCYWH normalized)
        bbox_rgb = outputs['rgb_boxes'][0].cpu().numpy()
        bbox_sonar = outputs['sonar_boxes'][0].cpu().numpy()
        bbox_fusion = outputs['fusion_boxes'][0].cpu().numpy()

        # Extract quality scores for each modality
        quality_score_rgb = None
        quality_score_sonar = None

        if 'rgb_quality' in outputs:
            quality_score_rgb = outputs['rgb_quality'][0].item()

        if 'sonar_quality' in outputs:
            quality_score_sonar = outputs['sonar_quality'][0].item()

        # Update template based on strategy
        # CRITICAL FIX: Use modality-specific bboxes for independent template updates
        # This ensures MixFormer-style independent RGB/Sonar template updating works correctly
        self._update_template(
            rgb_frame=rgb_frame,
            sonar_frame=sonar_frame,
            bbox_rgb=bbox_rgb,
            bbox_sonar=bbox_sonar,
            quality_score_rgb=quality_score_rgb,
            quality_score_sonar=quality_score_sonar
        )

        return bbox_rgb, bbox_sonar, bbox_fusion

    def _update_template(
        self,
        rgb_frame: np.ndarray,
        sonar_frame: np.ndarray,
        bbox_rgb: np.ndarray,
        bbox_sonar: np.ndarray,
        quality_score_rgb: Optional[float],
        quality_score_sonar: Optional[float]
    ):
        """
        Update template according to update strategy.

        For 'mixformer' strategy, RGB and Sonar templates are updated independently
        based on their respective quality scores, allowing robust tracking even when
        one modality degrades (e.g., RGB occlusion but Sonar clear).
        """
        if self.update_strategy == 'first_frame':
            return

        elif self.update_strategy == 'recent':
            if self.frame_count % self.update_interval == 0:
                rgb_template = self._preprocess_image(rgb_frame, is_rgb=True, target_size=self.template_size)
                sonar_template = self._preprocess_image(sonar_frame, is_rgb=False, target_size=self.template_size)

                self.template_buffer = TemplateBuffer(
                    rgb_template=rgb_template,
                    sonar_template=sonar_template,
                    template_bbox_rgb=bbox_rgb.copy(),
                    template_bbox_sonar=bbox_sonar.copy(),
                    frame_id_rgb=self.frame_count,
                    frame_id_sonar=self.frame_count,
                    quality_score_rgb=quality_score_rgb,
                    quality_score_sonar=quality_score_sonar
                )

        elif self.update_strategy == 'keyframe':
            avg_quality = None
            if quality_score_rgb is not None and quality_score_sonar is not None:
                avg_quality = (quality_score_rgb + quality_score_sonar) / 2.0

            if avg_quality is not None and avg_quality > max(self.quality_threshold_rgb, self.quality_threshold_sonar):
                rgb_template = self._preprocess_image(rgb_frame, is_rgb=True, target_size=self.template_size)
                sonar_template = self._preprocess_image(sonar_frame, is_rgb=False, target_size=self.template_size)

                self.template_buffer = TemplateBuffer(
                    rgb_template=rgb_template,
                    sonar_template=sonar_template,
                    template_bbox_rgb=bbox_rgb.copy(),
                    template_bbox_sonar=bbox_sonar.copy(),
                    frame_id_rgb=self.frame_count,
                    frame_id_sonar=self.frame_count,
                    quality_score_rgb=quality_score_rgb,
                    quality_score_sonar=quality_score_sonar
                )

        elif self.update_strategy == 'mixformer':
            update_rgb = False
            update_sonar = False

            if quality_score_rgb is not None and quality_score_rgb > self.quality_threshold_rgb:
                update_rgb = True

            if quality_score_sonar is not None and quality_score_sonar > self.quality_threshold_sonar:
                update_sonar = True

            if update_rgb or update_sonar:
                new_rgb_template = self.template_buffer.rgb_template
                new_sonar_template = self.template_buffer.sonar_template
                new_bbox_rgb = self.template_buffer.template_bbox_rgb
                new_bbox_sonar = self.template_buffer.template_bbox_sonar
                new_frame_id_rgb = self.template_buffer.frame_id_rgb
                new_frame_id_sonar = self.template_buffer.frame_id_sonar
                new_quality_rgb = self.template_buffer.quality_score_rgb
                new_quality_sonar = self.template_buffer.quality_score_sonar

                if update_rgb:
                    new_rgb_template = self._preprocess_image(rgb_frame, is_rgb=True, target_size=self.template_size)
                    new_bbox_rgb = bbox_rgb.copy()
                    new_frame_id_rgb = self.frame_count
                    new_quality_rgb = quality_score_rgb

                if update_sonar:
                    new_sonar_template = self._preprocess_image(sonar_frame, is_rgb=False, target_size=self.template_size)
                    new_bbox_sonar = bbox_sonar.copy()
                    new_frame_id_sonar = self.frame_count
                    new_quality_sonar = quality_score_sonar

                self.template_buffer = TemplateBuffer(
                    rgb_template=new_rgb_template,
                    sonar_template=new_sonar_template,
                    template_bbox_rgb=new_bbox_rgb,
                    template_bbox_sonar=new_bbox_sonar,
                    frame_id_rgb=new_frame_id_rgb,
                    frame_id_sonar=new_frame_id_sonar,
                    quality_score_rgb=new_quality_rgb,
                    quality_score_sonar=new_quality_sonar
                )

    def _preprocess_image(
        self,
        img: np.ndarray,
        is_rgb: bool,
        target_size: int
    ) -> torch.Tensor:
        """
        Preprocess image for tracker input.

        Applies letterbox resize to preserve aspect ratio, then normalizes.

        Args:
            img: Input image [H, W, C] or [H, W] in uint8 format
            is_rgb: Whether image is RGB (True) or sonar (False)
            target_size: Target size (H=W)

        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Ensure correct shape
        if is_rgb:
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] != 3:
                raise ValueError(f"RGB image must have 3 channels, got {img.shape[-1]}")
        else:
            if img.ndim == 3:
                img = img[:, :, 0]  # Take first channel if multi-channel

        # Letterbox resize
        img_resized = self._letterbox_resize(img, target_size)

        # Convert to float32 and normalize to [0, 1]
        img_float = img_resized.astype(np.float32) / 255.0

        # Apply normalization (ImageNet stats for RGB, mean=0.5 std=0.5 for sonar)
        if is_rgb:
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            img_normalized = (img_float - mean) / std
            # HWC to CHW
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        else:
            mean = 0.5
            std = 0.5
            img_normalized = (img_float - mean) / std
            # HW to 1HW
            img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)

        return img_tensor.to(self.device)

    def _letterbox_resize(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """
        Letterbox resize: resize to fit within target_size and pad to square.

        Preserves aspect ratio by:
        1. Scaling to fit within target_size
        2. Padding to target_size x target_size

        Args:
            img: Input image [H, W] or [H, W, C]
            target_size: Target size (H=W)

        Returns:
            Resized and padded image [target_size, target_size] or [target_size, target_size, C]
        """
        import cv2

        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        if img.ndim == 3:
            canvas = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)
        else:
            canvas = np.zeros((target_size, target_size), dtype=img.dtype)

        # Calculate padding
        pad_top = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2

        # Paste resized image onto canvas
        if img.ndim == 3:
            canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized
        else:
            canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

        return canvas

    def reset(self):
        """Reset tracker state."""
        self.template_buffer = None
        self.frame_count = 0
        self.is_initialized = False

    def get_template_info(self) -> Dict:
        """Get current template information for both RGB and Sonar modalities."""
        if self.template_buffer is None:
            return {}

        return {
            'frame_id_rgb': self.template_buffer.frame_id_rgb,
            'frame_id_sonar': self.template_buffer.frame_id_sonar,
            'quality_score_rgb': self.template_buffer.quality_score_rgb,
            'quality_score_sonar': self.template_buffer.quality_score_sonar,
            'bbox_rgb': self.template_buffer.template_bbox_rgb,
            'bbox_sonar': self.template_buffer.template_bbox_sonar
        }
