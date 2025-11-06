"""
RGB Quality Estimator

Estimates the quality of RGB frames based on sharpness, blur detection,
and other image quality metrics. Used for uncertainty-aware gating in
multi-modal fusion.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

from aquasot3d.core import BaseModule


class RGBQualityEstimator(BaseModule):
    """
    RGB frame quality estimator.

    Computes quality scores based on:
    - Laplacian variance (sharpness)
    - Gradient magnitude (edge strength)
    - Frequency domain analysis (blur detection)

    Args:
        method: Quality estimation method ('laplacian', 'gradient', 'frequency', 'combined')
        learnable: Whether to use learnable quality estimation
        sharpness_threshold: Threshold for sharpness normalization
        temperature: Temperature for sigmoid normalization

    Returns:
        Quality score in range [0, 1], where 1 = high quality, 0 = low quality

    Example:
        >>> estimator = RGBQualityEstimator(method='combined', learnable=True)
        >>> rgb_frame = torch.randn(4, 3, 256, 256)
        >>> bbox = torch.tensor([[64, 64, 128, 128]] * 4)
        >>> quality = estimator(rgb_frame, bbox)
        >>> print(quality.shape)  # (4, 1)
    """

    def __init__(
        self,
        method: str = 'combined',
        learnable: bool = True,
        sharpness_threshold: float = 100.0,
        temperature: float = 50.0,
        **kwargs
    ):
        super().__init__()

        self.method = method
        self.learnable = learnable
        self.sharpness_threshold = sharpness_threshold
        self.temperature = temperature

        # Laplacian kernel for edge detection
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([
                [[0., -1., 0.],
                 [-1., 4., -1.],
                 [0., -1., 0.]]
            ]).unsqueeze(0)  # (1, 1, 3, 3)
        )

        # Sobel kernels for gradient computation
        self.register_buffer(
            'sobel_x',
            torch.tensor([
                [[-1., 0., 1.],
                 [-2., 0., 2.],
                 [-1., 0., 1.]]
            ]).unsqueeze(0)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([
                [[-1., -2., -1.],
                 [0., 0., 0.],
                 [1., 2., 1.]]
            ]).unsqueeze(0)
        )

        # Learnable quality estimation network
        if learnable:
            self.quality_net = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            self.quality_net = None

        self.logger.info(
            f"Initialized RGBQualityEstimator: method={method}, "
            f"learnable={learnable}"
        )

    def compute_laplacian_variance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian variance as sharpness measure.

        Args:
            image: Input image (B, C, H, W) or (B, 1, H, W)

        Returns:
            Laplacian variance (B, 1)
        """
        # Convert to grayscale if RGB
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image

        # Apply Laplacian
        laplacian = F.conv2d(gray, self.laplacian_kernel, padding=1)

        # Compute variance
        variance = torch.var(laplacian, dim=[2, 3])  # (B, 1)

        return variance

    def compute_gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude using Sobel filters.

        Args:
            image: Input image (B, C, H, W)

        Returns:
            Mean gradient magnitude (B, 1)
        """
        # Convert to grayscale
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image

        # Compute gradients
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        # Gradient magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Mean magnitude
        mean_magnitude = torch.mean(magnitude, dim=[2, 3])  # (B, 1)

        return mean_magnitude

    def compute_frequency_quality(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute quality based on frequency domain analysis.

        High-frequency content indicates sharp images.

        Args:
            image: Input image (B, C, H, W)

        Returns:
            Frequency-based quality score (B, 1)
        """
        # Convert to grayscale
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image

        B, _, H, W = gray.shape

        # FFT
        fft = torch.fft.fft2(gray.squeeze(1))  # (B, H, W)
        fft_shift = torch.fft.fftshift(fft)

        # Magnitude spectrum
        magnitude = torch.abs(fft_shift)

        # High-frequency mask (outer regions)
        center_h, center_w = H // 2, W // 2
        radius = min(H, W) // 4

        y, x = torch.meshgrid(
            torch.arange(H, device=image.device),
            torch.arange(W, device=image.device),
            indexing='ij'
        )
        dist = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        high_freq_mask = (dist > radius).float()

        # High-frequency energy
        high_freq_energy = torch.sum(
            magnitude * high_freq_mask.unsqueeze(0),
            dim=[1, 2]
        )  # (B,)

        # Normalize
        total_energy = torch.sum(magnitude, dim=[1, 2]) + 1e-8
        quality = (high_freq_energy / total_energy).unsqueeze(1)  # (B, 1)

        return quality

    def crop_roi(self, image: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        """
        Crop region of interest from image.

        Args:
            image: Input image (B, C, H, W)
            bbox: Bounding boxes (B, 4) in format [x, y, w, h]

        Returns:
            List of cropped ROIs
        """
        B = image.shape[0]
        rois = []

        for i in range(B):
            x, y, w, h = bbox[i].long()
            # Clamp to image boundaries
            x = torch.clamp(x, 0, image.shape[3] - 1)
            y = torch.clamp(y, 0, image.shape[2] - 1)
            w = torch.clamp(w, 1, image.shape[3] - x)
            h = torch.clamp(h, 1, image.shape[2] - y)

            roi = image[i:i+1, :, y:y+h, x:x+w]

            # Resize to fixed size for consistency
            if roi.shape[2] > 0 and roi.shape[3] > 0:
                roi = F.interpolate(roi, size=(64, 64), mode='bilinear', align_corners=False)
            else:
                roi = torch.zeros(1, image.shape[1], 64, 64, device=image.device)

            rois.append(roi)

        return torch.cat(rois, dim=0)  # (B, C, 64, 64)

    def forward(
        self,
        rgb_frame: torch.Tensor,
        bbox: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate RGB frame quality.

        Args:
            rgb_frame: RGB frames (B, 3, H, W)
            bbox: Bounding boxes (B, 4) for ROI-based quality estimation
                  If None, use entire frame

        Returns:
            Quality scores (B, 1) in range [0, 1]
        """
        # Extract ROI if bbox provided
        if bbox is not None:
            roi = self.crop_roi(rgb_frame, bbox)
        else:
            roi = rgb_frame

        # Learnable quality estimation
        if self.learnable and self.quality_net is not None:
            quality = self.quality_net(roi)
            return quality

        # Traditional quality estimation
        if self.method == 'laplacian':
            variance = self.compute_laplacian_variance(roi)
            quality = torch.sigmoid(
                (variance - self.sharpness_threshold) / self.temperature
            )

        elif self.method == 'gradient':
            magnitude = self.compute_gradient_magnitude(roi)
            quality = torch.sigmoid(
                (magnitude - 10.0) / 5.0  # Tunable thresholds
            )

        elif self.method == 'frequency':
            quality = self.compute_frequency_quality(roi)

        elif self.method == 'combined':
            # Combine multiple metrics
            variance = self.compute_laplacian_variance(roi)
            magnitude = self.compute_gradient_magnitude(roi)
            freq_quality = self.compute_frequency_quality(roi)

            # Normalize
            variance_norm = torch.sigmoid(
                (variance - self.sharpness_threshold) / self.temperature
            )
            magnitude_norm = torch.sigmoid((magnitude - 10.0) / 5.0)

            # Weighted combination
            quality = 0.4 * variance_norm + 0.3 * magnitude_norm + 0.3 * freq_quality

        else:
            raise ValueError(f"Unknown quality estimation method: {self.method}")

        return quality


def create_rgb_quality_estimator(
    method: str = 'combined',
    learnable: bool = True,
    **kwargs
) -> RGBQualityEstimator:
    """
    Factory function to create RGB quality estimator.

    Args:
        method: Quality estimation method
        learnable: Whether to use learnable estimation
        **kwargs: Additional arguments

    Returns:
        RGBQualityEstimator instance

    Example:
        >>> estimator = create_rgb_quality_estimator('combined', learnable=True)
    """
    return RGBQualityEstimator(method=method, learnable=learnable, **kwargs)
