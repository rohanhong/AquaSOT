"""
Sonar Quality Estimator

Estimates the quality of Forward-Looking Sonar (FLS) frames based on
Signal-to-Noise Ratio (SNR), speckle noise level, and echo intensity.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from aquasot3d.core import BaseModule


class SonarQualityEstimator(BaseModule):
    """
    Sonar frame quality estimator.

    Computes quality scores based on:
    - Signal-to-Noise Ratio (SNR)
    - Speckle noise level (local variance)
    - Echo intensity within target region

    Args:
        method: Quality estimation method ('snr', 'speckle', 'intensity', 'combined')
        learnable: Whether to use learnable quality estimation
        snr_threshold: Threshold for SNR normalization
        temperature: Temperature for sigmoid normalization
        use_geometry: Whether to use sonar geometry information

    Returns:
        Quality score in range [0, 1], where 1 = high quality, 0 = low quality

    Example:
        >>> estimator = SonarQualityEstimator(method='combined', learnable=True)
        >>> sonar_frame = torch.randn(4, 1, 682, 376)  # Grayscale sonar
        >>> bbox = torch.tensor([[100, 200, 150, 120]] * 4)
        >>> quality = estimator(sonar_frame, bbox)
        >>> print(quality.shape)  # (4, 1)
    """

    def __init__(
        self,
        method: str = 'combined',
        learnable: bool = True,
        snr_threshold: float = 5.0,
        temperature: float = 2.0,
        use_geometry: bool = True,
        **kwargs
    ):
        super().__init__()

        self.method = method
        self.learnable = learnable
        self.snr_threshold = snr_threshold
        self.temperature = temperature
        self.use_geometry = use_geometry

        # Learnable quality estimation network
        if learnable:
            self.quality_net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.quality_net = None

        self.logger.info(
            f"Initialized SonarQualityEstimator: method={method}, "
            f"learnable={learnable}, use_geometry={use_geometry}"
        )

    def compute_snr(
        self,
        sonar_frame: torch.Tensor,
        bbox: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Signal-to-Noise Ratio (SNR).

        Args:
            sonar_frame: Sonar image (B, 1, H, W)
            bbox: Bounding boxes (B, 4) in format [x, y, w, h]

        Returns:
            SNR values (B, 1)
        """
        B = sonar_frame.shape[0]
        snr_values = []

        for i in range(B):
            x, y, w, h = bbox[i].long()

            # Clamp to image boundaries
            x = torch.clamp(x, 0, sonar_frame.shape[3] - 1)
            y = torch.clamp(y, 0, sonar_frame.shape[2] - 1)
            w = torch.clamp(w, 1, sonar_frame.shape[3] - x)
            h = torch.clamp(h, 1, sonar_frame.shape[2] - y)

            # Signal: mean intensity within bbox
            roi = sonar_frame[i, :, y:y+h, x:x+w]
            signal = torch.mean(roi)

            # Noise: std of background region
            # Create mask for background (everything outside bbox)
            mask = torch.ones_like(sonar_frame[i, 0], dtype=torch.bool)
            mask[y:y+h, x:x+w] = False

            background = sonar_frame[i, 0][mask]
            if background.numel() > 0:
                noise = torch.std(background) + 1e-6
            else:
                noise = torch.tensor(1.0, device=sonar_frame.device)

            # SNR
            snr = signal / noise
            snr_values.append(snr)

        return torch.stack(snr_values).unsqueeze(1)  # (B, 1)

    def compute_speckle_noise(
        self,
        sonar_frame: torch.Tensor,
        bbox: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute speckle noise level using local variance.

        Args:
            sonar_frame: Sonar image (B, 1, H, W)
            bbox: Bounding boxes (B, 4)

        Returns:
            Speckle noise level (B, 1), inverted so higher = better quality
        """
        B = sonar_frame.shape[0]
        noise_levels = []

        for i in range(B):
            x, y, w, h = bbox[i].long()

            # Clamp to boundaries
            x = torch.clamp(x, 0, sonar_frame.shape[3] - 1)
            y = torch.clamp(y, 0, sonar_frame.shape[2] - 1)
            w = torch.clamp(w, 1, sonar_frame.shape[3] - x)
            h = torch.clamp(h, 1, sonar_frame.shape[2] - y)

            # ROI
            roi = sonar_frame[i, :, y:y+h, x:x+w]

            # Local variance (measure of speckle)
            # Use sliding window
            if roi.shape[1] >= 5 and roi.shape[2] >= 5:
                kernel_size = 5
                unfold = F.unfold(
                    roi.unsqueeze(0),
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )  # (1, kernel_size^2, num_patches)

                local_variance = torch.var(unfold, dim=1).mean()
            else:
                local_variance = torch.var(roi)

            # Invert: lower variance = higher quality
            noise_level = 1.0 / (1.0 + local_variance)
            noise_levels.append(noise_level)

        return torch.stack(noise_levels).unsqueeze(1)  # (B, 1)

    def compute_echo_intensity(
        self,
        sonar_frame: torch.Tensor,
        bbox: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute echo intensity within target region.

        Args:
            sonar_frame: Sonar image (B, 1, H, W)
            bbox: Bounding boxes (B, 4)

        Returns:
            Normalized echo intensity (B, 1)
        """
        B = sonar_frame.shape[0]
        intensities = []

        for i in range(B):
            x, y, w, h = bbox[i].long()

            # Clamp
            x = torch.clamp(x, 0, sonar_frame.shape[3] - 1)
            y = torch.clamp(y, 0, sonar_frame.shape[2] - 1)
            w = torch.clamp(w, 1, sonar_frame.shape[3] - x)
            h = torch.clamp(h, 1, sonar_frame.shape[2] - y)

            # Mean intensity in ROI
            roi = sonar_frame[i, :, y:y+h, x:x+w]
            intensity = torch.mean(roi)

            intensities.append(intensity)

        return torch.stack(intensities).unsqueeze(1)  # (B, 1)

    def crop_roi(self, sonar_frame: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        """
        Crop region of interest from sonar image.

        Args:
            sonar_frame: Input sonar (B, 1, H, W)
            bbox: Bounding boxes (B, 4)

        Returns:
            Cropped ROIs (B, 1, 64, 64)
        """
        B = sonar_frame.shape[0]
        rois = []

        for i in range(B):
            x, y, w, h = bbox[i].long()

            # Clamp
            x = torch.clamp(x, 0, sonar_frame.shape[3] - 1)
            y = torch.clamp(y, 0, sonar_frame.shape[2] - 1)
            w = torch.clamp(w, 1, sonar_frame.shape[3] - x)
            h = torch.clamp(h, 1, sonar_frame.shape[2] - y)

            roi = sonar_frame[i:i+1, :, y:y+h, x:x+w]

            # Resize to fixed size
            if roi.shape[2] > 0 and roi.shape[3] > 0:
                roi = F.interpolate(roi, size=(64, 64), mode='bilinear', align_corners=False)
            else:
                roi = torch.zeros(1, 1, 64, 64, device=sonar_frame.device)

            rois.append(roi)

        return torch.cat(rois, dim=0)  # (B, 1, 64, 64)

    def forward(
        self,
        sonar_frame: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
        geometry_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate sonar frame quality.

        Args:
            sonar_frame: Sonar frames (B, 1, H, W)
            bbox: Bounding boxes (B, 4) for ROI-based quality estimation
            geometry_weights: Sonar geometry weights (B, H, W) from SonarGeometryHandler

        Returns:
            Quality scores (B, 1) in range [0, 1]
        """
        if bbox is None:
            # Use center region as default bbox
            B, _, H, W = sonar_frame.shape
            bbox = torch.tensor(
                [[W // 4, H // 4, W // 2, H // 2]] * B,
                device=sonar_frame.device,
                dtype=torch.float32
            )

        # Learnable quality estimation
        if self.learnable and self.quality_net is not None:
            roi = self.crop_roi(sonar_frame, bbox)
            quality = self.quality_net(roi)
            return quality

        # Traditional quality estimation
        if self.method == 'snr':
            snr = self.compute_snr(sonar_frame, bbox)
            quality = torch.sigmoid(
                (snr - self.snr_threshold) / self.temperature
            )

        elif self.method == 'speckle':
            quality = self.compute_speckle_noise(sonar_frame, bbox)

        elif self.method == 'intensity':
            intensity = self.compute_echo_intensity(sonar_frame, bbox)
            # Normalize intensity to [0, 1]
            quality = torch.sigmoid((intensity - 0.5) / 0.2)

        elif self.method == 'combined':
            # Combine multiple metrics
            snr = self.compute_snr(sonar_frame, bbox)
            speckle_quality = self.compute_speckle_noise(sonar_frame, bbox)
            intensity = self.compute_echo_intensity(sonar_frame, bbox)

            # Normalize
            snr_norm = torch.sigmoid(
                (snr - self.snr_threshold) / self.temperature
            )
            intensity_norm = torch.sigmoid((intensity - 0.5) / 0.2)

            # Weighted combination
            quality = 0.5 * snr_norm + 0.2 * speckle_quality + 0.3 * intensity_norm

        else:
            raise ValueError(f"Unknown quality estimation method: {self.method}")

        # Apply geometry weighting if provided
        if self.use_geometry and geometry_weights is not None:
            # Weight quality by geometry (near-range has higher weight)
            B = sonar_frame.shape[0]
            geo_weights = []

            for i in range(B):
                x, y, w, h = bbox[i].long()
                x = torch.clamp(x, 0, sonar_frame.shape[3] - 1)
                y = torch.clamp(y, 0, sonar_frame.shape[2] - 1)
                w = torch.clamp(w, 1, sonar_frame.shape[3] - x)
                h = torch.clamp(h, 1, sonar_frame.shape[2] - y)

                roi_weights = geometry_weights[i, y:y+h, x:x+w]
                mean_weight = torch.mean(roi_weights)
                geo_weights.append(mean_weight)

            geo_weights = torch.stack(geo_weights).unsqueeze(1)  # (B, 1)
            quality = quality * geo_weights

        return quality


def create_sonar_quality_estimator(
    method: str = 'combined',
    learnable: bool = True,
    **kwargs
) -> SonarQualityEstimator:
    """
    Factory function to create sonar quality estimator.

    Args:
        method: Quality estimation method
        learnable: Whether to use learnable estimation
        **kwargs: Additional arguments

    Returns:
        SonarQualityEstimator instance

    Example:
        >>> estimator = create_sonar_quality_estimator('combined', learnable=True)
    """
    return SonarQualityEstimator(method=method, learnable=learnable, **kwargs)
