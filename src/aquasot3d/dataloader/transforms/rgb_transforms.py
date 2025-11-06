"""
RGB-Specific Data Augmentation Transforms

This module implements data augmentation techniques specifically designed
for RGB camera images in underwater tracking scenarios:
- Color jitter (hue, saturation, brightness, contrast)
- Random blur and noise
- Random occlusion (CutMix/CopyPaste style)
- Geometric transforms (with bbox coordination)
- Normalization to ImageNet statistics

Reference:
    Standard augmentation practices from visual tracking literature,
    adapted for underwater imagery characteristics.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from typing import Optional, Tuple, Union
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ColorJitter:
    """
    Random color jittering for RGB images.

    Randomly adjusts brightness, contrast, saturation, and hue to
    simulate underwater lighting variations.

    Args:
        brightness: Brightness factor range (default: 0.2)
        contrast: Contrast factor range (default: 0.2)
        saturation: Saturation factor range (default: 0.2)
        hue: Hue factor range (default: 0.1)
        prob: Probability of applying jittering (default: 0.5)
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        prob: float = 0.5,
    ):
        self.prob = prob
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply color jittering.

        Args:
            img: RGB image (H, W, 3) in numpy format

        Returns:
            Jittered image (H, W, 3)
        """
        if random.random() < self.prob:
            # Convert to PIL for torchvision transforms
            img_pil = TF.to_pil_image(img)
            img_pil = self.jitter(img_pil)
            img = np.array(img_pil)

        return img


class RandomBlur:
    """
    Random Gaussian blur to simulate water turbidity.

    Args:
        kernel_sizes: List of possible kernel sizes (default: [3, 5, 7])
        sigma_range: Gaussian sigma range (default: (0.1, 2.0))
        prob: Probability of applying blur (default: 0.3)
    """

    def __init__(
        self,
        kernel_sizes: list = [3, 5, 7],
        sigma_range: Tuple[float, float] = (0.1, 2.0),
        prob: float = 0.3,
    ):
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range
        self.prob = prob

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random Gaussian blur.

        Args:
            img: RGB image (H, W, 3)

        Returns:
            Blurred image (H, W, 3)
        """
        if random.random() < self.prob:
            kernel_size = random.choice(self.kernel_sizes)
            sigma = random.uniform(*self.sigma_range)
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

        return img


class RandomOcclusion:
    """
    Random occlusion augmentation (CutMix-style).

    Randomly masks rectangular regions to simulate temporary occlusions.

    Args:
        min_size: Minimum occlusion size ratio (default: 0.05)
        max_size: Maximum occlusion size ratio (default: 0.3)
        prob: Probability of applying occlusion (default: 0.2)
        fill_mode: Fill mode ('zero', 'random', 'mean')
    """

    def __init__(
        self,
        min_size: float = 0.05,
        max_size: float = 0.3,
        prob: float = 0.2,
        fill_mode: str = 'zero',
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.prob = prob
        self.fill_mode = fill_mode

    def __call__(self, img: np.ndarray, bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply random occlusion.

        Args:
            img: RGB image (H, W, 3)
            bbox: Optional bounding box to avoid occluding target [x, y, w, h]

        Returns:
            Occluded image (H, W, 3)
        """
        if random.random() >= self.prob:
            return img

        H, W = img.shape[:2]

        # Random occlusion size
        size_ratio = random.uniform(self.min_size, self.max_size)
        occ_h = int(H * size_ratio)
        occ_w = int(W * size_ratio)

        # Random position
        max_attempts = 10
        for _ in range(max_attempts):
            x = random.randint(0, W - occ_w)
            y = random.randint(0, H - occ_h)

            # Check if occlusion overlaps with target bbox
            if bbox is not None and not np.all(bbox == 0):
                bbox_x, bbox_y, bbox_w, bbox_h = bbox
                # IoU check
                overlap_x1 = max(x, bbox_x)
                overlap_y1 = max(y, bbox_y)
                overlap_x2 = min(x + occ_w, bbox_x + bbox_w)
                overlap_y2 = min(y + occ_h, bbox_y + bbox_h)

                overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
                bbox_area = bbox_w * bbox_h

                # If significant overlap, try another position
                if overlap_area > 0.3 * bbox_area:
                    continue

            # Apply occlusion
            if self.fill_mode == 'zero':
                img[y:y+occ_h, x:x+occ_w] = 0
            elif self.fill_mode == 'mean':
                mean_color = img.mean(axis=(0, 1))
                img[y:y+occ_h, x:x+occ_w] = mean_color
            elif self.fill_mode == 'random':
                img[y:y+occ_h, x:x+occ_w] = np.random.randint(0, 256, (occ_h, occ_w, 3))

            break

        return img


class RandomHorizontalFlip:
    """
    Random horizontal flip with bbox coordination.

    Args:
        prob: Probability of flipping (default: 0.5)
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        img: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply random horizontal flip.

        Args:
            img: RGB image (H, W, 3)
            bbox: Optional bounding box [x, y, w, h]

        Returns:
            Flipped image and bbox (if provided)
        """
        if random.random() < self.prob:
            img = cv2.flip(img, 1)  # Horizontal flip

            if bbox is not None and not np.all(bbox == 0):
                W = img.shape[1]
                x, y, w, h = bbox
                # Flip bbox
                bbox = np.array([W - x - w, y, w, h], dtype=np.float32)

        if bbox is not None:
            return img, bbox
        return img


class Normalize:
    """
    Normalize RGB image to ImageNet statistics.

    Args:
        mean: Mean values for RGB channels (default: ImageNet mean)
        std: Std values for RGB channels (default: ImageNet std)
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image.

        Args:
            img: RGB image (H, W, 3) in range [0, 255]

        Returns:
            Normalized image (H, W, 3)
        """
        # Convert to [0, 1] range
        img = img.astype(np.float32) / 255.0

        # Normalize
        img = (img - self.mean) / self.std

        return img


class ToTensor:
    """
    Convert numpy array to PyTorch tensor.

    Converts (H, W, C) to (C, H, W) format.
    """

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert to tensor.

        Args:
            img: Image array (H, W, C)

        Returns:
            Tensor (C, H, W)
        """
        # Handle grayscale
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        # Transpose to (C, H, W)
        img = img.transpose(2, 0, 1)

        # Convert to tensor
        return torch.from_numpy(img.copy())


class RGBTransformPipeline:
    """
    Complete RGB transformation pipeline for training.

    Args:
        enable_color_jitter: Enable color jittering
        enable_blur: Enable random blur
        enable_occlusion: Enable random occlusion
        enable_flip: Enable horizontal flip
        normalize: Enable normalization
        to_tensor: Convert to tensor
    """

    def __init__(
        self,
        enable_color_jitter: bool = True,
        enable_blur: bool = True,
        enable_occlusion: bool = True,
        enable_flip: bool = True,
        normalize: bool = True,
        to_tensor: bool = True,
    ):
        self.transforms = []

        if enable_color_jitter:
            self.transforms.append(ColorJitter())

        if enable_blur:
            self.transforms.append(RandomBlur())

        if enable_occlusion:
            self.occlusion = RandomOcclusion()
            self.enable_occlusion = True
        else:
            self.enable_occlusion = False

        if enable_flip:
            self.flip = RandomHorizontalFlip()
            self.enable_flip = True
        else:
            self.enable_flip = False

        if normalize:
            self.normalize = Normalize()
            self.enable_normalize = True
        else:
            self.enable_normalize = False

        if to_tensor:
            self.to_tensor = ToTensor()
            self.enable_to_tensor = True
        else:
            self.enable_to_tensor = False

    def __call__(
        self,
        img: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        """
        Apply transformation pipeline.

        Args:
            img: RGB image (H, W, 3)
            bbox: Optional bounding box [x, y, w, h]

        Returns:
            Transformed image (and bbox if provided)
        """
        # Apply transforms that don't need bbox
        for transform in self.transforms:
            img = transform(img)

        # Apply occlusion (may need bbox to avoid target)
        if self.enable_occlusion:
            img = self.occlusion(img, bbox)

        # Apply flip (needs bbox coordination)
        if self.enable_flip:
            if bbox is not None:
                img, bbox = self.flip(img, bbox)
            else:
                img = self.flip(img)

        # Normalize
        if self.enable_normalize:
            img = self.normalize(img)

        # Convert to tensor
        if self.enable_to_tensor:
            img = self.to_tensor(img)

        if bbox is not None:
            return img, bbox
        return img