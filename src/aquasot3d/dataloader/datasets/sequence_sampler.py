"""
Temporal Sequence Sampler

This module implements various temporal sequence sampling strategies for
video-based object tracking. It handles frame interval control, occlusion-aware
sampling, and temporal window generation for training and evaluation.

Sampling Strategies:
    - Uniform: Sample frames at regular intervals
    - Random: Sample random frames within a temporal window
    - Adaptive: Sample based on motion/appearance change
    - Occlusion-Aware: Skip or handle occluded frames intelligently

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from typing import Dict, List, Optional, Tuple
import random
import numpy as np

from ...core.registry import SAMPLER_REGISTRY


class BaseSampler:
    """
    Base class for sequence samplers.

    Args:
        dataset: Dataset instance to sample from
        num_samples: Number of frames to sample per sequence
        temporal_window: Maximum temporal distance between sampled frames
        skip_occluded: Whether to skip occluded frames
    """

    def __init__(
        self,
        dataset,
        num_samples: int = 2,
        temporal_window: int = 100,
        skip_occluded: bool = True,
    ):
        self.dataset = dataset
        self.num_samples = num_samples
        self.temporal_window = temporal_window
        self.skip_occluded = skip_occluded

    def sample(self, start_idx: int) -> List[int]:
        """
        Sample frame indices starting from start_idx.

        Args:
            start_idx: Starting frame index (0-based)

        Returns:
            List of sampled frame indices

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement sample() method")

    def is_valid_frame(self, frame_idx: int) -> bool:
        """
        Check if frame is valid for sampling.

        Args:
            frame_idx: Frame index to check (0-based)

        Returns:
            True if frame is valid for sampling
        """
        if frame_idx < 0 or frame_idx >= self.dataset.num_frames:
            return False

        if self.skip_occluded:
            rgb_bbox = self.dataset.rgb_bboxes[frame_idx]
            sonar_bbox = self.dataset.sonar_bboxes[frame_idx]

            # Check if either modality is occluded
            rgb_occluded = np.all(rgb_bbox == 0)
            sonar_occluded = np.all(sonar_bbox == 0)

            return not (rgb_occluded or sonar_occluded)

        return True


@SAMPLER_REGISTRY.register("UniformSampler")
class UniformSampler(BaseSampler):
    """
    Uniform temporal sampling strategy.

    Samples frames at regular intervals within the temporal window.

    Args:
        dataset: Dataset instance
        num_samples: Number of frames to sample
        temporal_window: Maximum temporal distance
        skip_occluded: Whether to skip occluded frames
        interval: Fixed frame interval (if None, auto-computed)

    Example:
        >>> sampler = UniformSampler(dataset, num_samples=5, interval=10)
        >>> indices = sampler.sample(start_idx=0)
        >>> # Returns [0, 10, 20, 30, 40]
    """

    def __init__(
        self,
        dataset,
        num_samples: int = 2,
        temporal_window: int = 100,
        skip_occluded: bool = True,
        interval: Optional[int] = None,
    ):
        super().__init__(dataset, num_samples, temporal_window, skip_occluded)
        self.interval = interval

    def sample(self, start_idx: int) -> List[int]:
        """
        Sample frames uniformly from start_idx.

        Args:
            start_idx: Starting frame index (0-based)

        Returns:
            List of uniformly sampled frame indices
        """
        sampled_indices = []

        # Determine interval
        if self.interval is None:
            # Auto-compute interval based on window and num_samples
            interval = max(1, self.temporal_window // self.num_samples)
        else:
            interval = self.interval

        current_idx = start_idx
        attempts = 0
        max_attempts = self.temporal_window * 2  # Prevent infinite loop

        while len(sampled_indices) < self.num_samples and attempts < max_attempts:
            if self.is_valid_frame(current_idx):
                sampled_indices.append(current_idx)
                current_idx += interval
            else:
                # Skip invalid frame and continue
                current_idx += 1

            attempts += 1

        # If not enough samples, fill with last valid index
        if len(sampled_indices) > 0:
            while len(sampled_indices) < self.num_samples:
                sampled_indices.append(sampled_indices[-1])

        return sampled_indices


@SAMPLER_REGISTRY.register("RandomSampler")
class RandomSampler(BaseSampler):
    """
    Random temporal sampling strategy.

    Samples frames randomly within the temporal window.

    Args:
        dataset: Dataset instance
        num_samples: Number of frames to sample
        temporal_window: Maximum temporal distance
        skip_occluded: Whether to skip occluded frames
        sorted_output: If True, sort sampled indices chronologically

    Example:
        >>> sampler = RandomSampler(dataset, num_samples=3, temporal_window=50)
        >>> indices = sampler.sample(start_idx=10)
        >>> # Returns e.g., [12, 35, 48] (random within [10, 60])
    """

    def __init__(
        self,
        dataset,
        num_samples: int = 2,
        temporal_window: int = 100,
        skip_occluded: bool = True,
        sorted_output: bool = True,
    ):
        super().__init__(dataset, num_samples, temporal_window, skip_occluded)
        self.sorted_output = sorted_output

    def sample(self, start_idx: int) -> List[int]:
        """
        Sample frames randomly from start_idx to start_idx + temporal_window.

        Args:
            start_idx: Starting frame index (0-based)

        Returns:
            List of randomly sampled frame indices
        """
        # Determine valid range
        end_idx = min(start_idx + self.temporal_window, self.dataset.num_frames)

        # Collect valid candidate indices
        candidate_indices = [
            idx for idx in range(start_idx, end_idx)
            if self.is_valid_frame(idx)
        ]

        if len(candidate_indices) == 0:
            # No valid frames, return start_idx repeated
            return [start_idx] * self.num_samples

        # Sample from candidates
        if len(candidate_indices) >= self.num_samples:
            sampled_indices = random.sample(candidate_indices, self.num_samples)
        else:
            # Not enough candidates, sample with replacement
            sampled_indices = random.choices(candidate_indices, k=self.num_samples)

        # Sort if requested
        if self.sorted_output:
            sampled_indices.sort()

        return sampled_indices


@SAMPLER_REGISTRY.register("OcclusionAwareSampler")
class OcclusionAwareSampler(BaseSampler):
    """
    Occlusion-aware sampling strategy.

    Intelligently handles occluded frames by:
    1. Prioritizing non-occluded frames
    2. Maintaining temporal continuity across occlusions
    3. Using last valid bbox for occluded frames

    Args:
        dataset: Dataset instance
        num_samples: Number of frames to sample
        temporal_window: Maximum temporal distance
        occlusion_strategy: How to handle occlusions ("skip", "use_last", "interpolate")

    Example:
        >>> sampler = OcclusionAwareSampler(
        ...     dataset, num_samples=5, occlusion_strategy="use_last"
        ... )
        >>> indices = sampler.sample(start_idx=0)
    """

    def __init__(
        self,
        dataset,
        num_samples: int = 2,
        temporal_window: int = 100,
        occlusion_strategy: str = "skip",
    ):
        # Always enable skip_occluded for internal checking
        super().__init__(dataset, num_samples, temporal_window, skip_occluded=True)

        if occlusion_strategy not in ["skip", "use_last", "interpolate"]:
            raise ValueError(
                f"Invalid occlusion_strategy: {occlusion_strategy}. "
                "Must be 'skip', 'use_last', or 'interpolate'"
            )

        self.occlusion_strategy = occlusion_strategy

    def sample(self, start_idx: int) -> List[int]:
        """
        Sample frames with occlusion awareness.

        Args:
            start_idx: Starting frame index (0-based)

        Returns:
            List of sampled frame indices with occlusion handling
        """
        if self.occlusion_strategy == "skip":
            # Use uniform sampling but skip occluded frames
            sampler = UniformSampler(
                self.dataset,
                self.num_samples,
                self.temporal_window,
                skip_occluded=True,
            )
            return sampler.sample(start_idx)

        elif self.occlusion_strategy == "use_last":
            # Sample uniformly, but mark which frames need last-valid-bbox
            return self._sample_with_last_valid(start_idx)

        else:  # interpolate
            # Sample and interpolate bboxes across occlusions
            return self._sample_with_interpolation(start_idx)

    def _sample_with_last_valid(self, start_idx: int) -> List[int]:
        """
        Sample frames, allowing occluded frames but tracking last valid bbox.

        Args:
            start_idx: Starting frame index

        Returns:
            List of frame indices (includes occluded frames)
        """
        sampled_indices = []
        interval = max(1, self.temporal_window // self.num_samples)

        current_idx = start_idx
        last_valid_idx = start_idx

        for _ in range(self.num_samples):
            if current_idx >= self.dataset.num_frames:
                # Exceed dataset range, use last valid
                sampled_indices.append(last_valid_idx)
            else:
                sampled_indices.append(current_idx)

                # Update last valid index if current is valid
                if self.is_valid_frame(current_idx):
                    last_valid_idx = current_idx

            current_idx += interval

        return sampled_indices

    def _sample_with_interpolation(self, start_idx: int) -> List[int]:
        """
        Sample frames and interpolate bboxes for occluded regions.

        Args:
            start_idx: Starting frame index

        Returns:
            List of frame indices with interpolation markers
        """
        # For now, use same strategy as use_last
        # Full interpolation logic requires bbox manipulation in dataset
        return self._sample_with_last_valid(start_idx)


@SAMPLER_REGISTRY.register("AdaptiveSampler")
class AdaptiveSampler(BaseSampler):
    """
    Adaptive sampling based on motion/appearance change.

    Adjusts frame sampling density based on estimated motion or appearance
    variation. Dense sampling in high-motion regions, sparse in static regions.

    Args:
        dataset: Dataset instance
        num_samples: Number of frames to sample
        temporal_window: Maximum temporal distance
        skip_occluded: Whether to skip occluded frames
        motion_threshold: Threshold for motion-based sampling

    Example:
        >>> sampler = AdaptiveSampler(dataset, num_samples=5)
        >>> indices = sampler.sample(start_idx=0)
    """

    def __init__(
        self,
        dataset,
        num_samples: int = 2,
        temporal_window: int = 100,
        skip_occluded: bool = True,
        motion_threshold: float = 10.0,
    ):
        super().__init__(dataset, num_samples, temporal_window, skip_occluded)
        self.motion_threshold = motion_threshold

    def _estimate_motion(self, idx1: int, idx2: int) -> float:
        """
        Estimate motion between two frames based on bbox center displacement.

        Args:
            idx1: First frame index
            idx2: Second frame index

        Returns:
            Motion magnitude (Euclidean distance of bbox centers)
        """
        bbox1 = self.dataset.rgb_bboxes[idx1]
        bbox2 = self.dataset.rgb_bboxes[idx2]

        # Check for occlusion
        if np.all(bbox1 == 0) or np.all(bbox2 == 0):
            return 0.0

        # Calculate bbox centers
        center1 = np.array([bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2])
        center2 = np.array([bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2])

        # Euclidean distance
        motion = np.linalg.norm(center2 - center1)

        return motion

    def sample(self, start_idx: int) -> List[int]:
        """
        Sample frames adaptively based on motion estimation.

        Args:
            start_idx: Starting frame index (0-based)

        Returns:
            List of adaptively sampled frame indices
        """
        sampled_indices = [start_idx]
        current_idx = start_idx

        while len(sampled_indices) < self.num_samples:
            # Find next valid frame
            next_idx = current_idx + 1

            # Search for next valid frame within window
            while next_idx < min(
                current_idx + self.temporal_window, self.dataset.num_frames
            ):
                if self.is_valid_frame(next_idx):
                    break
                next_idx += 1

            # Check if we exceeded bounds
            if next_idx >= self.dataset.num_frames:
                # Use last sampled index
                sampled_indices.append(sampled_indices[-1])
                continue

            # Estimate motion
            motion = self._estimate_motion(current_idx, next_idx)

            # Adaptive interval: smaller interval for high motion
            if motion > self.motion_threshold:
                # High motion: sample more densely
                interval = max(1, int((next_idx - current_idx) / 2))
            else:
                # Low motion: sample sparsely
                interval = next_idx - current_idx

            # Update current index
            current_idx = min(current_idx + interval, next_idx)
            sampled_indices.append(current_idx)

        return sampled_indices[:self.num_samples]


def create_sampler(
    sampler_type: str,
    dataset,
    **kwargs
) -> BaseSampler:
    """
    Factory function to create sampler instance.

    Args:
        sampler_type: Type of sampler ("uniform", "random", "occlusion_aware", "adaptive")
        dataset: Dataset instance
        **kwargs: Additional arguments for specific sampler

    Returns:
        Sampler instance

    Raises:
        ValueError: If sampler_type is unknown
    """
    sampler_map = {
        "uniform": UniformSampler,
        "random": RandomSampler,
        "occlusion_aware": OcclusionAwareSampler,
        "adaptive": AdaptiveSampler,
    }

    if sampler_type not in sampler_map:
        raise ValueError(
            f"Unknown sampler type: {sampler_type}. "
            f"Available: {list(sampler_map.keys())}"
        )

    sampler_class = sampler_map[sampler_type]
    return sampler_class(dataset, **kwargs)