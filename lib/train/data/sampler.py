"""
Sampling module for AquaSOT training.

Based on OSTrack's TrackingSampler, extended for dual-modal RGB-Sonar data.
"""

import random
import torch
import torch.utils.data


def no_processing(data):
    """Default no-op processing function."""
    return data


class MultiModalSampler(torch.utils.data.Dataset):
    """Sampler for dual-modal RGB-Sonar tracking pairs.

    Samples template and search frame pairs from video sequences,
    ensuring both RGB and Sonar modalities are synchronized.

    Based on OSTrack's TrackingSampler with dual-modal extensions.
    """

    def __init__(
        self,
        datasets,
        p_datasets=None,
        samples_per_epoch=60000,
        max_gap=20,
        num_search_frames=1,
        num_template_frames=1,
        processing=no_processing,
        frame_sample_mode='causal',
    ):
        """
        Args:
            datasets: List of AquaSOTDataset instances
            p_datasets: List of probabilities for sampling each dataset
            samples_per_epoch: Number of samples per epoch
            max_gap: Maximum frame gap between template and search
            num_search_frames: Number of search frames to sample
            num_template_frames: Number of template frames to sample
            processing: Processing function for data augmentation
            frame_sample_mode: 'causal' (search after template) or 'interval'
        """
        self.datasets = datasets

        # Normalize probabilities
        if p_datasets is None:
            p_datasets = [len(d) for d in datasets]
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """Sample frame IDs where target is visible.

        Args:
            visible: Boolean tensor indicating visibility
            num_ids: Number of IDs to sample
            min_id: Minimum frame index
            max_id: Maximum frame index

        Returns:
            List of sampled frame indices, or None if not enough visible frames
        """
        if num_ids == 0:
            return []

        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        # Get valid (visible) frame indices
        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        if len(valid_ids) < num_ids:
            return None

        return random.sample(valid_ids, num_ids)

    def __getitem__(self, index):
        """Sample a training data point.

        Returns:
            dict containing:
                - rgb_template_images: List of RGB template images
                - rgb_search_images: List of RGB search images
                - sonar_template_images: List of sonar template images
                - sonar_search_images: List of sonar search images
                - rgb_template_anno: RGB template bboxes
                - rgb_search_anno: RGB search bboxes
                - sonar_template_anno: Sonar template bboxes
                - sonar_search_anno: Sonar search bboxes
                - valid: Whether the sample is valid
        """
        return self.getitem()

    def getitem(self):
        """Sample template and search frames."""
        valid = False

        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            # Sample a sequence
            seq_id, visible, seq_info = self._sample_sequence(dataset)

            # Sample frame IDs
            template_frame_ids, search_frame_ids = self._sample_frame_ids(visible)

            if template_frame_ids is None or search_frame_ids is None:
                continue

            try:
                # Get frames
                rgb_frames, sonar_frames, rgb_anno, sonar_anno, meta = \
                    dataset.get_frames(seq_id, template_frame_ids + search_frame_ids, seq_info)

                # Split into template and search
                n_t = self.num_template_frames
                n_s = self.num_search_frames

                data = {
                    'rgb_template_images': rgb_frames[:n_t],
                    'rgb_search_images': rgb_frames[n_t:n_t + n_s],
                    'sonar_template_images': sonar_frames[:n_t],
                    'sonar_search_images': sonar_frames[n_t:n_t + n_s],
                    'rgb_template_anno': rgb_anno['bbox'][:n_t],
                    'rgb_search_anno': rgb_anno['bbox'][n_t:n_t + n_s],
                    'sonar_template_anno': sonar_anno['bbox'][:n_t],
                    'sonar_search_anno': sonar_anno['bbox'][n_t:n_t + n_s],
                    'dataset': dataset.get_name(),
                    'sequence': meta.get('sequence_name', ''),
                }

                # Apply processing (augmentation, cropping, etc.)
                data = self.processing(data)

                # Check validity
                valid = data.get('valid', True)

            except Exception as e:
                # Skip invalid samples
                valid = False

        return data

    def _sample_sequence(self, dataset):
        """Sample a sequence with enough visible frames.

        Args:
            dataset: Dataset to sample from

        Returns:
            seq_id: Sequence index
            visible: Visibility tensor
            seq_info: Sequence information dict
        """
        enough_visible = False

        while not enough_visible:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)
            seq_info = dataset.get_sequence_info(seq_id)
            visible = seq_info['visible']

            # Need at least 2x (template + search) visible frames
            min_visible = 2 * (self.num_template_frames + self.num_search_frames)
            n_visible = visible.sum().item() if isinstance(visible, torch.Tensor) else sum(visible)

            enough_visible = n_visible > min_visible and len(visible) >= 20

        return seq_id, visible, seq_info

    def _sample_frame_ids(self, visible):
        """Sample template and search frame IDs.

        Args:
            visible: Visibility tensor

        Returns:
            template_frame_ids: List of template frame indices
            search_frame_ids: List of search frame indices
        """
        if self.frame_sample_mode == 'causal':
            return self._sample_causal(visible)
        else:
            return self._sample_interval(visible)

    def _sample_causal(self, visible):
        """Sample frames causally (search after template).

        Args:
            visible: Visibility tensor

        Returns:
            template_frame_ids, search_frame_ids
        """
        template_frame_ids = None
        search_frame_ids = None
        gap_increase = 0

        while search_frame_ids is None:
            # Sample base frame for template
            base_frame_id = self._sample_visible_ids(
                visible,
                num_ids=1,
                min_id=self.num_template_frames - 1,
                max_id=len(visible) - self.num_search_frames
            )

            if base_frame_id is None:
                gap_increase += 5
                if gap_increase > 100:
                    return None, None
                continue

            # Sample additional template frames before base
            if self.num_template_frames > 1:
                prev_frame_ids = self._sample_visible_ids(
                    visible,
                    num_ids=self.num_template_frames - 1,
                    min_id=max(0, base_frame_id[0] - self.max_gap - gap_increase),
                    max_id=base_frame_id[0]
                )
                if prev_frame_ids is None:
                    gap_increase += 5
                    continue
                template_frame_ids = base_frame_id + prev_frame_ids
            else:
                template_frame_ids = base_frame_id

            # Sample search frames after template
            search_frame_ids = self._sample_visible_ids(
                visible,
                num_ids=self.num_search_frames,
                min_id=template_frame_ids[0] + 1,
                max_id=min(len(visible), template_frame_ids[0] + self.max_gap + gap_increase)
            )

            gap_increase += 5

            if gap_increase > 100:
                return None, None

        return template_frame_ids, search_frame_ids

    def _sample_interval(self, visible):
        """Sample frames within an interval (non-causal).

        Args:
            visible: Visibility tensor

        Returns:
            template_frame_ids, search_frame_ids
        """
        # Sample template frame
        template_frame_ids = self._sample_visible_ids(
            visible,
            num_ids=self.num_template_frames
        )

        if template_frame_ids is None:
            return None, None

        # Sample search frame within max_gap of template
        center_id = template_frame_ids[0]
        search_frame_ids = self._sample_visible_ids(
            visible,
            num_ids=self.num_search_frames,
            min_id=max(0, center_id - self.max_gap),
            max_id=min(len(visible), center_id + self.max_gap)
        )

        return template_frame_ids, search_frame_ids
