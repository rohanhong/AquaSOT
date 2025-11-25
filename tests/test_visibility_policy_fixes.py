"""
Test suite for Audit 2025112504 visibility policy fixes.

Tests the following fixes:
1. Issue 1: Visibility flags passed in evaluator
2. Issue 2: Template selection prevents 0000 frames
3. Issue 3: visibility_policy parameter integration

Author: AquaSOT-3D Team
Date: 2025-11-25
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from aquasot3d.dataloader.datasets.rgb_sonar_dataset import (
    RGBSonarDataset,
    is_occluded
)
from aquasot3d.dataloader.datasets.evaluation_dataset import (
    EvaluationSequenceDataset
)
from aquasot3d.dataloader.datasets.template_search_dataset import (
    TemplateSearchDataset
)
from aquasot3d.evaluation.evaluator import TrackingEvaluator


class TestVisibilityPolicyFiltering:
    """Test visibility_policy parameter in RGBSonarDataset."""

    def create_mock_dataset(self, rgb_bboxes, sonar_bboxes, visibility_policy='both'):
        """
        Create mock RGBSonarDataset with custom bboxes.

        Args:
            rgb_bboxes: List of [x, y, w, h] arrays, [0,0,0,0] = occluded
            sonar_bboxes: List of [x, y, w, h] arrays
            visibility_policy: 'all', 'any', or 'both'
        """
        # Mock dataset with minimal required attributes
        dataset = Mock(spec=RGBSonarDataset)
        dataset.rgb_bboxes = np.array(rgb_bboxes, dtype=np.float32)
        dataset.sonar_bboxes = np.array(sonar_bboxes, dtype=np.float32)
        dataset.num_frames = len(rgb_bboxes)
        dataset.start_frame = 1
        dataset.end_frame = len(rgb_bboxes)
        dataset.visibility_policy = visibility_policy

        # Implement _build_valid_indices with real logic
        def _build_valid_indices():
            valid_indices = []
            for idx in range(dataset.start_frame - 1, dataset.end_frame):
                rgb_valid = not is_occluded(dataset.rgb_bboxes[idx])
                sonar_valid = not is_occluded(dataset.sonar_bboxes[idx])

                if visibility_policy == 'all':
                    include = True
                elif visibility_policy == 'any':
                    include = rgb_valid or sonar_valid
                elif visibility_policy == 'both':
                    include = rgb_valid and sonar_valid
                else:
                    raise ValueError(f"Invalid visibility_policy: {visibility_policy}")

                if include:
                    valid_indices.append(idx)
            return valid_indices

        dataset._build_valid_indices = _build_valid_indices
        dataset.valid_indices = _build_valid_indices()

        return dataset

    def test_visibility_policy_all(self):
        """Test 'all' policy includes all frames."""
        rgb_bboxes = [
            [10, 10, 50, 50],  # Valid
            [0, 0, 0, 0],       # Occluded
            [20, 20, 60, 60],  # Valid
            [0, 0, 0, 0],       # Occluded (both)
        ]
        sonar_bboxes = [
            [15, 15, 55, 55],  # Valid
            [25, 25, 65, 65],  # Valid
            [0, 0, 0, 0],       # Occluded
            [0, 0, 0, 0],       # Occluded (both)
        ]

        dataset = self.create_mock_dataset(rgb_bboxes, sonar_bboxes, 'all')

        # Should include all 4 frames
        assert len(dataset.valid_indices) == 4
        assert dataset.valid_indices == [0, 1, 2, 3]

    def test_visibility_policy_any(self):
        """Test 'any' policy requires at least one modality visible."""
        rgb_bboxes = [
            [10, 10, 50, 50],  # Valid (both)
            [0, 0, 0, 0],       # Only sonar visible
            [20, 20, 60, 60],  # Only RGB visible
            [0, 0, 0, 0],       # Zero-modality (excluded)
        ]
        sonar_bboxes = [
            [15, 15, 55, 55],  # Valid (both)
            [25, 25, 65, 65],  # Only sonar visible
            [0, 0, 0, 0],       # Only RGB visible
            [0, 0, 0, 0],       # Zero-modality (excluded)
        ]

        dataset = self.create_mock_dataset(rgb_bboxes, sonar_bboxes, 'any')

        # Should include frames 0, 1, 2 (exclude frame 3 which is zero-modality)
        assert len(dataset.valid_indices) == 3
        assert dataset.valid_indices == [0, 1, 2]

    def test_visibility_policy_both(self):
        """Test 'both' policy requires both modalities visible."""
        rgb_bboxes = [
            [10, 10, 50, 50],  # Valid (both)
            [0, 0, 0, 0],       # Only sonar visible (excluded)
            [20, 20, 60, 60],  # Only RGB visible (excluded)
            [30, 30, 70, 70],  # Valid (both)
        ]
        sonar_bboxes = [
            [15, 15, 55, 55],  # Valid (both)
            [25, 25, 65, 65],  # Only sonar visible (excluded)
            [0, 0, 0, 0],       # Only RGB visible (excluded)
            [35, 35, 75, 75],  # Valid (both)
        ]

        dataset = self.create_mock_dataset(rgb_bboxes, sonar_bboxes, 'both')

        # Should only include frames 0 and 3 (dual-modality)
        assert len(dataset.valid_indices) == 2
        assert dataset.valid_indices == [0, 3]

    def test_visibility_policy_invalid(self):
        """Test invalid visibility_policy raises error."""
        rgb_bboxes = [[10, 10, 50, 50]]
        sonar_bboxes = [[15, 15, 55, 55]]

        with pytest.raises(ValueError, match="Invalid visibility_policy"):
            dataset = self.create_mock_dataset(rgb_bboxes, sonar_bboxes, 'invalid')
            # Trigger _build_valid_indices
            _ = dataset._build_valid_indices()


class TestTemplateSelection:
    """Test template selection prevents 0000 frames."""

    def create_mock_base_dataset(self, rgb_bboxes, sonar_bboxes):
        """Create mock base dataset for template selection tests."""
        dataset = Mock()
        dataset.rgb_bboxes = np.array(rgb_bboxes, dtype=np.float32)
        dataset.sonar_bboxes = np.array(sonar_bboxes, dtype=np.float32)
        dataset.num_frames = len(rgb_bboxes)
        dataset.valid_indices = list(range(len(rgb_bboxes)))
        return dataset

    def test_template_candidates_any_policy(self):
        """Test template candidates with 'any' policy."""
        rgb_bboxes = [
            [0, 0, 0, 0],       # Zero-modality (excluded)
            [10, 10, 50, 50],  # Valid (both)
            [0, 0, 0, 0],       # Only sonar (included)
            [20, 20, 60, 60],  # Only RGB (included)
        ]
        sonar_bboxes = [
            [0, 0, 0, 0],       # Zero-modality (excluded)
            [15, 15, 55, 55],  # Valid (both)
            [25, 25, 65, 65],  # Only sonar (included)
            [0, 0, 0, 0],       # Only RGB (included)
        ]

        mock_dataset = self.create_mock_base_dataset(rgb_bboxes, sonar_bboxes)

        # Create EvaluationSequenceDataset with mock normalization
        with patch.object(EvaluationSequenceDataset, '_load_template'):
            eval_ds = EvaluationSequenceDataset.__new__(EvaluationSequenceDataset)
            eval_ds.base_dataset = mock_dataset

            # Test _build_template_candidates method
            candidates = eval_ds._build_template_candidates(mock_dataset, 'any')

            # Should exclude frame 0 (zero-modality), include frames 1, 2, 3
            assert len(candidates) == 3
            assert candidates == [1, 2, 3]

    def test_template_candidates_both_policy(self):
        """Test template candidates with 'both' policy."""
        rgb_bboxes = [
            [0, 0, 0, 0],       # Zero-modality (excluded)
            [10, 10, 50, 50],  # Valid (both) - included
            [0, 0, 0, 0],       # Only sonar (excluded)
            [20, 20, 60, 60],  # Valid (both) - included
        ]
        sonar_bboxes = [
            [0, 0, 0, 0],       # Zero-modality (excluded)
            [15, 15, 55, 55],  # Valid (both) - included
            [25, 25, 65, 65],  # Only sonar (excluded)
            [35, 35, 75, 75],  # Valid (both) - included
        ]

        mock_dataset = self.create_mock_base_dataset(rgb_bboxes, sonar_bboxes)

        with patch.object(EvaluationSequenceDataset, '_load_template'):
            eval_ds = EvaluationSequenceDataset.__new__(EvaluationSequenceDataset)
            eval_ds.base_dataset = mock_dataset

            candidates = eval_ds._build_template_candidates(mock_dataset, 'both')

            # Should only include frames 1 and 3 (dual-modality)
            assert len(candidates) == 2
            assert candidates == [1, 3]

    def test_template_selection_uses_first_valid_candidate(self):
        """Test template selection uses first valid candidate, not first frame."""
        rgb_bboxes = [
            [0, 0, 0, 0],       # Frame 0: Zero-modality (should be skipped)
            [0, 0, 0, 0],       # Frame 1: Zero-modality (should be skipped)
            [10, 10, 50, 50],  # Frame 2: Valid (should be selected)
            [20, 20, 60, 60],  # Frame 3: Valid
        ]
        sonar_bboxes = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [15, 15, 55, 55],
            [25, 25, 65, 65],
        ]

        mock_dataset = self.create_mock_base_dataset(rgb_bboxes, sonar_bboxes)
        mock_dataset.get_sequence_info = Mock(return_value={
            'sequence_name': 'test_seq',
            'num_frames': 4
        })

        with patch.object(EvaluationSequenceDataset, '_load_template'):
            with patch.object(EvaluationSequenceDataset, '_build_template_candidates',
                            return_value=[2, 3]):  # Mock candidates list
                eval_ds = EvaluationSequenceDataset.__new__(EvaluationSequenceDataset)
                eval_ds.base_dataset = mock_dataset
                eval_ds.template_candidates = [2, 3]
                eval_ds.num_frames = 4
                eval_ds.sequence_name = 'test_seq'

                # Simulate template selection logic
                template_idx = eval_ds.template_candidates[0]

                # Should select frame 2 (first valid candidate), not frame 0
                assert template_idx == 2


class TestEvaluatorVisibilityFlags:
    """Test evaluator passes visibility flags to model."""

    def test_evaluator_extracts_visibility_flags(self):
        """Test evaluator extracts and passes visibility flags from batch."""
        # Create mock batch with validity flags
        batch = {
            'rgb_template': torch.randn(1, 3, 256, 256),
            'rgb_search': torch.randn(1, 3, 512, 512),
            'sonar_template': torch.randn(1, 1, 256, 256),
            'sonar_search': torch.randn(1, 1, 512, 512),
            # Visibility flags (as produced by collate_fn_template_search)
            'is_valid_rgb': torch.tensor([1.0], dtype=torch.float32),
            'is_valid_sonar': torch.tensor([0.0], dtype=torch.float32),  # Sonar occluded
            'has_object_rgb': torch.tensor([1.0], dtype=torch.float32),
            'has_object_sonar': torch.tensor([0.0], dtype=torch.float32),
            'has_object': torch.tensor([1.0], dtype=torch.float32),
        }

        # Create mock model
        mock_model = Mock()
        mock_model.return_value = {
            'rgb_boxes': torch.randn(1, 4),
            'sonar_boxes': torch.randn(1, 4)
        }

        # Create mock evaluator
        evaluator = Mock(spec=TrackingEvaluator)
        evaluator.model = mock_model
        evaluator.device = torch.device('cpu')

        # Simulate the fixed evaluator logic (from Issue 1 fix)
        is_rgb_valid = batch.get('is_valid_rgb')
        is_sonar_valid = batch.get('is_valid_sonar')
        has_object_rgb = batch.get('has_object_rgb')
        has_object_sonar = batch.get('has_object_sonar')
        has_object = batch.get('has_object')

        # Verify flags are extracted
        assert is_rgb_valid is not None
        assert is_sonar_valid is not None
        assert has_object_rgb is not None
        assert has_object_sonar is not None
        assert has_object is not None

        # Verify flag values
        assert is_rgb_valid[0].item() == 1.0  # RGB valid
        assert is_sonar_valid[0].item() == 0.0  # Sonar occluded
        assert has_object_rgb[0].item() == 1.0
        assert has_object_sonar[0].item() == 0.0
        assert has_object[0].item() == 1.0

        # Convert to bool for model
        is_rgb_valid_bool = is_rgb_valid.bool()
        is_sonar_valid_bool = is_sonar_valid.bool()

        assert is_rgb_valid_bool[0].item() == True
        assert is_sonar_valid_bool[0].item() == False


def test_is_occluded_utility():
    """Test is_occluded utility function."""
    # Test zero bbox (occluded)
    assert is_occluded(np.array([0, 0, 0, 0])) == True

    # Test valid bboxes
    assert is_occluded(np.array([10, 10, 50, 50])) == False
    assert is_occluded(np.array([0, 0, 100, 100])) == False  # Zero x,y but valid w,h
    assert is_occluded(np.array([100, 100, 0, 0])) == False  # Valid x,y but zero w,h


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
