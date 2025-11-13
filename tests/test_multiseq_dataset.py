"""
Test Multi-Sequence Dataset Functionality

This test validates the multi-sequence dataset implementation including:
- MultiSequenceRGBSonarDataset creation and indexing
- TrackingDataset intra-sequence temporal constraints
- Per-sequence resolution detection
- Global-to-local index mapping

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pytest
import torch
import numpy as np

from aquasot3d.dataloader import (
    MultiSequenceRGBSonarDataset,
    TrackingDataset,
)


@pytest.fixture
def multi_sequence_dataset():
    """Create a multi-sequence dataset for testing"""
    # Use small subset for fast testing
    dataset = MultiSequenceRGBSonarDataset(
        data_root='data',
        sequence_names=['uuv1', 'uuv2'],  # Only 2 sequences for speed
        return_occluded=True,
        sampling_policy='frame_balanced'
    )
    return dataset


def test_multiseq_creation(multi_sequence_dataset):
    """Test MultiSequenceRGBSonarDataset creation"""
    dataset = multi_sequence_dataset

    # Should have 2 sequences
    assert len(dataset.sequences) == 2

    # Should have non-zero total frames
    assert dataset.total_frames > 0

    # Check sequence metadata
    for seq_metadata in dataset.sequences:
        assert seq_metadata.sequence_name in ['uuv1', 'uuv2']
        assert seq_metadata.num_frames > 0
        assert len(seq_metadata.valid_indices) > 0
        assert seq_metadata.rgb_resolution is not None
        assert seq_metadata.sonar_resolution is not None


def test_multiseq_indexing(multi_sequence_dataset):
    """Test global indexing and sequence name propagation"""
    dataset = multi_sequence_dataset

    # Get first and last samples
    first_sample = dataset[0]
    last_sample = dataset[len(dataset) - 1]

    # Should have sequence_name field
    assert 'sequence_name' in first_sample
    assert 'sequence_name' in last_sample

    # Sequence names should be valid
    assert first_sample['sequence_name'] in ['uuv1', 'uuv2']
    assert last_sample['sequence_name'] in ['uuv1', 'uuv2']

    # Should have frame data
    assert 'rgb_img' in first_sample
    assert 'sonar_img' in first_sample
    assert 'rgb_bbox' in first_sample
    assert 'sonar_bbox' in first_sample


def test_multiseq_sequence_info(multi_sequence_dataset):
    """Test get_sequence_info functionality"""
    dataset = multi_sequence_dataset

    # Get info for all sequences
    all_info = dataset.get_sequence_info()
    assert isinstance(all_info, list)
    assert len(all_info) == 2

    # Get info for specific sequence
    uuv1_info = dataset.get_sequence_info('uuv1')
    assert isinstance(uuv1_info, dict)
    assert uuv1_info['sequence_name'] == 'uuv1'
    assert 'num_frames' in uuv1_info
    assert 'rgb_resolution' in uuv1_info
    assert 'sonar_resolution' in uuv1_info


def test_tracking_intra_sequence_constraint(multi_sequence_dataset):
    """Test TrackingDataset enforces intra-sequence temporal constraints"""
    base_dataset = multi_sequence_dataset

    # Create tracking dataset
    tracking_dataset = TrackingDataset(
        base_dataset=base_dataset,
        template_search_gap=30,
        min_gap=1,
        template_size=128,
        search_size=256
    )

    # Sample multiple pairs
    num_samples = min(100, len(tracking_dataset))
    violations = 0

    for i in range(num_samples):
        sample = tracking_dataset[i]

        # Check that sequence_name exists
        assert 'sequence_name' in sample, "Missing sequence_name in tracking sample"

        # Get frame indices
        template_idx = sample.get('template_idx')
        search_idx = sample.get('search_idx')

        if template_idx is not None and search_idx is not None:
            # Get sequence names for template and search
            template_seq = base_dataset.get_frame_sequence(template_idx)
            search_seq = base_dataset.get_frame_sequence(search_idx)

            # CRITICAL: Template and search must be from same sequence
            if template_seq != search_seq:
                violations += 1
                print(f"Violation at sample {i}: template={template_seq}, search={search_seq}")

    # Should have ZERO violations
    assert violations == 0, f"Found {violations} intra-sequence constraint violations!"


def test_sequence_boundary_detection(multi_sequence_dataset):
    """Test that sequence boundaries are correctly maintained"""
    dataset = multi_sequence_dataset

    # Iterate through all frames and track sequence transitions
    prev_seq = None
    transitions = []

    for i in range(len(dataset)):
        sample = dataset[i]
        curr_seq = sample['sequence_name']

        if prev_seq is not None and curr_seq != prev_seq:
            transitions.append((i, prev_seq, curr_seq))

        prev_seq = curr_seq

    # Should have exactly 1 transition (from uuv1 to uuv2)
    assert len(transitions) == 1, f"Expected 1 transition, found {len(transitions)}"

    # Verify transition sequence
    idx, from_seq, to_seq = transitions[0]
    assert from_seq == 'uuv1'
    assert to_seq == 'uuv2'


def test_resolution_detection(multi_sequence_dataset):
    """Test per-sequence resolution detection"""
    dataset = multi_sequence_dataset

    # Get resolutions for each sequence
    for seq_metadata in dataset.sequences:
        rgb_res = seq_metadata.rgb_resolution
        sonar_res = seq_metadata.sonar_resolution

        # Resolutions should be tuples of (width, height)
        assert isinstance(rgb_res, tuple)
        assert isinstance(sonar_res, tuple)
        assert len(rgb_res) == 2
        assert len(sonar_res) == 2

        # Resolutions should be positive integers
        assert rgb_res[0] > 0 and rgb_res[1] > 0
        assert sonar_res[0] > 0 and sonar_res[1] > 0

        print(f"Sequence {seq_metadata.sequence_name}: "
              f"RGB {rgb_res}, Sonar {sonar_res}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
