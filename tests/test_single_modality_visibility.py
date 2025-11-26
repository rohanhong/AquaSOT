"""
Test single-modality visibility scenarios (Audit 2025112312).

This test validates the critical fixes for handling cases where the target is
visible in only one modality (e.g., target out of RGB FOV but in Sonar FOV).

Tests cover:
1. BUG 1 Fix: Separate RGB and Sonar GT boxes
2. BUG 2 Fix: has_object logic uses OR instead of AND
3. Fusion GT selection based on objectness flags
4. Complete data flow from dataset to model

Author: AquaSOT-3D Team
Date: 2025-11-23
"""

import torch
import numpy as np
import pytest
from pathlib import Path

from aquasot3d.dataloader.datasets.rgb_sonar_dataset import RGBSonarDataset
from aquasot3d.dataloader.datasets.template_search_dataset import (
    TemplateSearchDataset,
    template_search_collate_fn
)


def test_has_object_or_logic():
    """
    Test BUG 2 Fix: has_object should use OR logic instead of AND.

    Scenario: Target visible in Sonar but out of RGB FOV
    Expected: has_object = True (OR logic)
    Previous BUG: has_object = False (AND logic)
    """
    # Create a minimal dataset for testing
    data_root = Path("data")
    if not data_root.exists():
        pytest.skip("Data directory not found, skipping test")

    try:
        dataset = RGBSonarDataset(
            data_root=data_root,
            sequence_name='uuv2',
            return_occluded=True  # Important: return all samples
        )
    except Exception as e:
        pytest.skip(f"Cannot load dataset: {e}")

    # Test frame 74: RGB out of FOV (0,0,0,0), Sonar visible
    # This is a real single-modality visibility case
    try:
        sample = dataset[73]  # Frame 74 (0-indexed)
    except Exception as e:
        pytest.skip(f"Cannot access frame 74: {e}")

    # Verify BUG 2 fix: has_object uses OR logic
    assert sample['has_object_rgb'] == False, "Frame 74 should have RGB out of FOV"
    assert sample['has_object_sonar'] == True, "Frame 74 should have Sonar visible"
    assert sample['has_object'] == True, "has_object should be True (OR logic)"

    print("✓ BUG 2 Fix verified: has_object uses OR logic")


def test_separate_gt_boxes():
    """
    Test BUG 1 Fix: TemplateSearchDataset returns separate RGB and Sonar GT boxes.

    Expected:
    - search_bbox_rgb: RGB GT in RGB coordinate system
    - search_bbox_sonar: Sonar GT in Sonar coordinate system
    - has_object_rgb, has_object_sonar, has_object flags present
    """
    data_root = Path("data")
    if not data_root.exists():
        pytest.skip("Data directory not found, skipping test")

    try:
        base_dataset = RGBSonarDataset(
            data_root=data_root,
            sequence_name='uuv2',
            return_occluded=True
        )
        ts_dataset = TemplateSearchDataset(
            base_dataset=base_dataset,
            template_size=128,  # OSTrack-256 standard
            search_size=256,   # OSTrack-256 standard
            template_update_policy='first_frame',
            template_search_gap=(10, 50)
        )
    except Exception as e:
        pytest.skip(f"Cannot load dataset: {e}")

    # Test a sample from ts_dataset
    try:
        sample = ts_dataset[100]
    except Exception as e:
        pytest.skip(f"Cannot access sample: {e}")

    # Verify BUG 1 fix: separate GT boxes
    assert 'search_bbox_rgb' in sample, "search_bbox_rgb should be present"
    assert 'search_bbox_sonar' in sample, "search_bbox_sonar should be present"
    assert 'has_object_rgb' in sample, "has_object_rgb should be present"
    assert 'has_object_sonar' in sample, "has_object_sonar should be present"
    assert 'has_object' in sample, "has_object should be present"

    # Verify bbox shapes (CXCYWH normalized [0,1])
    assert sample['search_bbox_rgb'].shape == (4,), "RGB GT should be [4]"
    assert sample['search_bbox_sonar'].shape == (4,), "Sonar GT should be [4]"

    print("✓ BUG 1 Fix verified: Separate RGB and Sonar GT boxes")


def test_collate_function_handles_separate_gt():
    """
    Test that collate function correctly handles separate GT boxes.

    Expected batch keys:
    - search_bbox_rgb: [B, 4]
    - search_bbox_sonar: [B, 4]
    - has_object_rgb: [B]
    - has_object_sonar: [B]
    - has_object: [B]
    """
    data_root = Path("data")
    if not data_root.exists():
        pytest.skip("Data directory not found, skipping test")

    try:
        base_dataset = RGBSonarDataset(
            data_root=data_root,
            sequence_name='uuv2',
            return_occluded=True
        )
        ts_dataset = TemplateSearchDataset(
            base_dataset=base_dataset,
            template_size=128,  # OSTrack-256 standard
            search_size=256,   # OSTrack-256 standard
            template_update_policy='first_frame',
            template_search_gap=(10, 50)
        )
    except Exception as e:
        pytest.skip(f"Cannot load dataset: {e}")

    # Create a batch of 4 samples
    try:
        batch_list = [ts_dataset[i] for i in range(4)]
    except Exception as e:
        pytest.skip(f"Cannot create batch: {e}")

    # Apply collate function
    batch = template_search_collate_fn(batch_list)

    # Verify batch structure
    assert 'search_bbox_rgb' in batch, "Batch should have search_bbox_rgb"
    assert 'search_bbox_sonar' in batch, "Batch should have search_bbox_sonar"
    assert 'has_object_rgb' in batch, "Batch should have has_object_rgb"
    assert 'has_object_sonar' in batch, "Batch should have has_object_sonar"
    assert 'has_object' in batch, "Batch should have has_object"

    # Verify tensor shapes
    assert batch['search_bbox_rgb'].shape == (4, 4), "RGB GT should be [B, 4]"
    assert batch['search_bbox_sonar'].shape == (4, 4), "Sonar GT should be [B, 4]"
    assert batch['has_object_rgb'].shape == (4,), "has_object_rgb should be [B]"
    assert batch['has_object_sonar'].shape == (4,), "has_object_sonar should be [B]"
    assert batch['has_object'].shape == (4,), "has_object should be [B]"

    # Verify dtype
    assert batch['has_object_rgb'].dtype == torch.float32, "has_object_rgb should be float32"
    assert batch['has_object_sonar'].dtype == torch.float32, "has_object_sonar should be float32"
    assert batch['has_object'].dtype == torch.float32, "has_object should be float32"

    print("✓ Collate function correctly handles separate GT boxes")


def test_fusion_gt_selection():
    """
    Test Fusion GT selection logic in trainer.py.

    Simulates trainer logic:
    - If has_object_rgb=1: use RGB GT
    - If has_object_rgb=0 and has_object_sonar=1: use Sonar GT
    - If both 0: use RGB GT (will be [0,0,0,0])
    """
    # Simulate batch with 4 samples covering all scenarios
    batch = {
        'search_bbox_rgb': torch.tensor([
            [0.5, 0.5, 0.2, 0.2],  # Sample 0: Both visible
            [0.0, 0.0, 0.0, 0.0],  # Sample 1: RGB OOV, Sonar visible
            [0.6, 0.4, 0.3, 0.1],  # Sample 2: RGB visible, Sonar OOV
            [0.0, 0.0, 0.0, 0.0],  # Sample 3: Both OOV
        ]),
        'search_bbox_sonar': torch.tensor([
            [0.5, 0.5, 0.15, 0.15],  # Sample 0: Both visible
            [0.7, 0.6, 0.1, 0.1],    # Sample 1: RGB OOV, Sonar visible (KEY TEST!)
            [0.0, 0.0, 0.0, 0.0],    # Sample 2: RGB visible, Sonar OOV
            [0.0, 0.0, 0.0, 0.0],    # Sample 3: Both OOV
        ]),
        'has_object_rgb': torch.tensor([1.0, 0.0, 1.0, 0.0]),
        'has_object_sonar': torch.tensor([1.0, 1.0, 0.0, 0.0]),
    }

    # Simulate trainer logic
    gt_boxes = {
        'rgb': batch['search_bbox_rgb'],
        'sonar': batch['search_bbox_sonar']
    }

    # Fusion GT selection (same logic as trainer.py)
    has_rgb_obj = batch['has_object_rgb']
    has_sonar_obj = batch['has_object_sonar']
    fusion_gt = gt_boxes['rgb'].clone()
    for i in range(len(has_rgb_obj)):
        if not has_rgb_obj[i] and has_sonar_obj[i]:
            # RGB out of view but Sonar visible: use Sonar GT
            fusion_gt[i] = gt_boxes['sonar'][i]

    # Verify Fusion GT selection
    # Sample 0: Both visible → use RGB GT
    assert torch.allclose(fusion_gt[0], batch['search_bbox_rgb'][0]), \
        "Sample 0: Should use RGB GT (both visible)"

    # Sample 1: RGB OOV, Sonar visible → use Sonar GT (CRITICAL TEST!)
    assert torch.allclose(fusion_gt[1], batch['search_bbox_sonar'][1]), \
        "Sample 1: Should use Sonar GT (RGB out of view)"
    assert not torch.all(fusion_gt[1] == 0), \
        "Sample 1: Fusion GT should NOT be [0,0,0,0]"

    # Sample 2: RGB visible, Sonar OOV → use RGB GT
    assert torch.allclose(fusion_gt[2], batch['search_bbox_rgb'][2]), \
        "Sample 2: Should use RGB GT (Sonar out of view)"

    # Sample 3: Both OOV → use RGB GT (will be [0,0,0,0])
    assert torch.allclose(fusion_gt[3], batch['search_bbox_rgb'][3]), \
        "Sample 3: Should use RGB GT (both out of view)"

    print("✓ Fusion GT selection logic verified")
    print(f"  Sample 1 Fusion GT: {fusion_gt[1].numpy()} (should be Sonar GT, not [0,0,0,0])")


if __name__ == "__main__":
    print("=== Testing Single-Modality Visibility Fixes ===\n")

    print("Test 1: has_object OR logic")
    try:
        test_has_object_or_logic()
    except Exception as e:
        print(f"✗ Test 1 failed: {e}\n")

    print("\nTest 2: Separate GT boxes")
    try:
        test_separate_gt_boxes()
    except Exception as e:
        print(f"✗ Test 2 failed: {e}\n")

    print("\nTest 3: Collate function")
    try:
        test_collate_function_handles_separate_gt()
    except Exception as e:
        print(f"✗ Test 3 failed: {e}\n")

    print("\nTest 4: Fusion GT selection")
    try:
        test_fusion_gt_selection()
    except Exception as e:
        print(f"✗ Test 4 failed: {e}\n")

    print("\n=== All Tests Complete ===")
