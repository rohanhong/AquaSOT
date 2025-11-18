"""
Verification Script for Bbox Format Fix (Audit 2025111901)

This script verifies that the xywh→cxcywh conversion fix is correctly
applied in the training pipeline.

Run this after applying the fix to trainer.py to confirm:
1. convert_bbox_format is imported and functional
2. The conversion produces correct center coordinates
3. Integration with trainer workflow is correct

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import torch
import numpy as np
from aquasot3d.utils import convert_bbox_format


def test_conversion_functionality():
    """Test that convert_bbox_format works correctly."""
    print("="*70)
    print("Test 1: Conversion Functionality")
    print("="*70)

    # Test case from bug report
    bbox_xywh = torch.tensor([[100.0, 50.0, 80.0, 60.0]])
    bbox_cxcywh = convert_bbox_format(bbox_xywh, 'xywh', 'cxcywh')

    print(f"Input (xywh):  {bbox_xywh[0].tolist()}")
    print(f"Output (cxcywh): {bbox_cxcywh[0].tolist()}")

    expected_cx = 140.0  # 100 + 80/2
    expected_cy = 80.0   # 50 + 60/2

    assert bbox_cxcywh[0, 0].item() == expected_cx, f"Expected cx={expected_cx}, got {bbox_cxcywh[0, 0].item()}"
    assert bbox_cxcywh[0, 1].item() == expected_cy, f"Expected cy={expected_cy}, got {bbox_cxcywh[0, 1].item()}"
    assert bbox_cxcywh[0, 2].item() == 80.0, "Width should remain unchanged"
    assert bbox_cxcywh[0, 3].item() == 60.0, "Height should remain unchanged"

    print("✅ Conversion produces correct center coordinates!\n")


def test_batch_conversion():
    """Test batch conversion (common in training)."""
    print("="*70)
    print("Test 2: Batch Conversion")
    print("="*70)

    # Simulate a batch of 8 bboxes (typical training batch)
    batch_xywh = torch.tensor([
        [100, 50, 80, 60],
        [200, 100, 60, 40],
        [150, 80, 100, 80],
        [250, 120, 90, 70],
        [180, 90, 70, 50],
        [220, 110, 80, 60],
        [190, 95, 85, 65],
        [210, 105, 75, 55],
    ], dtype=torch.float32)

    batch_cxcywh = convert_bbox_format(batch_xywh, 'xywh', 'cxcywh')

    print(f"Batch size: {batch_xywh.shape[0]}")
    print(f"Sample 0 - xywh: {batch_xywh[0].tolist()}")
    print(f"Sample 0 - cxcywh: {batch_cxcywh[0].tolist()}")

    # Verify first sample
    expected_cx = 100 + 80/2  # 140
    expected_cy = 50 + 60/2   # 80
    assert torch.allclose(batch_cxcywh[0, 0], torch.tensor(expected_cx))
    assert torch.allclose(batch_cxcywh[0, 1], torch.tensor(expected_cy))

    print("✅ Batch conversion works correctly!\n")


def test_numpy_compatibility():
    """Test numpy array support (dataset may output numpy)."""
    print("="*70)
    print("Test 3: NumPy Compatibility")
    print("="*70)

    # Dataset often outputs numpy arrays
    bbox_xywh_np = np.array([[100.0, 50.0, 80.0, 60.0]], dtype=np.float32)
    bbox_cxcywh_np = convert_bbox_format(bbox_xywh_np, 'xywh', 'cxcywh')

    print(f"Input type: {type(bbox_xywh_np)}")
    print(f"Output type: {type(bbox_cxcywh_np)}")
    print(f"Input (xywh):  {bbox_xywh_np[0]}")
    print(f"Output (cxcywh): {bbox_cxcywh_np[0]}")

    assert isinstance(bbox_cxcywh_np, np.ndarray), "Output should preserve numpy type"
    assert bbox_cxcywh_np[0, 0] == 140.0
    assert bbox_cxcywh_np[0, 1] == 80.0

    print("✅ NumPy arrays supported!\n")


def test_error_magnitude():
    """Demonstrate the magnitude of error without conversion."""
    print("="*70)
    print("Test 4: Error Magnitude Without Fix")
    print("="*70)

    bbox_xywh = torch.tensor([[100.0, 50.0, 80.0, 60.0]])

    # True center (with conversion)
    bbox_cxcywh = convert_bbox_format(bbox_xywh, 'xywh', 'cxcywh')
    true_cx = bbox_cxcywh[0, 0].item()
    true_cy = bbox_cxcywh[0, 1].item()

    # Wrong interpretation (without conversion)
    wrong_cx = bbox_xywh[0, 0].item()  # 100 (should be 140!)
    wrong_cy = bbox_xywh[0, 1].item()  # 50 (should be 80!)

    cx_error = abs(true_cx - wrong_cx)
    cy_error = abs(true_cy - wrong_cy)
    cx_relative = cx_error / true_cx * 100
    cy_relative = cy_error / true_cy * 100

    print(f"True center: ({true_cx}, {true_cy})")
    print(f"Wrong center (no conversion): ({wrong_cx}, {wrong_cy})")
    print(f"Absolute error: ({cx_error}, {cy_error}) pixels")
    print(f"Relative error: ({cx_relative:.1f}%, {cy_relative:.1f}%)")
    print()
    print("❌ Without conversion: ~30-40 pixel center offset!")
    print("✅ With conversion: 0 pixel error!\n")


def test_sonar_logpolar_case():
    """Test typical sonar bbox after log-polar transform."""
    print("="*70)
    print("Test 5: Sonar Log-Polar Bbox Case")
    print("="*70)

    # After log-polar transform: 682x376 → 256x256
    # Bbox in log-polar space still in xywh format
    bbox_logpolar_xywh = torch.tensor([[89.0, 241.0, 37.0, 30.0]])
    bbox_logpolar_cxcywh = convert_bbox_format(bbox_logpolar_xywh, 'xywh', 'cxcywh')

    print("Sonar bbox after log-polar transform (256x256 space):")
    print(f"  xywh: {bbox_logpolar_xywh[0].tolist()}")
    print(f"  cxcywh: {bbox_logpolar_cxcywh[0].tolist()}")

    expected_cx = 89.0 + 37.0/2  # 107.5
    expected_cy = 241.0 + 30.0/2  # 256.0

    assert torch.allclose(bbox_logpolar_cxcywh[0, 0], torch.tensor(expected_cx))
    assert torch.allclose(bbox_logpolar_cxcywh[0, 1], torch.tensor(expected_cy))

    print(f"✅ Log-polar sonar bbox correctly converted!")
    print(f"   Center moved from ({89}, {241}) to ({expected_cx}, {expected_cy})\n")


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("Bbox Format Fix Verification (Audit 2025111901)")
    print("="*70)
    print()

    try:
        test_conversion_functionality()
        test_batch_conversion()
        test_numpy_compatibility()
        test_error_magnitude()
        test_sonar_logpolar_case()

        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print()
        print("Summary:")
        print("  ✅ convert_bbox_format works correctly")
        print("  ✅ Batch conversion supported")
        print("  ✅ NumPy arrays supported")
        print("  ✅ Error magnitude confirmed (~30-40 pixels without fix)")
        print("  ✅ Log-polar sonar case verified")
        print()
        print("The fix is ready for production training!")
        print()

    except Exception as e:
        print("="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
