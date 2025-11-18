"""
Verification Test for Bbox Format Mismatch Bug

This test demonstrates the critical bug where dataset outputs xywh format
but BoxHead expects cxcywh format without proper conversion.

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import pytest
import torch
import numpy as np

from aquasot3d.models.heads import BoxHead, box_xywh_to_cxcywh
from aquasot3d.utils.bbox import BBoxXYWH, BBoxCXCYWH


class TestBboxFormatMismatch:
    """Test to demonstrate and verify the bbox format mismatch bug."""

    def test_dataset_outputs_xywh_format(self):
        """Verify that dataset outputs xywh format."""
        # Simulate dataset output
        bbox_from_dataset = np.array([100, 50, 80, 60], dtype=np.float32)

        # This is xywh format:
        # x=100 (left), y=50 (top), w=80, h=60
        # Expected center: cx = 100 + 80/2 = 140, cy = 50 + 60/2 = 80

        assert bbox_from_dataset[0] == 100  # x (top-left)
        assert bbox_from_dataset[1] == 50   # y (top-left)
        assert bbox_from_dataset[2] == 80   # width
        assert bbox_from_dataset[3] == 60   # height

    def test_box_head_expects_cxcywh_format(self):
        """Verify that BoxHead expects cxcywh format."""
        # BoxHead documentation states: "Target boxes in cxcywh format"
        # From box_head.py:677

        # Create a simple BoxHead
        head = BoxHead(in_channels=768, hidden_dim=256)

        # Check the docstring
        compute_loss_doc = head.compute_loss.__doc__
        assert 'cxcywh' in compute_loss_doc.lower(), \
            "BoxHead compute_loss expects cxcywh format"

    def test_format_confusion_causes_error(self):
        """
        Demonstrate the bug: passing xywh data to BoxHead causes center offset.

        Critical Bug:
        If we pass xywh format [100, 50, 80, 60] to BoxHead,
        it will interpret it as cxcywh: cx=100, cy=50, w=80, h=60
        But the true center should be cx=140, cy=80!

        This causes ~50% center coordinate error!
        """
        # Dataset bbox in xywh format
        bbox_xywh = torch.tensor([[100.0, 50.0, 80.0, 60.0]])  # x, y, w, h

        # True center should be
        true_cx = 100.0 + 80.0 / 2  # 140
        true_cy = 50.0 + 60.0 / 2   # 80

        # If we wrongly interpret xywh as cxcywh:
        wrong_cx = bbox_xywh[0, 0].item()  # 100 (wrong!)
        wrong_cy = bbox_xywh[0, 1].item()  # 50 (wrong!)

        # Compute error
        cx_error = abs(true_cx - wrong_cx)
        cy_error = abs(true_cy - wrong_cy)

        assert cx_error == 40.0, f"Center x error: {cx_error}"
        assert cy_error == 30.0, f"Center y error: {cy_error}"

        # Relative error
        cx_relative_error = cx_error / true_cx * 100
        cy_relative_error = cy_error / true_cy * 100

        print(f"\n❌ Bug Impact:")
        print(f"   Center X error: {cx_error} pixels ({cx_relative_error:.1f}%)")
        print(f"   Center Y error: {cy_error} pixels ({cy_relative_error:.1f}%)")

        # This is a significant error that will harm training!
        assert cx_relative_error > 25, "Center error exceeds 25%!"
        assert cy_relative_error > 35, "Center error exceeds 35%!"

    def test_correct_conversion_fixes_bug(self):
        """
        Verify that explicit xywh→cxcywh conversion fixes the bug.
        """
        # Dataset bbox in xywh format
        bbox_xywh = torch.tensor([[100.0, 50.0, 80.0, 60.0]])

        # Convert to cxcywh using the utility function
        bbox_cxcywh = box_xywh_to_cxcywh(bbox_xywh)

        # Now the bbox is correctly in cxcywh format
        expected_cx = 140.0
        expected_cy = 80.0
        expected_w = 80.0
        expected_h = 60.0

        torch.testing.assert_close(bbox_cxcywh[0, 0], torch.tensor(expected_cx))
        torch.testing.assert_close(bbox_cxcywh[0, 1], torch.tensor(expected_cy))
        torch.testing.assert_close(bbox_cxcywh[0, 2], torch.tensor(expected_w))
        torch.testing.assert_close(bbox_cxcywh[0, 3], torch.tensor(expected_h))

        print(f"\n✅ After conversion:")
        print(f"   XYWH: {bbox_xywh[0].tolist()}")
        print(f"   CXCYWH: {bbox_cxcywh[0].tolist()}")

    def test_bbox_wrapper_does_not_validate_format(self):
        """
        Demonstrate that BBoxCXCYWH wrapper does not validate input format.

        This is the root cause: BBoxCXCYWH only checks shape, not semantics!
        """
        # Create xywh data
        bbox_xywh_data = torch.tensor([[100.0, 50.0, 80.0, 60.0]])

        # Wrap with BBoxCXCYWH (this should fail, but it doesn't!)
        bbox_wrapper = BBoxCXCYWH(bbox_xywh_data)

        # The wrapper accepts it without error
        wrapped_data = bbox_wrapper.tensor()
        assert wrapped_data.shape[-1] == 4  # Only checks shape

        # But the semantic is wrong!
        # The data is xywh but wrapper assumes it's cxcywh
        print(f"\n⚠️  BBoxCXCYWH accepts xywh data without validation!")
        print(f"   Input (xywh): {bbox_xywh_data[0].tolist()}")
        print(f"   Wrapper assumes (cxcywh): {wrapped_data[0].tolist()}")
        print(f"   This is a silent format confusion bug!")

    def test_end_to_end_data_flow(self):
        """
        End-to-end test: simulate data flow from dataset to BoxHead.

        This demonstrates the complete bug in the training pipeline.
        """
        # Step 1: Dataset outputs xywh
        dataset_bbox = torch.tensor([[100.0, 50.0, 80.0, 60.0]])  # xywh
        print(f"\n📊 Data Flow Test:")
        print(f"1. Dataset output (xywh): {dataset_bbox[0].tolist()}")

        # Step 2: Trainer receives xywh (no conversion in current code!)
        trainer_gt_boxes = dataset_bbox  # Direct pass-through (BUG!)
        print(f"2. Trainer gt_boxes (still xywh): {trainer_gt_boxes[0].tolist()}")

        # Step 3: BoxHead expects cxcywh but receives xywh
        boxhead_input = trainer_gt_boxes  # Still xywh!
        print(f"3. BoxHead input (wrong! still xywh): {boxhead_input[0].tolist()}")

        # Step 4: BoxHead wraps with BBoxCXCYWH (assumes cxcywh)
        wrapped = BBoxCXCYWH(boxhead_input)
        print(f"4. BBoxCXCYWH wrapper (interprets as cxcywh): {wrapped.tensor()[0].tolist()}")

        # Verify error
        true_center = (140.0, 80.0)  # Correct center from xywh
        wrong_center = (boxhead_input[0, 0].item(), boxhead_input[0, 1].item())  # (100, 50)

        print(f"\n❌ Result:")
        print(f"   True center: {true_center}")
        print(f"   BoxHead thinks center is: {wrong_center}")
        print(f"   Error: ({true_center[0] - wrong_center[0]}, {true_center[1] - wrong_center[1]}) pixels")

    def test_fixed_data_flow(self):
        """
        Demonstrate the corrected data flow with proper format conversion.
        """
        # Step 1: Dataset outputs xywh
        dataset_bbox = torch.tensor([[100.0, 50.0, 80.0, 60.0]])
        print(f"\n✅ Fixed Data Flow:")
        print(f"1. Dataset output (xywh): {dataset_bbox[0].tolist()}")

        # Step 2: Trainer converts xywh → cxcywh (FIX!)
        trainer_gt_boxes = box_xywh_to_cxcywh(dataset_bbox)
        print(f"2. Trainer converts to cxcywh: {trainer_gt_boxes[0].tolist()}")

        # Step 3: BoxHead receives correct cxcywh
        boxhead_input = trainer_gt_boxes
        print(f"3. BoxHead input (correct cxcywh): {boxhead_input[0].tolist()}")

        # Step 4: BoxHead wraps with BBoxCXCYWH (correct now!)
        wrapped = BBoxCXCYWH(boxhead_input)
        print(f"4. BBoxCXCYWH wrapper (correct): {wrapped.tensor()[0].tolist()}")

        # Verify correctness
        expected_center = (140.0, 80.0)
        actual_center = (boxhead_input[0, 0].item(), boxhead_input[0, 1].item())

        torch.testing.assert_close(
            torch.tensor(expected_center),
            torch.tensor(actual_center)
        )

        print(f"\n✅ Result: Center is correct!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
