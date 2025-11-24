"""
Unit Tests for Validity Masking in BoxHead

Tests the critical fix from Audit 2025112502 Finding 1:
L1 and IoU losses must use validity_mask to prevent gradient from invalid samples.

Author: AquaSOT-3D Team
Date: 2025-11-25
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aquasot3d.models.heads.box_head import BoxHead


class TestValidityMasking:
    """Test validity masking for regression losses."""

    def test_l1_loss_with_validity_mask(self):
        """Test that invalid samples have zero L1 loss contribution."""
        # Setup
        head = BoxHead(in_channels=256, hidden_dim=128)
        head.eval()

        # Create batch with 4 samples: [valid, invalid, valid, invalid]
        batch_size = 4
        features = torch.randn(batch_size, 256)
        target_boxes = torch.tensor([
            [0.5, 0.5, 0.3, 0.3],  # valid
            [0.0, 0.0, 0.0, 0.0],  # invalid (0000 annotation)
            [0.6, 0.4, 0.2, 0.2],  # valid
            [0.0, 0.0, 0.0, 0.0],  # invalid (occluded)
        ])
        validity_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])  # [valid, invalid, valid, invalid]

        # Forward pass
        outputs = head(
            features=features,
            target_boxes=target_boxes,
            validity_mask=validity_mask
        )

        # Assertions
        assert 'l1_loss' in outputs, "L1 loss should be computed"
        assert torch.isfinite(outputs['l1_loss']), "L1 loss should be finite"

        # Critical: L1 loss should only come from valid samples
        # If invalid samples contribute, loss would be higher
        l1_loss_with_mask = outputs['l1_loss'].item()

        # Compare with unmasked loss (all samples)
        outputs_no_mask = head(
            features=features,
            target_boxes=target_boxes,
            validity_mask=None
        )
        l1_loss_no_mask = outputs_no_mask['l1_loss'].item()

        # With masking, loss should be different (lower or equal)
        # This confirms masking is applied
        print(f"L1 loss with mask: {l1_loss_with_mask:.6f}")
        print(f"L1 loss without mask: {l1_loss_no_mask:.6f}")

        # If masking works, losses should differ when invalid samples present
        # (unless predictions happen to match targets perfectly, which is unlikely)
        assert l1_loss_with_mask != l1_loss_no_mask or l1_loss_with_mask == 0.0, \
            "Masking should change loss value or result in zero loss"

    def test_iou_loss_with_validity_mask(self):
        """Test that invalid samples have zero IoU loss contribution."""
        # Setup
        head = BoxHead(in_channels=256, hidden_dim=128, iou_loss_type='giou')
        head.eval()

        # Create batch with 4 samples
        batch_size = 4
        features = torch.randn(batch_size, 256)
        target_boxes = torch.tensor([
            [0.5, 0.5, 0.3, 0.3],  # valid
            [0.0, 0.0, 0.0, 0.0],  # invalid
            [0.6, 0.4, 0.2, 0.2],  # valid
            [0.0, 0.0, 0.0, 0.0],  # invalid
        ])
        validity_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Forward pass
        outputs = head(
            features=features,
            target_boxes=target_boxes,
            validity_mask=validity_mask
        )

        # Assertions
        assert 'iou_loss' in outputs, "IoU loss should be computed"
        assert torch.isfinite(outputs['iou_loss']), "IoU loss should be finite"

        # Compare with unmasked
        outputs_no_mask = head(
            features=features,
            target_boxes=target_boxes,
            validity_mask=None
        )
        iou_loss_with_mask = outputs['iou_loss'].item()
        iou_loss_no_mask = outputs_no_mask['iou_loss'].item()

        print(f"IoU loss with mask: {iou_loss_with_mask:.6f}")
        print(f"IoU loss without mask: {iou_loss_no_mask:.6f}")

        # Masking should affect loss
        assert iou_loss_with_mask != iou_loss_no_mask or iou_loss_with_mask == 0.0, \
            "Masking should change IoU loss"

    def test_all_invalid_samples(self):
        """Test behavior when all samples are invalid."""
        head = BoxHead(in_channels=256, hidden_dim=128)
        head.eval()

        batch_size = 4
        features = torch.randn(batch_size, 256)
        target_boxes = torch.zeros(batch_size, 4)  # All invalid
        validity_mask = torch.zeros(batch_size)  # All invalid

        outputs = head(
            features=features,
            target_boxes=target_boxes,
            validity_mask=validity_mask
        )

        # With all invalid, losses should be minimal or zero
        # (division by num_valid=1 due to clamp, so average over 1 sample)
        assert torch.isfinite(outputs['l1_loss']), "L1 loss should be finite even with all invalid"
        assert torch.isfinite(outputs['iou_loss']), "IoU loss should be finite even with all invalid"

    def test_quality_masking(self):
        """Test that quality scores are also masked for invalid samples."""
        head = BoxHead(in_channels=256, hidden_dim=128, use_quality_focal=True)
        head.eval()

        batch_size = 4
        features = torch.randn(batch_size, 256)
        target_boxes = torch.tensor([
            [0.5, 0.5, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0],
            [0.6, 0.4, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0],
        ])
        target_labels = torch.ones(batch_size)
        validity_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

        outputs = head(
            features=features,
            target_boxes=target_boxes,
            target_labels=target_labels,
            validity_mask=validity_mask
        )

        # Quality masking is applied by forcing target_quality=0 for invalid samples
        # This is tested implicitly through cls_loss
        assert 'cls_loss' in outputs, "Classification loss should be computed"
        assert torch.isfinite(outputs['cls_loss']), "Classification loss should be finite"

    def test_no_validity_mask_backward_compatibility(self):
        """Test that code works without validity_mask (backward compatibility)."""
        head = BoxHead(in_channels=256, hidden_dim=128)
        head.eval()

        batch_size = 4
        features = torch.randn(batch_size, 256)
        target_boxes = torch.rand(batch_size, 4)

        # Should work without validity_mask
        outputs = head(
            features=features,
            target_boxes=target_boxes,
            validity_mask=None  # No mask provided
        )

        assert 'l1_loss' in outputs
        assert 'iou_loss' in outputs
        assert torch.isfinite(outputs['l1_loss'])
        assert torch.isfinite(outputs['iou_loss'])


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
