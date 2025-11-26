"""
Unit Tests for 2D Box Prediction Heads

Tests for BoxHead, RGBBoxHead, SonarBoxHead and related utilities.

ARCHITECTURE UPDATE (Audit 2025112313): FusionBoxHead removed (dual-view fusion architecture).
CRITICAL FIX (Audit 2025112601): Loss weights corrected to match OSTrack (L1=5.0, GIoU=2.0).

Author: AquaSOT-3D Team
Date: 2025-11-10
Updated: 2025-11-26
"""

import pytest
import torch
import torch.nn as nn
import math

from aquasot3d.models.heads import (
    BoxHead,
    RGBBoxHead,
    SonarBoxHead,
    build_box_head,
    GIoULoss,
    CIoULoss,
    QualityFocalLoss,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    compute_iou,
    compute_giou,
    compute_ciou,
)


class TestBoxUtilities:
    """Tests for box format conversion and IoU computation utilities."""

    def test_box_cxcywh_to_xyxy(self):
        """Test conversion from center format to corner format."""
        boxes_cxcywh = torch.tensor([
            [50, 50, 20, 30],  # cx=50, cy=50, w=20, h=30
            [100, 100, 40, 40],
        ], dtype=torch.float32)

        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)

        expected = torch.tensor([
            [40, 35, 60, 65],  # x1=40, y1=35, x2=60, y2=65
            [80, 80, 120, 120],
        ], dtype=torch.float32)

        assert torch.allclose(boxes_xyxy, expected, atol=1e-5)

    def test_box_xyxy_to_cxcywh(self):
        """Test conversion from corner format to center format."""
        boxes_xyxy = torch.tensor([
            [40, 35, 60, 65],
            [80, 80, 120, 120],
        ], dtype=torch.float32)

        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)

        expected = torch.tensor([
            [50, 50, 20, 30],
            [100, 100, 40, 40],
        ], dtype=torch.float32)

        assert torch.allclose(boxes_cxcywh, expected, atol=1e-5)

    def test_box_conversion_inverse(self):
        """Test that converting back and forth preserves boxes."""
        # Generate valid xyxy boxes where x2 > x1 and y2 > y1
        x1y1 = torch.rand(10, 2) * 50  # x1, y1 in [0, 50]
        wh = torch.rand(10, 2) * 50 + 10  # w, h in [10, 60]
        x2y2 = x1y1 + wh  # x2 = x1 + w, y2 = y1 + h
        original = torch.cat([x1y1, x2y2], dim=1)  # [x1, y1, x2, y2]

        converted = box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(original))
        assert torch.allclose(converted, original, atol=1e-4)

    def test_compute_iou_perfect_overlap(self):
        """Test IoU computation for perfect overlap."""
        boxes1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        boxes2 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)

        iou = compute_iou(boxes1, boxes2)
        assert torch.allclose(iou, torch.tensor([1.0]), atol=1e-5)

    def test_compute_iou_no_overlap(self):
        """Test IoU computation for no overlap."""
        boxes1 = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
        boxes2 = torch.tensor([[30, 30, 40, 40]], dtype=torch.float32)

        iou = compute_iou(boxes1, boxes2)
        assert torch.allclose(iou, torch.tensor([0.0]), atol=1e-5)

    def test_compute_iou_partial_overlap(self):
        """Test IoU computation for partial overlap."""
        boxes1 = torch.tensor([[0, 0, 20, 20]], dtype=torch.float32)
        boxes2 = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)

        iou = compute_iou(boxes1, boxes2)

        # Intersection: 10x10 = 100
        # Union: 400 + 400 - 100 = 700
        # IoU: 100/700 ≈ 0.1429
        expected = torch.tensor([100.0 / 700.0])
        assert torch.allclose(iou, expected, atol=1e-4)

    def test_compute_giou_perfect_overlap(self):
        """Test GIoU computation for perfect overlap."""
        boxes1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        boxes2 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)

        giou = compute_giou(boxes1, boxes2)
        assert torch.allclose(giou, torch.tensor([1.0]), atol=1e-5)

    def test_compute_giou_no_overlap(self):
        """Test GIoU computation for no overlap."""
        boxes1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        boxes2 = torch.tensor([[20, 20, 30, 30]], dtype=torch.float32)

        giou = compute_giou(boxes1, boxes2)
        # GIoU should be negative when there's no overlap
        assert giou[0] < 0

    def test_compute_ciou_perfect_overlap(self):
        """Test CIoU computation for perfect overlap."""
        boxes1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        boxes2 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)

        ciou = compute_ciou(boxes1, boxes2)
        assert torch.allclose(ciou, torch.tensor([1.0]), atol=1e-5)

    def test_compute_ciou_no_overlap(self):
        """Test CIoU computation for no overlap."""
        boxes1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        boxes2 = torch.tensor([[20, 20, 30, 30]], dtype=torch.float32)

        ciou = compute_ciou(boxes1, boxes2)
        # CIoU should be negative when there's no overlap
        assert ciou[0] < 0

    def test_batch_iou_computation(self):
        """Test IoU computation for batch of boxes."""
        batch_size = 16
        boxes1 = torch.rand(batch_size, 4) * 50
        boxes1[:, 2:] = boxes1[:, 2:] + boxes1[:, :2] + 5  # Ensure x2>x1, y2>y1

        boxes2 = torch.rand(batch_size, 4) * 50
        boxes2[:, 2:] = boxes2[:, 2:] + boxes2[:, :2] + 5

        iou = compute_iou(boxes1, boxes2)
        assert iou.shape == (batch_size,)
        assert torch.all(iou >= 0) and torch.all(iou <= 1)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_giou_loss_initialization(self):
        """Test GIoU loss initialization."""
        loss_fn = GIoULoss(reduction='mean')
        assert isinstance(loss_fn, nn.Module)
        assert loss_fn.reduction == 'mean'

    def test_giou_loss_perfect_match(self):
        """Test GIoU loss for perfect match."""
        loss_fn = GIoULoss(reduction='mean')

        pred = torch.tensor([[50, 50, 20, 30]], dtype=torch.float32)
        target = torch.tensor([[50, 50, 20, 30]], dtype=torch.float32)

        loss = loss_fn(pred, target)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_giou_loss_no_overlap(self):
        """Test GIoU loss for no overlap."""
        loss_fn = GIoULoss(reduction='mean')

        pred = torch.tensor([[10, 10, 10, 10]], dtype=torch.float32)
        target = torch.tensor([[50, 50, 10, 10]], dtype=torch.float32)

        loss = loss_fn(pred, target)
        assert loss > 0  # Loss should be positive

    def test_ciou_loss_initialization(self):
        """Test CIoU loss initialization."""
        loss_fn = CIoULoss(reduction='mean')
        assert isinstance(loss_fn, nn.Module)
        assert loss_fn.reduction == 'mean'

    def test_ciou_loss_perfect_match(self):
        """Test CIoU loss for perfect match."""
        loss_fn = CIoULoss(reduction='mean')

        pred = torch.tensor([[50, 50, 20, 30]], dtype=torch.float32)
        target = torch.tensor([[50, 50, 20, 30]], dtype=torch.float32)

        loss = loss_fn(pred, target)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_quality_focal_loss_initialization(self):
        """Test Quality Focal Loss initialization."""
        loss_fn = QualityFocalLoss(beta=2.0)
        assert isinstance(loss_fn, nn.Module)
        assert loss_fn.beta == 2.0

    def test_quality_focal_loss_computation(self):
        """Test Quality Focal Loss computation."""
        loss_fn = QualityFocalLoss(beta=2.0)

        pred_scores = torch.tensor([0.8, 0.6, 0.9], dtype=torch.float32)
        target_labels = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        target_quality = torch.tensor([0.9, 0.7, 0.85], dtype=torch.float32)

        loss = loss_fn(pred_scores, target_labels, target_quality)
        assert loss.numel() == 1
        assert loss >= 0


class TestBoxHead:
    """Tests for BoxHead module."""

    @pytest.fixture
    def box_head(self):
        """Create BoxHead for testing."""
        return BoxHead(
            in_channels=768,
            hidden_dim=256,
            num_layers=3,
            box_format='cxcywh',
        )

    def test_box_head_initialization(self, box_head):
        """Test BoxHead initialization."""
        assert isinstance(box_head, BoxHead)
        assert box_head.in_channels == 768
        assert box_head.hidden_dim == 256
        assert box_head.num_layers == 3
        assert box_head.box_format == 'cxcywh'

    def test_box_head_forward_1d_features(self, box_head):
        """Test forward pass with 1D features."""
        batch_size = 4
        features = torch.randn(batch_size, 768)

        outputs = box_head(features)

        assert 'pred_boxes' in outputs
        assert 'pred_scores' in outputs
        assert outputs['pred_boxes'].shape == (batch_size, 4)
        assert outputs['pred_scores'].shape == (batch_size,)

    def test_box_head_forward_3d_features(self, box_head):
        """Test forward pass with 3D features (sequence)."""
        batch_size = 4
        seq_len = 10
        features = torch.randn(batch_size, seq_len, 768)

        outputs = box_head(features)

        assert 'pred_boxes' in outputs
        assert 'pred_scores' in outputs
        assert outputs['pred_boxes'].shape == (batch_size, 4)
        assert outputs['pred_scores'].shape == (batch_size,)

    def test_box_head_forward_4d_features(self, box_head):
        """Test forward pass with 4D features (spatial)."""
        batch_size = 4
        features = torch.randn(batch_size, 768, 14, 14)

        outputs = box_head(features)

        assert 'pred_boxes' in outputs
        assert 'pred_scores' in outputs
        assert outputs['pred_boxes'].shape == (batch_size, 4)
        assert outputs['pred_scores'].shape == (batch_size,)

    def test_box_head_forward_with_targets(self, box_head):
        """Test forward pass with target boxes."""
        batch_size = 4
        features = torch.randn(batch_size, 768)
        target_boxes = torch.rand(batch_size, 4) * 100
        target_boxes[:, 2:] = target_boxes[:, 2:] + 10  # Positive width/height
        target_labels = torch.ones(batch_size)

        outputs = box_head(features, target_boxes, target_labels)

        assert 'pred_boxes' in outputs
        assert 'pred_scores' in outputs
        assert 'cls_loss' in outputs
        assert 'l1_loss' in outputs
        assert 'iou_loss' in outputs
        assert 'total_loss' in outputs

        # Check losses are scalars
        assert outputs['cls_loss'].numel() == 1
        assert outputs['l1_loss'].numel() == 1
        assert outputs['iou_loss'].numel() == 1
        assert outputs['total_loss'].numel() == 1

        # Check losses are non-negative
        assert outputs['cls_loss'] >= 0
        assert outputs['l1_loss'] >= 0
        assert outputs['iou_loss'] >= 0
        assert outputs['total_loss'] >= 0

    def test_box_head_xyxy_format(self):
        """Test BoxHead with xyxy output format."""
        head = BoxHead(
            in_channels=768,
            hidden_dim=256,
            box_format='xyxy',
        )

        features = torch.randn(4, 768)
        outputs = head(features)

        assert outputs['pred_boxes'].shape == (4, 4)
        # Verify it's in xyxy format (x2 > x1, y2 > y1 for valid boxes)
        # Note: Can't guarantee this without targets, just check shape

    def test_box_head_ciou_loss(self):
        """Test BoxHead with CIoU loss."""
        head = BoxHead(
            in_channels=768,
            hidden_dim=256,
            iou_loss_type='ciou',
        )

        assert head.iou_loss_type == 'ciou'
        assert isinstance(head.iou_loss_fn, CIoULoss)

    def test_box_head_giou_loss(self):
        """Test BoxHead with GIoU loss."""
        head = BoxHead(
            in_channels=768,
            hidden_dim=256,
            iou_loss_type='giou',
        )

        assert head.iou_loss_type == 'giou'
        assert isinstance(head.iou_loss_fn, GIoULoss)

    def test_box_head_quality_focal_loss(self):
        """Test BoxHead with Quality Focal Loss."""
        head = BoxHead(
            in_channels=768,
            hidden_dim=256,
            use_quality_focal=True,
        )

        assert head.use_quality_focal
        assert isinstance(head.cls_loss_fn, QualityFocalLoss)

    def test_box_head_bce_loss(self):
        """Test BoxHead with BCE loss."""
        head = BoxHead(
            in_channels=768,
            hidden_dim=256,
            use_quality_focal=False,
        )

        assert not head.use_quality_focal
        # CRITICAL FIX: Uses BCEWithLogitsLoss for AMP safety
        assert isinstance(head.cls_loss_fn, nn.BCEWithLogitsLoss)

    def test_box_head_loss_weights(self):
        """Test BoxHead with custom loss weights."""
        head = BoxHead(
            in_channels=768,
            hidden_dim=256,
            loss_weight_cls=2.0,
            loss_weight_l1=10.0,
            loss_weight_iou=5.0,
        )

        assert head.loss_weight_cls == 2.0
        assert head.loss_weight_l1 == 10.0
        assert head.loss_weight_iou == 5.0

    def test_box_head_default_loss_weights_match_ostrack(self):
        """Test BoxHead default loss weights match OSTrack/MixFormer standard.

        CRITICAL FIX (Audit 2025112601): Verify L1=5.0, GIoU=2.0.
        """
        head = BoxHead(in_channels=768, hidden_dim=256)

        assert head.loss_weight_l1 == 5.0, f"L1 weight should be 5.0, got {head.loss_weight_l1}"
        assert head.loss_weight_iou == 2.0, f"GIoU weight should be 2.0, got {head.loss_weight_iou}"

    def test_box_head_gradient_flow(self, box_head):
        """Test that gradients flow properly through BoxHead."""
        features = torch.randn(4, 768, requires_grad=True)
        target_boxes = torch.rand(4, 4) * 100
        target_boxes[:, 2:] = target_boxes[:, 2:] + 10
        target_labels = torch.ones(4)

        outputs = box_head(features, target_boxes, target_labels)
        loss = outputs['total_loss']

        loss.backward()

        assert features.grad is not None
        assert not torch.all(features.grad == 0)


class TestSpecializedBoxHeads:
    """Tests for specialized box head variants.

    ARCHITECTURE UPDATE (Audit 2025112313): FusionBoxHead removed.
    """

    def test_rgb_box_head(self):
        """Test RGBBoxHead initialization and forward."""
        head = RGBBoxHead(in_channels=768, hidden_dim=256)

        assert isinstance(head, RGBBoxHead)
        assert isinstance(head, BoxHead)

        features = torch.randn(4, 768)
        outputs = head(features)

        assert 'pred_boxes' in outputs
        assert 'pred_scores' in outputs

    def test_sonar_box_head(self):
        """Test SonarBoxHead initialization and forward."""
        head = SonarBoxHead(in_channels=384, hidden_dim=256)

        assert isinstance(head, SonarBoxHead)
        assert isinstance(head, BoxHead)

        features = torch.randn(4, 384)
        outputs = head(features)

        assert 'pred_boxes' in outputs
        assert 'pred_scores' in outputs


class TestBoxHeadFactory:
    """Tests for build_box_head factory function.

    ARCHITECTURE UPDATE (Audit 2025112313): FusionBoxHead removed.
    """

    def test_build_rgb_box_head(self):
        """Test building RGB box head."""
        head = build_box_head('rgb', in_channels=768, hidden_dim=256)
        assert isinstance(head, RGBBoxHead)

    def test_build_sonar_box_head(self):
        """Test building Sonar box head."""
        head = build_box_head('sonar', in_channels=384, hidden_dim=256)
        assert isinstance(head, SonarBoxHead)

    def test_build_base_box_head(self):
        """Test building base box head."""
        head = build_box_head('base', in_channels=768, hidden_dim=256)
        assert isinstance(head, BoxHead)
        assert not isinstance(head, (RGBBoxHead, SonarBoxHead))


class TestBoxHeadIntegration:
    """Integration tests for BoxHead with other modules.

    ARCHITECTURE UPDATE (Audit 2025112313): FusionBoxHead removed.
    """

    def test_box_head_with_vit_features(self):
        """Test BoxHead with ViT-like features."""
        head = BoxHead(in_channels=768, hidden_dim=256)

        # Simulate ViT output: [B, N, C] where N includes cls token
        batch_size = 4
        num_patches = 196  # 14x14
        features = torch.randn(batch_size, num_patches + 1, 768)

        outputs = head(features)

        assert outputs['pred_boxes'].shape == (batch_size, 4)
        assert outputs['pred_scores'].shape == (batch_size,)

    def test_box_head_with_conv_features(self):
        """Test BoxHead with ConvNet features."""
        head = BoxHead(in_channels=512, hidden_dim=256)

        # Simulate ConvNet output: [B, C, H, W]
        batch_size = 4
        features = torch.randn(batch_size, 512, 7, 7)

        outputs = head(features)

        assert outputs['pred_boxes'].shape == (batch_size, 4)
        assert outputs['pred_scores'].shape == (batch_size,)

    def test_multi_head_setup(self):
        """Test using multiple box heads together (RGB + Sonar)."""
        rgb_head = build_box_head('rgb', in_channels=768)
        sonar_head = build_box_head('sonar', in_channels=384)

        # Simulate features from different modalities
        rgb_feat = torch.randn(4, 768)
        sonar_feat = torch.randn(4, 384)

        rgb_out = rgb_head(rgb_feat)
        sonar_out = sonar_head(sonar_feat)

        assert all('pred_boxes' in out for out in [rgb_out, sonar_out])
        assert all('pred_scores' in out for out in [rgb_out, sonar_out])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
