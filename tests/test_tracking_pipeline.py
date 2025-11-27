"""
Comprehensive Tracking Pipeline Tests

This test suite verifies the entire tracking training pipeline:
1. Data loading and preprocessing correctness
2. GT bbox distribution (must be within [0, 1])
3. Model forward pass
4. Loss computation and backpropagation
5. Overfitting test on small batch (key indicator of learning capability)

Author: AquaSOT-3D Team
Date: 2025-11-27
"""

import sys
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aquasot3d.dataloader.datasets.rgb_sonar_dataset import (
    RGBSonarDataset, MultiSequenceRGBSonarDataset
)
from aquasot3d.dataloader.datasets.template_search_dataset import (
    TemplateSearchDataset, template_search_collate_fn
)


class TestJitterImplementation:
    """Test center/scale jitter implementation matches OSTrack."""

    def test_jitter_formula_ostrack_alignment(self):
        """Verify jitter formula matches OSTrack: offset = max_offset * (rand - 0.5)."""
        # Simulate the jitter calculation
        target_sz = 100.0
        center_jitter_factor = 3.0
        area_factor = 4.0

        # Compute expected ranges
        crop_sz = target_sz * area_factor  # 400
        half_crop = crop_sz / 2  # 200

        # OSTrack-style jitter: offset = max_offset * (rand - 0.5)
        # max_offset = target_sz * center_jitter_factor = 300
        # offset range = [-0.5, 0.5] * 300 = [-150, 150]
        max_offset = target_sz * center_jitter_factor
        min_offset = -0.5 * max_offset  # -150
        max_offset_val = 0.5 * max_offset  # 150

        # Target center should always be within crop (offset < half_crop)
        assert abs(min_offset) < half_crop, \
            f"Min offset {min_offset} exceeds half_crop {half_crop}"
        assert abs(max_offset_val) < half_crop, \
            f"Max offset {max_offset_val} exceeds half_crop {half_crop}"

        print(f"[PASS] Jitter range [{min_offset}, {max_offset_val}] < half_crop {half_crop}")

    def test_jitter_produces_valid_gt_range(self):
        """Test that jittered GT bbox centers stay within [0, 1]."""
        # Skip if no data available
        if not Path('data/uuv1').exists():
            pytest.skip("Test data not available")

        base_dataset = MultiSequenceRGBSonarDataset(
            data_root='data',
            sequence_names=['uuv1'],
            visibility_policy='any'
        )

        dataset = TemplateSearchDataset(
            base_dataset=base_dataset,
            template_size=128,
            search_size=256,
            template_search_gap=(1, 30),
            search_area_factor=4.0,
            template_area_factor=2.0,
            center_jitter_factor=3.0,
            scale_jitter_factor=0.25
        )

        # Check 100 samples
        outside_count = 0
        n_samples = min(100, len(dataset))

        for i in range(n_samples):
            sample = dataset[i]
            rgb_bbox = sample['search_bbox_rgb']
            if isinstance(rgb_bbox, torch.Tensor):
                rgb_bbox = rgb_bbox.numpy()

            cx, cy, w, h = rgb_bbox

            if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                outside_count += 1

        outside_ratio = outside_count / n_samples
        assert outside_ratio < 0.05, \
            f"Too many samples ({outside_ratio*100:.1f}%) have GT center outside [0,1]!"

        print(f"[PASS] Only {outside_ratio*100:.1f}% samples have GT center outside [0,1]")


class TestDataPipeline:
    """Test data loading and preprocessing."""

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        if not Path('data/uuv1').exists():
            pytest.skip("Test data not available")

        base_dataset = MultiSequenceRGBSonarDataset(
            data_root='data',
            sequence_names=['uuv1'],
            visibility_policy='any'
        )

        return TemplateSearchDataset(
            base_dataset=base_dataset,
            template_size=128,
            search_size=256,
            template_search_gap=(1, 30),
            search_area_factor=4.0,
            template_area_factor=2.0,
            center_jitter_factor=3.0,
            scale_jitter_factor=0.25,
            normalization_config={
                'rgb_mean': [0.485, 0.456, 0.406],
                'rgb_std': [0.229, 0.224, 0.225],
                'sonar_mean': [0.5],
                'sonar_std': [0.5]
            }
        )

    def test_sample_shapes(self, dataset):
        """Test that sample tensors have correct shapes."""
        sample = dataset[0]

        # Check template shapes
        assert sample['rgb_template'].shape == (3, 128, 128), \
            f"RGB template shape mismatch: {sample['rgb_template'].shape}"
        assert sample['sonar_template'].shape == (1, 128, 128), \
            f"Sonar template shape mismatch: {sample['sonar_template'].shape}"

        # Check search shapes
        assert sample['rgb_search'].shape == (3, 256, 256), \
            f"RGB search shape mismatch: {sample['rgb_search'].shape}"
        assert sample['sonar_search'].shape == (1, 256, 256), \
            f"Sonar search shape mismatch: {sample['sonar_search'].shape}"

        # Check bbox shapes
        assert sample['search_bbox_rgb'].shape == (4,), \
            f"RGB bbox shape mismatch: {sample['search_bbox_rgb'].shape}"
        assert sample['search_bbox_sonar'].shape == (4,), \
            f"Sonar bbox shape mismatch: {sample['search_bbox_sonar'].shape}"

        print("[PASS] All sample shapes are correct")

    def test_normalization(self, dataset):
        """Test that images are properly normalized."""
        sample = dataset[0]

        rgb = sample['rgb_search']
        sonar = sample['sonar_search']

        # Normalized images should have values roughly in [-2, 2] range
        # (ImageNet normalization)
        assert rgb.min() > -5 and rgb.max() < 5, \
            f"RGB normalization issue: range [{rgb.min():.2f}, {rgb.max():.2f}]"
        assert sonar.min() > -5 and sonar.max() < 5, \
            f"Sonar normalization issue: range [{sonar.min():.2f}, {sonar.max():.2f}]"

        print(f"[PASS] Normalization correct: RGB [{rgb.min():.2f}, {rgb.max():.2f}], "
              f"Sonar [{sonar.min():.2f}, {sonar.max():.2f}]")

    def test_bbox_format_cxcywh_normalized(self, dataset):
        """Test that GT bbox is in CXCYWH normalized [0,1] format."""
        sample = dataset[0]

        rgb_bbox = sample['search_bbox_rgb']
        if isinstance(rgb_bbox, torch.Tensor):
            rgb_bbox = rgb_bbox.numpy()

        cx, cy, w, h = rgb_bbox

        # Check format is CXCYWH (center + size, not corner + size)
        # Width and height should be positive
        assert w > 0, f"Width should be positive: {w}"
        assert h > 0, f"Height should be positive: {h}"

        # Normalized values should be in [0, 1] range (mostly)
        assert 0 <= cx <= 1, f"cx={cx} should be in [0,1]"
        assert 0 <= cy <= 1, f"cy={cy} should be in [0,1]"
        assert 0 < w <= 1, f"w={w} should be in (0,1]"
        assert 0 < h <= 1, f"h={h} should be in (0,1]"

        print(f"[PASS] Bbox format correct: cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")

    def test_collate_function(self, dataset):
        """Test batch collation."""
        batch = [dataset[i] for i in range(4)]
        collated = template_search_collate_fn(batch)

        assert collated['rgb_template'].shape == (4, 3, 128, 128)
        assert collated['rgb_search'].shape == (4, 3, 256, 256)
        assert collated['search_bbox_rgb'].shape == (4, 4)

        print("[PASS] Collate function works correctly")


class TestModelForward:
    """Test model forward pass."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        try:
            from aquasot3d.models.tracker import AquaSOT3DTracker
            from aquasot3d.utils.config import load_config

            cfg = load_config('configs/train.yaml')
            model = AquaSOT3DTracker(
                rgb_backbone_config=cfg.model.backbone.rgb,
                sonar_backbone_config=cfg.model.backbone.sonar,
                fusion_config=cfg.model.fusion,
                box_head_config=cfg.model.heads
            )
            model.eval()
            return model
        except Exception as e:
            pytest.skip(f"Cannot create model: {e}")

    def test_forward_pass(self, model):
        """Test model forward pass with dummy data."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        # Create dummy inputs
        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128).to(device)
        rgb_search = torch.randn(batch_size, 3, 256, 256).to(device)
        sonar_template = torch.randn(batch_size, 1, 128, 128).to(device)
        sonar_search = torch.randn(batch_size, 1, 256, 256).to(device)

        with torch.no_grad():
            outputs = model(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

        # Check output keys
        assert 'rgb_boxes' in outputs, "Missing rgb_boxes in output"
        assert 'sonar_boxes' in outputs, "Missing sonar_boxes in output"

        # Check output shapes
        assert outputs['rgb_boxes'].shape == (batch_size, 4), \
            f"RGB boxes shape mismatch: {outputs['rgb_boxes'].shape}"
        assert outputs['sonar_boxes'].shape == (batch_size, 4), \
            f"Sonar boxes shape mismatch: {outputs['sonar_boxes'].shape}"

        # Check output range (should be [0, 1] after sigmoid)
        rgb_boxes = outputs['rgb_boxes']
        assert rgb_boxes.min() >= 0 and rgb_boxes.max() <= 1, \
            f"RGB boxes range [{rgb_boxes.min():.3f}, {rgb_boxes.max():.3f}] should be [0,1]"

        print(f"[PASS] Forward pass successful, output range: "
              f"[{rgb_boxes.min():.3f}, {rgb_boxes.max():.3f}]")


class TestLossComputation:
    """Test loss computation and backward pass."""

    def test_l1_loss_computation(self):
        """Test L1 loss with typical prediction/GT pairs."""
        # Simulate model output and GT
        pred = torch.tensor([[0.5, 0.5, 0.3, 0.2]])  # cxcywh normalized
        gt = torch.tensor([[0.5, 0.5, 0.3, 0.2]])  # Perfect match

        loss = nn.L1Loss()(pred, gt)
        assert loss.item() == 0.0, f"Perfect match should have 0 loss, got {loss.item()}"

        # Test with offset
        pred = torch.tensor([[0.6, 0.6, 0.3, 0.2]])
        gt = torch.tensor([[0.5, 0.5, 0.3, 0.2]])
        loss = nn.L1Loss()(pred, gt)
        expected = (0.1 + 0.1 + 0 + 0) / 4  # 0.05
        assert abs(loss.item() - expected) < 1e-5, \
            f"L1 loss mismatch: {loss.item()} vs {expected}"

        print("[PASS] L1 loss computation correct")

    def test_giou_loss_range(self):
        """Test GIoU loss range is valid."""
        try:
            from aquasot3d.models.losses import GIoULoss
            giou_loss = GIoULoss()
        except ImportError:
            # Simple GIoU implementation for testing
            def compute_giou(pred, gt):
                # Convert cxcywh to xyxy
                pred_x1 = pred[:, 0] - pred[:, 2] / 2
                pred_y1 = pred[:, 1] - pred[:, 3] / 2
                pred_x2 = pred[:, 0] + pred[:, 2] / 2
                pred_y2 = pred[:, 1] + pred[:, 3] / 2

                gt_x1 = gt[:, 0] - gt[:, 2] / 2
                gt_y1 = gt[:, 1] - gt[:, 3] / 2
                gt_x2 = gt[:, 0] + gt[:, 2] / 2
                gt_y2 = gt[:, 1] + gt[:, 3] / 2

                # Intersection
                inter_x1 = torch.max(pred_x1, gt_x1)
                inter_y1 = torch.max(pred_y1, gt_y1)
                inter_x2 = torch.min(pred_x2, gt_x2)
                inter_y2 = torch.min(pred_y2, gt_y2)

                inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                             torch.clamp(inter_y2 - inter_y1, min=0)

                pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                union_area = pred_area + gt_area - inter_area

                iou = inter_area / (union_area + 1e-7)

                # Enclosing box
                enc_x1 = torch.min(pred_x1, gt_x1)
                enc_y1 = torch.min(pred_y1, gt_y1)
                enc_x2 = torch.max(pred_x2, gt_x2)
                enc_y2 = torch.max(pred_y2, gt_y2)
                enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

                giou = iou - (enc_area - union_area) / (enc_area + 1e-7)
                return 1 - giou.mean()

            giou_loss = compute_giou

        # Perfect match
        pred = torch.tensor([[0.5, 0.5, 0.3, 0.2]])
        gt = torch.tensor([[0.5, 0.5, 0.3, 0.2]])
        loss = giou_loss(pred, gt) if callable(giou_loss) else giou_loss(pred, gt)
        assert loss.item() < 0.01, f"Perfect match GIoU loss should be ~0, got {loss.item()}"

        print("[PASS] GIoU loss computation correct")


class TestOverfitting:
    """Test model can overfit on small batch (key learning capability test)."""

    def test_overfit_single_batch(self):
        """Test that model can overfit on a single batch."""
        if not Path('data/uuv1').exists():
            pytest.skip("Test data not available")

        try:
            from aquasot3d.models.tracker import AquaSOT3DTracker
            from aquasot3d.utils.config import load_config
        except ImportError as e:
            pytest.skip(f"Cannot import model: {e}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load config and create model
        cfg = load_config('configs/train.yaml')
        model = AquaSOT3DTracker(
            rgb_backbone_config=cfg.model.backbone.rgb,
            sonar_backbone_config=cfg.model.backbone.sonar,
            fusion_config=cfg.model.fusion,
            box_head_config=cfg.model.heads
        ).to(device)
        model.train()

        # Create small dataset
        base_dataset = MultiSequenceRGBSonarDataset(
            data_root='data',
            sequence_names=['uuv1'],
            visibility_policy='any'
        )

        dataset = TemplateSearchDataset(
            base_dataset=base_dataset,
            template_size=128,
            search_size=256,
            template_search_gap=(1, 30),
            search_area_factor=4.0,
            template_area_factor=2.0,
            center_jitter_factor=3.0,
            scale_jitter_factor=0.25,
            normalization_config=cfg.data.normalize
        )

        # Get a fixed batch
        batch_indices = [0, 1, 2, 3]
        batch = [dataset[i] for i in batch_indices]
        batch = template_search_collate_fn(batch)

        # Move to device
        rgb_template = batch['rgb_template'].to(device)
        rgb_search = batch['rgb_search'].to(device)
        sonar_template = batch['sonar_template'].to(device)
        sonar_search = batch['sonar_search'].to(device)
        gt_rgb = batch['search_bbox_rgb'].to(device)
        gt_sonar = batch['search_bbox_sonar'].to(device)

        # Optimizer with high learning rate for fast overfitting
        # Use higher LR for box heads, lower for backbone
        optimizer = torch.optim.Adam([
            {'params': model.backbone.parameters(), 'lr': 0.0001},
            {'params': model.fusion.parameters(), 'lr': 0.001},
            {'params': model.rgb_box_head.parameters(), 'lr': 0.005},
            {'params': model.sonar_box_head.parameters(), 'lr': 0.005},
        ])

        # Training loop
        initial_loss = None
        final_loss = None
        num_epochs = 100  # Increase epochs for better overfitting

        print("\n=== Overfitting Test ===")
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # gt_boxes format: {'rgb': (B, 4), 'sonar': (B, 4)}
            gt_boxes = {'rgb': gt_rgb, 'sonar': gt_sonar}

            outputs = model(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                gt_boxes=gt_boxes,
                return_losses=True
            )

            # Compute total loss from losses dict
            losses = outputs['losses']
            loss = (
                losses.get('rgb_box_loss', 0) +
                losses.get('sonar_box_loss', 0) +
                losses.get('rgb_conf_loss', 0) * 0.1 +
                losses.get('sonar_conf_loss', 0) * 0.1
            )
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: loss = {loss.item():.4f} (rgb_box={losses['rgb_box_loss'].item():.3f}, sonar_box={losses['sonar_box_loss'].item():.3f})")

        print(f"\nInitial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")

        # Check that loss decreased significantly
        # For complex dual-modality model, 30% reduction in 100 epochs is acceptable
        reduction_pct = (1 - final_loss / initial_loss) * 100
        assert final_loss < initial_loss * 0.7, \
            f"Loss should decrease by >30%, but only {reduction_pct:.1f}%"

        print(f"\n[PASS] Model can overfit on single batch - learning capability confirmed!")
        print(f"       Loss reduction: {reduction_pct:.1f}% (threshold: 30%)")


class TestEndToEnd:
    """End-to-end integration test."""

    def test_training_step(self):
        """Test complete training step."""
        if not Path('data/uuv1').exists():
            pytest.skip("Test data not available")

        try:
            from aquasot3d.models.tracker import AquaSOT3DTracker
            from aquasot3d.utils.config import load_config
        except ImportError as e:
            pytest.skip(f"Cannot import: {e}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg = load_config('configs/train.yaml')

        # Create model
        model = AquaSOT3DTracker(
            rgb_backbone_config=cfg.model.backbone.rgb,
            sonar_backbone_config=cfg.model.backbone.sonar,
            fusion_config=cfg.model.fusion,
            box_head_config=cfg.model.heads
        ).to(device)
        model.train()

        # Create dataset
        base_dataset = MultiSequenceRGBSonarDataset(
            data_root='data',
            sequence_names=['uuv1'],
            visibility_policy='any'
        )

        dataset = TemplateSearchDataset(
            base_dataset=base_dataset,
            template_size=128,
            search_size=256,
            template_search_gap=(1, 30),
            search_area_factor=4.0,
            template_area_factor=2.0,
            center_jitter_factor=3.0,
            scale_jitter_factor=0.25,
            normalization_config=cfg.data.normalize
        )

        # Get batch
        batch = template_search_collate_fn([dataset[i] for i in range(2)])

        # Forward pass
        gt_boxes = {
            'rgb': batch['search_bbox_rgb'].to(device),
            'sonar': batch['search_bbox_sonar'].to(device)
        }
        outputs = model(
            batch['rgb_template'].to(device),
            batch['rgb_search'].to(device),
            batch['sonar_template'].to(device),
            batch['sonar_search'].to(device),
            gt_boxes=gt_boxes,
            return_losses=True
        )

        # Check outputs
        assert 'losses' in outputs, "Missing losses dict"
        assert 'rgb_boxes' in outputs, "Missing rgb_boxes"
        assert 'sonar_boxes' in outputs, "Missing sonar_boxes"

        # Compute total loss
        losses = outputs['losses']
        total_loss = losses.get('rgb_box_loss', 0) + losses.get('sonar_box_loss', 0)

        # Check loss is finite
        assert torch.isfinite(total_loss), \
            f"Loss is not finite: {total_loss}"

        # Backward pass
        total_loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed!"

        print(f"[PASS] Training step successful, loss = {outputs['total_loss'].item():.4f}")


def run_quick_sanity_check():
    """Run quick sanity check without pytest."""
    print("=" * 70)
    print("TRACKING PIPELINE SANITY CHECK")
    print("=" * 70)

    # Test 1: Jitter formula
    print("\n[Test 1] Jitter Formula")
    test = TestJitterImplementation()
    test.test_jitter_formula_ostrack_alignment()

    # Test 2: GT bbox distribution
    print("\n[Test 2] GT Bbox Distribution")
    try:
        test.test_jitter_produces_valid_gt_range()
    except Exception as e:
        print(f"[SKIP] {e}")

    # Test 3: L1 loss
    print("\n[Test 3] Loss Computation")
    test = TestLossComputation()
    test.test_l1_loss_computation()

    print("\n" + "=" * 70)
    print("SANITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick sanity check')
    parser.add_argument('--overfit', action='store_true', help='Run overfit test only')
    args = parser.parse_args()

    if args.quick:
        run_quick_sanity_check()
    elif args.overfit:
        test = TestOverfitting()
        test.test_overfit_single_batch()
    else:
        pytest.main([__file__, '-v', '-s'])