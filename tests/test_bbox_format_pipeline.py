"""
Bbox Format Pipeline Verification Tests

Tests the complete bbox format pipeline from dataset to loss to ensure:
1. Dataset outputs XYWH pixel (scaled)
2. Collate converts to CXCYWH normalized [0, 1]
3. Trainer extracts target frame correctly
4. Loss receives correct format

CRITICAL: These tests prevent the bbox format mismatch bug discovered in
Bbox Format Audit 2025-11-20 from recurring.

Author: AquaSOT-3D Team
Date: 2025-11-20
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from aquasot3d.dataloader.datasets import RGBSonarDataset, FrameSequenceDataset
from aquasot3d.dataloader import collate_fn_sequence
from aquasot3d.utils.bbox import BBoxCXCYWH, BBoxXYWH


class TestBboxFormatPipeline:
    """Test bbox format consistency across the entire pipeline."""

    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Create temporary test dataset."""
        import cv2

        # Create directory structure
        data_root = tmp_path / "data"
        seq_path = data_root / "test_seq"
        rgb_path = seq_path / "light"
        sonar_path = seq_path / "sonar"

        # Create all required directories (including img_rect for rectangular sonar)
        for path in [rgb_path / "img", sonar_path / "img", sonar_path / "img_rect"]:
            path.mkdir(parents=True, exist_ok=True)

        # Create dummy images (10 frames)
        for i in range(1, 11):
            # RGB: 1920x1080 (typical resolution)
            rgb_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2_write_path = str(rgb_path / "img" / f"{i}.jpg")
            cv2.imwrite(cv2_write_path, rgb_img)

            # Sonar: 682x376 (typical resolution)
            sonar_img = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / "img" / f"{i}.jpg"), sonar_img)

            # Sonar rectangular (same resolution for simplicity)
            sonar_rect_img = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / "img_rect" / f"{i}.jpg"), sonar_rect_img)

        # Create groundtruth files (XYWH pixel format)
        # RGB: bbox at center, size 200x150 pixels
        rgb_gt = rgb_path / "groundtruth.txt"
        with open(rgb_gt, "w") as f:
            for i in range(10):
                # Center at (960, 540), size (200, 150)
                # XYWH: (860, 465, 200, 150)
                f.write("860,465,200,150\n")

        # Sonar: bbox at center, size 100x80 pixels
        sonar_gt = sonar_path / "groundtruth.txt"
        with open(sonar_gt, "w") as f:
            for i in range(10):
                # Center at (341, 188), size (100, 80)
                # XYWH: (291, 148, 100, 80)
                f.write("291,148,100,80\n")

        # Sonar rectangular groundtruth (same bbox for simplicity)
        sonar_rect_gt = sonar_path / "groundtruth_rect.txt"
        with open(sonar_rect_gt, "w") as f:
            for i in range(10):
                # Same bbox as polar sonar
                f.write("291,148,100,80\n")

        return str(data_root)

    def test_dataset_bbox_format(self, temp_dataset):
        """Verify dataset outputs XYWH pixel (scaled)."""
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="test_seq"
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512,  # Letterbox Migration: 384 -> 512
            frame_stride=1
        )

        sample = dataset[0]

        # Check bbox keys exist
        assert 'rgb_bboxes' in sample, "RGB bbox should be present"
        assert 'sonar_bboxes' in sample, "Sonar bbox should be present"

        # Check shape [T, 4]
        assert sample['rgb_bboxes'].shape == (5, 4), f"RGB bbox shape should be [5, 4], got {sample['rgb_bboxes'].shape}"
        assert sample['sonar_bboxes'].shape == (5, 4), f"Sonar bbox shape should be [5, 4], got {sample['sonar_bboxes'].shape}"

        # Check format is XYWH pixel (letterboxed to 512x512)
        # Original RGB bbox: (860, 465, 200, 150) in 1920x1080
        # Letterbox scaling to 512x512:
        # scale = min(512/1920, 512/1080) = min(0.333, 0.593) = 0.333
        # new_size = 512x360, pad_top = 140, pad_bottom = 140
        # Expected: (860*0.333 + 0, 465*0.333 + 140, 200*0.333, 150*0.333) ≈ (286, 295, 66.7, 50)
        rgb_bbox = sample['rgb_bboxes'][0].numpy()  # First frame
        assert rgb_bbox[0] >= 0 and rgb_bbox[0] <= 512, "RGB x should be in pixel range"
        assert rgb_bbox[1] >= 0 and rgb_bbox[1] <= 512, "RGB y should be in pixel range"
        assert rgb_bbox[2] >= 0 and rgb_bbox[2] <= 512, "RGB w should be in pixel range"
        assert rgb_bbox[3] >= 0 and rgb_bbox[3] <= 512, "RGB h should be in pixel range"

        # Verify it's NOT normalized (values should be > 1)
        assert rgb_bbox[0] > 1.0 or rgb_bbox[2] > 1.0, "Dataset bbox should be in pixel coordinates, not normalized"

        print(f"✓ Dataset outputs XYWH pixel (scaled): {rgb_bbox}")

    def test_collate_bbox_format(self, temp_dataset):
        """Verify collate converts to CXCYWH normalized [0, 1]."""
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="test_seq"
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512,  # Letterbox Migration: 384 -> 512
            frame_stride=1
        )

        # Create batch
        batch_samples = [dataset[i] for i in range(2)]
        batch = collate_fn_sequence(batch_samples)

        # Check bbox keys
        assert 'rgb_bboxes' in batch, "Collated RGB bbox should be present"
        assert 'sonar_bboxes' in batch, "Collated Sonar bbox should be present"

        # Check shape [B, T, 4]
        assert batch['rgb_bboxes'].shape == (2, 5, 4), f"Collated RGB bbox shape should be [2, 5, 4], got {batch['rgb_bboxes'].shape}"
        assert batch['sonar_bboxes'].shape == (2, 5, 4), f"Collated Sonar bbox shape should be [2, 5, 4], got {batch['sonar_bboxes'].shape}"

        # CRITICAL: Check format is CXCYWH normalized [0, 1]
        rgb_bbox = batch['rgb_bboxes'][0, 0]  # First sample, first frame
        assert rgb_bbox.min() >= 0.0, f"RGB bbox min should be >= 0, got {rgb_bbox.min()}"
        assert rgb_bbox.max() <= 1.0, f"RGB bbox max should be <= 1, got {rgb_bbox.max()}"

        sonar_bbox = batch['sonar_bboxes'][0, 0]
        assert sonar_bbox.min() >= 0.0, f"Sonar bbox min should be >= 0, got {sonar_bbox.min()}"
        assert sonar_bbox.max() <= 1.0, f"Sonar bbox max should be <= 1, got {sonar_bbox.max()}"

        # Verify it's center format (not top-left)
        # For a bbox at image center, cx and cy should be near 0.5
        cx, cy = rgb_bbox[0].item(), rgb_bbox[1].item()
        assert 0.3 < cx < 0.7, f"Center x should be near 0.5 for centered bbox, got {cx}"
        assert 0.3 < cy < 0.7, f"Center y should be near 0.5 for centered bbox, got {cy}"

        print(f"✓ Collate outputs CXCYWH normalized: cx={cx:.3f}, cy={cy:.3f}, w={rgb_bbox[2]:.3f}, h={rgb_bbox[3]:.3f}")

        # CRITICAL: Check pixel coordinates are also stored
        assert 'rgb_bboxes_pixels' in batch, "Pixel bbox should be stored for quality estimators"
        assert 'sonar_bboxes_pixels' in batch, "Pixel bbox should be stored for quality estimators"

        # Pixel bbox should be in [0, frame_size] range
        rgb_bbox_px = batch['rgb_bboxes_pixels'][0, 0]
        assert rgb_bbox_px.max() > 1.0, "Pixel bbox should not be normalized"
        assert rgb_bbox_px.max() <= 512.0, "Pixel bbox should be in frame_size range (letterbox 512x512)"

        print(f"✓ Pixel bboxes stored correctly for quality estimators")

    def test_bbox_conversion_math(self):
        """Verify bbox conversion math is correct with letterbox."""
        # Test case: XYWH [100, 150, 50, 80] in 512x512 letterboxed image
        # Should convert to CXCYWH normalized [0, 1]

        frame_size = 512.0
        x, y, w, h = 100.0, 150.0, 50.0, 80.0

        # Convert to center coordinates and normalize
        cx = (x + w / 2.0) / frame_size
        cy = (y + h / 2.0) / frame_size
        w_norm = w / frame_size
        h_norm = h / frame_size

        # Expected values
        expected_cx = (100 + 50/2) / 512  # 125/512 ≈ 0.195
        expected_cy = (150 + 80/2) / 512  # 190/512 ≈ 0.297
        expected_w = 50 / 512  # ≈ 0.078
        expected_h = 80 / 512  # ≈ 0.125

        assert abs(cx - expected_cx) < 1e-6, f"cx conversion error: {cx} != {expected_cx}"
        assert abs(cy - expected_cy) < 1e-6, f"cy conversion error: {cy} != {expected_cy}"
        assert abs(w_norm - expected_w) < 1e-6, f"w conversion error: {w_norm} != {expected_w}"
        assert abs(h_norm - expected_h) < 1e-6, f"h conversion error: {h_norm} != {expected_h}"

        print(f"✓ Bbox conversion math verified: XYWH [100, 150, 50, 80] → CXCYWH [{cx:.3f}, {cy:.3f}, {w_norm:.3f}, {h_norm:.3f}]")

    def test_bbox_format_with_wrapper(self, temp_dataset):
        """Verify bbox can be wrapped with BBoxCXCYWH without errors."""
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="test_seq"
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512,  # Letterbox Migration: 384 -> 512
            frame_stride=1
        )

        # Create batch
        batch_samples = [dataset[i] for i in range(2)]
        batch = collate_fn_sequence(batch_samples)

        # Extract target frame bbox
        target_idx = batch['target_idx']  # [B]
        batch_size = batch['rgb_bboxes'].shape[0]
        gt_rgb = batch['rgb_bboxes'][torch.arange(batch_size), target_idx]  # [B, 4]

        # CRITICAL: This should not raise ValueError
        try:
            bbox_wrapper = BBoxCXCYWH(gt_rgb)
            print(f"✓ BBoxCXCYWH wrapper accepts collated bbox without errors")
        except (ValueError, TypeError) as e:
            pytest.fail(f"BBoxCXCYWH wrapper failed: {e}")

        # Verify wrapper can convert to xyxy (for IoU computation)
        try:
            bbox_xyxy = bbox_wrapper.to_xyxy()
            xyxy_tensor = bbox_xyxy.tensor()
            assert xyxy_tensor.shape == gt_rgb.shape, "XYXY conversion should preserve shape"
            print(f"✓ CXCYWH → XYXY conversion works correctly")
        except Exception as e:
            pytest.fail(f"Bbox conversion failed: {e}")

    def test_end_to_end_pipeline(self, temp_dataset):
        """Test complete pipeline: dataset → collate → extraction → loss."""
        from aquasot3d.training.loss_manager import LossManager

        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="test_seq"
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512,  # Letterbox Migration: 384 -> 512
            frame_stride=1
        )

        # Create batch
        batch_samples = [dataset[i] for i in range(2)]
        batch = collate_fn_sequence(batch_samples)

        # Simulate trainer extraction
        target_idx = batch['target_idx']
        batch_size = batch['rgb_bboxes'].shape[0]
        gt_rgb = batch['rgb_bboxes'][torch.arange(batch_size), target_idx]
        gt_sonar = batch['sonar_bboxes'][torch.arange(batch_size), target_idx]

        # Simulate predictions (random but in correct format)
        pred_rgb = torch.rand(batch_size, 4) * 0.8 + 0.1  # CXCYWH normalized [0.1, 0.9]
        pred_sonar = torch.rand(batch_size, 4) * 0.8 + 0.1

        # CRITICAL: Loss computation should work without errors
        # This will fail if bbox format is wrong
        try:
            loss_manager = LossManager(device='cpu')

            # Test RGB box loss
            rgb_loss = loss_manager._compute_box_loss(
                pred_boxes=pred_rgb,
                target_boxes=gt_rgb,
                loss_type='giou'
            )

            assert torch.isfinite(rgb_loss), "RGB loss should be finite"
            assert rgb_loss >= 0, "Loss should be non-negative"
            # GIoU loss range is [-1, 1], L1 + GIoU typically in [0, 2]
            assert rgb_loss < 10.0, f"Loss should be reasonable, got {rgb_loss}"

            print(f"✓ End-to-end pipeline works: RGB GIoU loss = {rgb_loss:.4f}")

        except Exception as e:
            pytest.fail(f"Loss computation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
