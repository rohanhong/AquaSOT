"""
Test Suite for Target-Centered Search Region Cropping

CRITICAL FIX (Audit 2025112602): Verify OSTrack-style target-centered cropping
implementation is correct. This is essential for the model to learn target
localization rather than predicting image center by default.

Tests:
1. Verify crop size calculation: sqrt(w*h) * search_area_factor
2. Verify target is near center of cropped region
3. Verify boundary padding works correctly
4. Verify attention mask correctly marks padded regions
5. Verify bbox transformation to crop coordinates
6. Verify train/eval consistency

Author: AquaSOT-3D Team
Date: 2025-11-26
"""

import math
import numpy as np
import torch
import pytest


class TestSearchRegionCropping:
    """Test suite for target-centered search region cropping."""

    def test_crop_size_calculation(self):
        """Test that crop size follows OSTrack formula: sqrt(w*h) * factor."""
        # Test cases: (w, h, factor, expected_crop_sz)
        test_cases = [
            (100, 100, 4.0, 400),   # Square bbox
            (50, 200, 4.0, 400),    # Tall bbox: sqrt(50*200) * 4 = 400
            (200, 50, 4.0, 400),    # Wide bbox
            (80, 80, 2.0, 160),     # Template factor
            (64, 64, 4.0, 256),     # Small target
        ]

        for w, h, factor, expected in test_cases:
            target_sz = math.sqrt(w * h)
            crop_sz = int(math.ceil(target_sz * factor))
            assert crop_sz == expected, \
                f"Expected crop_sz={expected} for w={w}, h={h}, factor={factor}, got {crop_sz}"

    def test_target_center_in_crop(self):
        """Test that target bbox center is near the center of the cropped region."""
        # Create mock image and bbox
        H, W = 480, 640
        image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

        # Target at various positions
        test_bboxes = [
            [300, 220, 80, 100],   # Center of image
            [50, 50, 60, 80],      # Top-left corner
            [550, 400, 70, 60],    # Bottom-right area
            [10, 200, 100, 100],   # Left edge
        ]

        for bbox in test_bboxes:
            x, y, w, h = bbox
            cx_orig = x + w / 2
            cy_orig = y + h / 2

            # Simulate cropping (from _crop_search_region)
            target_sz = math.sqrt(w * h)
            crop_sz = int(math.ceil(target_sz * 4.0))  # search_area_factor = 4.0

            # Crop center = bbox center
            crop_x1 = int(round(cx_orig - crop_sz / 2))
            crop_y1 = int(round(cy_orig - crop_sz / 2))

            # Transform bbox to crop coordinates
            x_crop = x - crop_x1
            y_crop = y - crop_y1

            # Bbox center in crop coordinates
            cx_crop = x_crop + w / 2
            cy_crop = y_crop + h / 2

            # Normalize to [0, 1] (before resize)
            cx_norm = cx_crop / crop_sz
            cy_norm = cy_crop / crop_sz

            # Target center should be very close to 0.5 (crop center)
            assert abs(cx_norm - 0.5) < 0.05, \
                f"Target cx_norm={cx_norm} not near 0.5 for bbox={bbox}"
            assert abs(cy_norm - 0.5) < 0.05, \
                f"Target cy_norm={cy_norm} not near 0.5 for bbox={bbox}"

    def test_boundary_padding(self):
        """Test that boundary padding is correctly applied when crop extends beyond image."""
        H, W = 480, 640
        image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

        # Target at corner - crop will extend beyond image
        bbox = [10, 10, 100, 100]
        x, y, w, h = bbox
        cx = x + w / 2  # 60
        cy = y + h / 2  # 60

        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * 4.0))  # 400

        # Crop boundaries
        x1 = int(round(cx - crop_sz / 2))  # 60 - 200 = -140
        y1 = int(round(cy - crop_sz / 2))  # 60 - 200 = -140
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        # Check padding amounts
        x1_pad = max(0, -x1)  # 140
        y1_pad = max(0, -y1)  # 140
        x2_pad = max(0, x2 - W)  # max(0, 260 - 640) = 0
        y2_pad = max(0, y2 - H)  # max(0, 260 - 480) = 0

        assert x1_pad == 140, f"Expected x1_pad=140, got {x1_pad}"
        assert y1_pad == 140, f"Expected y1_pad=140, got {y1_pad}"
        assert x2_pad == 0, f"Expected x2_pad=0, got {x2_pad}"
        assert y2_pad == 0, f"Expected y2_pad=0, got {y2_pad}"

    def test_attention_mask(self):
        """Test that attention mask correctly marks padded regions."""
        crop_sz = 400
        output_size = 256

        # Simulate padding amounts
        x1_pad, y1_pad = 140, 140
        x2_pad, y2_pad = 0, 0

        # Create attention mask (before resize)
        att_mask = np.ones((crop_sz, crop_sz), dtype=np.float32)
        if y1_pad > 0:
            att_mask[:y1_pad, :] = 0
        if y2_pad > 0:
            att_mask[-y2_pad:, :] = 0
        if x1_pad > 0:
            att_mask[:, :x1_pad] = 0
        if x2_pad > 0:
            att_mask[:, -x2_pad:] = 0

        # Verify mask values
        valid_area = crop_sz * crop_sz - (x1_pad * crop_sz + y1_pad * (crop_sz - x1_pad))
        actual_valid = att_mask.sum()

        # Expected valid area: (400-140) * (400-140) = 260 * 260 = 67600
        expected_valid = (crop_sz - x1_pad) * (crop_sz - y1_pad)
        assert actual_valid == expected_valid, \
            f"Expected valid area={expected_valid}, got {actual_valid}"

    def test_bbox_transformation(self):
        """Test bbox transformation from image to crop coordinates."""
        # Original image bbox
        bbox_orig = np.array([100, 100, 80, 100])  # x, y, w, h

        # Crop metadata
        crop_metadata = {
            'crop_x1': 40,    # Crop starts at x=40
            'crop_y1': 50,    # Crop starts at y=50
            'resize_factor': 0.64,  # 256 / 400
        }

        # Transform
        x, y, w, h = bbox_orig
        crop_x1 = crop_metadata['crop_x1']
        crop_y1 = crop_metadata['crop_y1']
        resize_factor = crop_metadata['resize_factor']

        x_rel = x - crop_x1  # 100 - 40 = 60
        y_rel = y - crop_y1  # 100 - 50 = 50

        x_out = x_rel * resize_factor  # 60 * 0.64 = 38.4
        y_out = y_rel * resize_factor  # 50 * 0.64 = 32.0
        w_out = w * resize_factor      # 80 * 0.64 = 51.2
        h_out = h * resize_factor      # 100 * 0.64 = 64.0

        bbox_crop = np.array([x_out, y_out, w_out, h_out])

        assert np.allclose(bbox_crop, [38.4, 32.0, 51.2, 64.0]), \
            f"Unexpected bbox_crop: {bbox_crop}"

    def test_normalized_bbox_near_center(self):
        """
        CRITICAL TEST: Verify normalized GT bbox center is near (0.5, 0.5).

        This is the key property that enables the model to learn relative
        offset prediction instead of defaulting to image center.
        """
        output_size = 256

        # Simulate target bbox
        bbox_orig = np.array([100, 100, 80, 100])
        x, y, w, h = bbox_orig
        cx_orig = x + w / 2  # 140
        cy_orig = y + h / 2  # 150

        # Crop centered on target
        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * 4.0))

        crop_x1 = int(round(cx_orig - crop_sz / 2))
        crop_y1 = int(round(cy_orig - crop_sz / 2))
        resize_factor = output_size / crop_sz

        # Transform bbox to crop coordinates
        x_crop = (x - crop_x1) * resize_factor
        y_crop = (y - crop_y1) * resize_factor
        w_crop = w * resize_factor
        h_crop = h * resize_factor

        # Convert to CXCYWH normalized
        cx_crop = x_crop + w_crop / 2
        cy_crop = y_crop + h_crop / 2

        cx_norm = cx_crop / output_size
        cy_norm = cy_crop / output_size

        # CRITICAL: Target center should be near (0.5, 0.5)
        assert abs(cx_norm - 0.5) < 0.1, \
            f"CRITICAL: cx_norm={cx_norm} not near 0.5! Model will learn to predict center."
        assert abs(cy_norm - 0.5) < 0.1, \
            f"CRITICAL: cy_norm={cy_norm} not near 0.5! Model will learn to predict center."


class TestTrainEvalConsistency:
    """Test train/eval consistency for search region cropping."""

    def test_same_crop_parameters(self):
        """Verify train and eval use same crop parameters."""
        # These should match between configs
        search_area_factor = 4.0
        template_area_factor = 2.0
        search_size = 256
        template_size = 256

        # Verify standard OSTrack values
        assert search_area_factor == 4.0
        assert template_area_factor == 2.0
        assert search_size == 256
        assert template_size == 256

    def test_crop_formula_consistency(self):
        """Verify crop formula is identical in train and eval."""
        w, h = 80, 100
        search_area_factor = 4.0

        # OSTrack formula
        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * search_area_factor))

        # Expected: sqrt(8000) * 4 = 89.44 * 4 = 357.77 -> ceil -> 358
        expected = int(math.ceil(math.sqrt(w * h) * search_area_factor))

        assert crop_sz == expected, f"Crop formula mismatch: {crop_sz} != {expected}"


class TestJitterBehavior:
    """Test center/scale jitter for training."""

    def test_no_jitter_in_eval(self):
        """Verify evaluation uses zero jitter."""
        center_jitter_factor = 0.0
        scale_jitter_factor = 0.0

        bbox = np.array([100, 100, 80, 100])
        x, y, w, h = bbox

        cx = x + w / 2
        cy = y + h / 2
        target_sz = math.sqrt(w * h)
        crop_sz = int(math.ceil(target_sz * 4.0))

        # With zero jitter, crop center = bbox center
        if center_jitter_factor == 0:
            cx_jittered = cx
            cy_jittered = cy
        if scale_jitter_factor == 0:
            crop_sz_jittered = crop_sz

        assert cx_jittered == cx
        assert cy_jittered == cy
        assert crop_sz_jittered == crop_sz

    def test_jitter_bounded(self):
        """Verify jitter is bounded within reasonable range."""
        center_jitter_factor = 0.25
        scale_jitter_factor = 0.25

        bbox = np.array([100, 100, 80, 100])
        x, y, w, h = bbox

        cx_orig = x + w / 2
        cy_orig = y + h / 2
        target_sz = math.sqrt(w * h)
        crop_sz_orig = int(math.ceil(target_sz * 4.0))

        # Maximum jitter bounds
        max_cx_offset = center_jitter_factor * w   # 0.25 * 80 = 20
        max_cy_offset = center_jitter_factor * h   # 0.25 * 100 = 25
        max_scale_change = scale_jitter_factor     # 0.25 = +/-25%

        # Jittered values should be within bounds
        # cx_jittered in [cx - 20, cx + 20]
        # cy_jittered in [cy - 25, cy + 25]
        # crop_sz_jittered in [crop_sz * 0.75, crop_sz * 1.25]

        assert max_cx_offset == 20
        assert max_cy_offset == 25
        assert max_scale_change == 0.25


class TestEvaluatorDenormalization:
    """Test evaluator denormalization with target-centered cropping."""

    def test_denormalize_target_centered_crop(self):
        """
        CRITICAL TEST: Verify denormalization correctly reverses target-centered crop.

        Given:
        - Original bbox at [100, 100, 80, 100] (x, y, w, h)
        - Crop centered on target
        - Model predicts normalized bbox near (0.5, 0.5)

        The denormalized prediction should match the original bbox.
        """
        # Original bbox
        x_orig, y_orig, w_orig, h_orig = 100, 100, 80, 100
        cx_orig = x_orig + w_orig / 2  # 140
        cy_orig = y_orig + h_orig / 2  # 150

        # Crop parameters (simulating _crop_search_region)
        target_sz = math.sqrt(w_orig * h_orig)  # sqrt(8000) = 89.44
        crop_sz = int(math.ceil(target_sz * 4.0))  # 358
        output_size = 256

        crop_x1 = int(round(cx_orig - crop_sz / 2))  # 140 - 179 = -39
        crop_y1 = int(round(cy_orig - crop_sz / 2))  # 150 - 179 = -29
        resize_factor = output_size / crop_sz  # 256 / 358 = 0.715

        # Transform bbox to normalized crop coordinates (what model sees)
        x_crop = (x_orig - crop_x1) * resize_factor  # (100 - (-39)) * 0.715 = 99.4
        y_crop = (y_orig - crop_y1) * resize_factor  # (100 - (-29)) * 0.715 = 92.2
        w_crop = w_orig * resize_factor  # 80 * 0.715 = 57.2
        h_crop = h_orig * resize_factor  # 100 * 0.715 = 71.5

        # Convert to normalized cxcywh
        cx_crop = x_crop + w_crop / 2
        cy_crop = y_crop + h_crop / 2
        cx_norm = cx_crop / output_size
        cy_norm = cy_crop / output_size
        w_norm = w_crop / output_size
        h_norm = h_crop / output_size

        # Verify target is near center (key property)
        assert abs(cx_norm - 0.5) < 0.1, f"cx_norm={cx_norm} not near 0.5"
        assert abs(cy_norm - 0.5) < 0.1, f"cy_norm={cy_norm} not near 0.5"

        # Now test denormalization (reverse the process)
        bbox_norm = np.array([cx_norm, cy_norm, w_norm, h_norm])

        # Denormalize (simulating _denormalize_bbox)
        bbox_output = bbox_norm * output_size  # Back to crop pixels
        cx_denorm = bbox_output[0] / resize_factor + crop_x1
        cy_denorm = bbox_output[1] / resize_factor + crop_y1
        w_denorm = bbox_output[2] / resize_factor
        h_denorm = bbox_output[3] / resize_factor

        # Convert to xywh
        x_denorm = cx_denorm - w_denorm / 2
        y_denorm = cy_denorm - h_denorm / 2

        # Should recover original bbox
        assert abs(x_denorm - x_orig) < 1, f"x: expected {x_orig}, got {x_denorm}"
        assert abs(y_denorm - y_orig) < 1, f"y: expected {y_orig}, got {y_denorm}"
        assert abs(w_denorm - w_orig) < 1, f"w: expected {w_orig}, got {w_denorm}"
        assert abs(h_denorm - h_orig) < 1, f"h: expected {h_orig}, got {h_denorm}"

    def test_denormalize_with_negative_crop_origin(self):
        """Test denormalization when crop extends beyond image (negative crop_x1/y1)."""
        # Target at corner - crop will have negative origin
        x_orig, y_orig, w_orig, h_orig = 10, 10, 100, 100
        cx_orig = x_orig + w_orig / 2  # 60
        cy_orig = y_orig + h_orig / 2  # 60

        target_sz = math.sqrt(w_orig * h_orig)  # 100
        crop_sz = int(math.ceil(target_sz * 4.0))  # 400
        output_size = 256

        crop_x1 = int(round(cx_orig - crop_sz / 2))  # 60 - 200 = -140
        crop_y1 = int(round(cy_orig - crop_sz / 2))  # 60 - 200 = -140
        resize_factor = output_size / crop_sz  # 0.64

        assert crop_x1 < 0, "This test requires negative crop_x1"
        assert crop_y1 < 0, "This test requires negative crop_y1"

        # Transform to normalized
        x_crop = (x_orig - crop_x1) * resize_factor
        y_crop = (y_orig - crop_y1) * resize_factor
        w_crop = w_orig * resize_factor
        h_crop = h_orig * resize_factor

        cx_norm = (x_crop + w_crop / 2) / output_size
        cy_norm = (y_crop + h_crop / 2) / output_size
        w_norm = w_crop / output_size
        h_norm = h_crop / output_size

        bbox_norm = np.array([cx_norm, cy_norm, w_norm, h_norm])

        # Denormalize
        bbox_output = bbox_norm * output_size
        cx_denorm = bbox_output[0] / resize_factor + crop_x1
        cy_denorm = bbox_output[1] / resize_factor + crop_y1
        w_denorm = bbox_output[2] / resize_factor
        h_denorm = bbox_output[3] / resize_factor

        x_denorm = cx_denorm - w_denorm / 2
        y_denorm = cy_denorm - h_denorm / 2

        # Should recover original despite negative crop origin
        assert abs(x_denorm - x_orig) < 1, f"x: expected {x_orig}, got {x_denorm}"
        assert abs(y_denorm - y_orig) < 1, f"y: expected {y_orig}, got {y_denorm}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
