"""
Test: Verify model learns to track target position, not just predict center.

This test verifies that the model can:
1. Predict bbox that follows GT position (not stuck at center)
2. Respond to different target positions in search region
3. Learn meaningful feature matching instead of position bias

Key insight: If model always predicts center [0.5, 0.5, w, h], it means:
- Model has NOT learned feature matching
- Model exploits the bias that targets are often centered during training
- This is a CRITICAL training failure

IMPORTANT: ViT-based trackers require extensive training (OSTrack uses 300 epochs).
Short tests (10-100 batches) may show CENTER BIAS, but this is expected for
untrained/undertrained models. The key metrics to watch during full training:
- Prediction std should approach GT std
- Correlation between pred and GT should increase
- L1 error should decrease

Author: AquaSOT-3D Team
Date: 2025-12-01
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class TestBBoxLearningCapability:
    """Test suite for verifying bbox prediction learning capability."""

    @pytest.fixture
    def model_and_config(self):
        """Load model and config for testing."""
        from aquasot3d.utils.config import load_config, _config_to_dict
        from aquasot3d.models.tracker import AquaSOT3DTracker

        config = load_config('configs/train.yaml')

        # Prepare configs
        rgb_backbone_cfg = _config_to_dict(config.model.backbone.rgb)
        sonar_backbone_cfg = _config_to_dict(config.model.backbone.sonar)
        fusion_cfg = _config_to_dict(config.model.fusion)

        def clean_head_config(head_cfg):
            cfg_dict = _config_to_dict(head_cfg)
            cfg_dict.pop('type', None)
            cfg_dict.pop('in_channels', None)
            return cfg_dict

        box_head_cfg = clean_head_config(config.model.heads.get('rgb_box', {}))

        # Build model
        model = AquaSOT3DTracker(
            rgb_backbone_config=rgb_backbone_cfg,
            sonar_backbone_config=sonar_backbone_cfg,
            fusion_config=fusion_cfg,
            box_head_config=box_head_cfg,
        )

        return model, config

    def create_synthetic_pair(
        self,
        target_cx: float,
        target_cy: float,
        target_w: float = 0.3,
        target_h: float = 0.3,
        template_size: int = 128,
        search_size: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create synthetic template-search pair with target at specified position.

        Args:
            target_cx, target_cy: Target center in normalized [0, 1] coordinates
            target_w, target_h: Target size in normalized coordinates
            template_size: Template image size
            search_size: Search image size

        Returns:
            rgb_template, rgb_search, sonar_template, sonar_search, gt_bbox
        """
        # Create template with target at center (always)
        rgb_template = torch.zeros(1, 3, template_size, template_size)
        sonar_template = torch.zeros(1, 1, template_size, template_size)

        # Draw target in template center
        t_cx, t_cy = template_size // 2, template_size // 2
        t_w, t_h = int(template_size * 0.6), int(template_size * 0.6)
        t_x1 = max(0, t_cx - t_w // 2)
        t_y1 = max(0, t_cy - t_h // 2)
        t_x2 = min(template_size, t_cx + t_w // 2)
        t_y2 = min(template_size, t_cy + t_h // 2)

        # Fill with distinct pattern (simulating target appearance)
        rgb_template[:, :, t_y1:t_y2, t_x1:t_x2] = 0.8
        sonar_template[:, :, t_y1:t_y2, t_x1:t_x2] = 0.8

        # Create search with target at specified position
        rgb_search = torch.zeros(1, 3, search_size, search_size)
        sonar_search = torch.zeros(1, 1, search_size, search_size)

        # Convert normalized coords to pixel coords
        s_cx = int(target_cx * search_size)
        s_cy = int(target_cy * search_size)
        s_w = int(target_w * search_size)
        s_h = int(target_h * search_size)

        s_x1 = max(0, s_cx - s_w // 2)
        s_y1 = max(0, s_cy - s_h // 2)
        s_x2 = min(search_size, s_cx + s_w // 2)
        s_y2 = min(search_size, s_cy + s_h // 2)

        # Fill with same pattern as template
        rgb_search[:, :, s_y1:s_y2, s_x1:s_x2] = 0.8
        sonar_search[:, :, s_y1:s_y2, s_x1:s_x2] = 0.8

        # Add some background noise
        rgb_template += torch.randn_like(rgb_template) * 0.1
        rgb_search += torch.randn_like(rgb_search) * 0.1
        sonar_template += torch.randn_like(sonar_template) * 0.1
        sonar_search += torch.randn_like(sonar_search) * 0.1

        # Clamp to valid range
        rgb_template = rgb_template.clamp(0, 1)
        rgb_search = rgb_search.clamp(0, 1)
        sonar_template = sonar_template.clamp(0, 1)
        sonar_search = sonar_search.clamp(0, 1)

        # GT bbox in normalized cxcywh format
        gt_bbox = torch.tensor([[target_cx, target_cy, target_w, target_h]])

        return rgb_template, rgb_search, sonar_template, sonar_search, gt_bbox

    def test_untrained_model_prediction_variance(self, model_and_config):
        """
        Test that untrained model predictions vary with target position.

        An untrained but properly initialized model should:
        - NOT always predict exact center (that would indicate bad init)
        - Show SOME variance in predictions for different inputs
        """
        model, config = model_and_config
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Test positions: corners and center
        test_positions = [
            (0.3, 0.3),  # top-left
            (0.7, 0.3),  # top-right
            (0.3, 0.7),  # bottom-left
            (0.7, 0.7),  # bottom-right
            (0.5, 0.5),  # center
        ]

        predictions = []

        with torch.no_grad():
            for cx, cy in test_positions:
                rgb_t, rgb_s, sonar_t, sonar_s, gt = self.create_synthetic_pair(cx, cy)

                rgb_t = rgb_t.to(device)
                rgb_s = rgb_s.to(device)
                sonar_t = sonar_t.to(device)
                sonar_s = sonar_s.to(device)

                outputs = model(rgb_t, rgb_s, sonar_t, sonar_s, return_losses=False)

                rgb_pred = outputs['rgb_boxes'][0].cpu().numpy()
                predictions.append(rgb_pred)

        predictions = np.array(predictions)

        # Check prediction variance
        pred_variance = np.var(predictions, axis=0)
        print(f"\nPrediction variance across positions: {pred_variance}")
        print(f"Predictions:\n{predictions}")

        # Untrained model should have SOME variance (not always same output)
        # But variance might be small due to initialization
        total_variance = pred_variance.sum()
        print(f"Total variance: {total_variance:.6f}")

        # Check if predictions are NOT all identical
        pred_range = np.max(predictions, axis=0) - np.min(predictions, axis=0)
        print(f"Prediction range: {pred_range}")

        # At minimum, predictions should not be exactly identical
        assert not np.allclose(predictions[0], predictions[1], atol=1e-6), \
            "Model outputs identical predictions for different inputs - bad initialization!"

    def test_model_responds_to_position_after_training_step(self, model_and_config):
        """
        Test that model learns to respond to target position after a few training steps.

        This simulates a mini-training to verify:
        1. Gradients flow correctly
        2. Model can learn to predict different positions
        3. Loss decreases when predicting correct position
        """
        model, config = model_and_config
        model.train()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Simple optimizer for testing
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create training pairs with targets at different positions
        train_positions = [
            (0.3, 0.3),
            (0.7, 0.7),
            (0.5, 0.5),
            (0.3, 0.7),
            (0.7, 0.3),
        ]

        initial_losses = []
        final_losses = []

        # Train for a few steps on each position
        num_steps = 10

        for cx, cy in train_positions:
            rgb_t, rgb_s, sonar_t, sonar_s, gt = self.create_synthetic_pair(cx, cy)

            rgb_t = rgb_t.to(device)
            rgb_s = rgb_s.to(device)
            sonar_t = sonar_t.to(device)
            sonar_s = sonar_s.to(device)
            gt = gt.to(device)

            for step in range(num_steps):
                optimizer.zero_grad()

                outputs = model(rgb_t, rgb_s, sonar_t, sonar_s, return_losses=False)
                rgb_pred = outputs['rgb_boxes']

                # Simple L1 loss
                loss = torch.nn.functional.l1_loss(rgb_pred, gt)

                if step == 0:
                    initial_losses.append(loss.item())

                loss.backward()
                optimizer.step()

                if step == num_steps - 1:
                    final_losses.append(loss.item())

        print(f"\nInitial losses: {initial_losses}")
        print(f"Final losses: {final_losses}")
        print(f"Average improvement: {np.mean(initial_losses) - np.mean(final_losses):.4f}")

        # Verify loss decreased
        avg_initial = np.mean(initial_losses)
        avg_final = np.mean(final_losses)

        assert avg_final < avg_initial, \
            f"Loss did not decrease! Initial: {avg_initial:.4f}, Final: {avg_final:.4f}"

        # Verify significant improvement (at least 10% reduction)
        improvement_ratio = (avg_initial - avg_final) / avg_initial
        print(f"Improvement ratio: {improvement_ratio:.2%}")

        assert improvement_ratio > 0.1, \
            f"Insufficient learning: only {improvement_ratio:.2%} improvement"

    def test_prediction_follows_gt_not_center_bias(self, model_and_config):
        """
        Critical test: Verify model predictions follow GT, not center bias.

        After brief training, predictions should be closer to GT than to center.
        If predictions are always near center regardless of GT, model has center bias.
        """
        model, config = model_and_config
        model.train()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train on off-center targets
        train_positions = [(0.3, 0.3), (0.7, 0.7), (0.25, 0.75), (0.75, 0.25)]

        # Brief training
        for _ in range(20):
            for cx, cy in train_positions:
                rgb_t, rgb_s, sonar_t, sonar_s, gt = self.create_synthetic_pair(cx, cy)

                rgb_t = rgb_t.to(device)
                rgb_s = rgb_s.to(device)
                sonar_t = sonar_t.to(device)
                sonar_s = sonar_s.to(device)
                gt = gt.to(device)

                optimizer.zero_grad()
                outputs = model(rgb_t, rgb_s, sonar_t, sonar_s, return_losses=False)
                loss = torch.nn.functional.l1_loss(outputs['rgb_boxes'], gt)
                loss.backward()
                optimizer.step()

        # Now test: predictions should be closer to GT than to center
        model.eval()
        center = np.array([0.5, 0.5])

        results = []

        with torch.no_grad():
            for cx, cy in train_positions:
                rgb_t, rgb_s, sonar_t, sonar_s, gt = self.create_synthetic_pair(cx, cy)

                rgb_t = rgb_t.to(device)
                rgb_s = rgb_s.to(device)
                sonar_t = sonar_t.to(device)
                sonar_s = sonar_s.to(device)

                outputs = model(rgb_t, rgb_s, sonar_t, sonar_s, return_losses=False)
                pred = outputs['rgb_boxes'][0, :2].cpu().numpy()  # cx, cy only

                gt_pos = np.array([cx, cy])
                dist_to_gt = np.linalg.norm(pred - gt_pos)
                dist_to_center = np.linalg.norm(pred - center)

                results.append({
                    'gt': gt_pos,
                    'pred': pred,
                    'dist_to_gt': dist_to_gt,
                    'dist_to_center': dist_to_center,
                    'follows_gt': dist_to_gt < dist_to_center
                })

                print(f"GT: {gt_pos}, Pred: {pred}, "
                      f"Dist to GT: {dist_to_gt:.3f}, Dist to center: {dist_to_center:.3f}")

        # Count how many predictions follow GT vs center
        follows_gt_count = sum(1 for r in results if r['follows_gt'])
        total = len(results)

        print(f"\nPredictions following GT: {follows_gt_count}/{total}")

        # At least 50% should follow GT after training
        assert follows_gt_count >= total // 2, \
            f"Model has center bias! Only {follows_gt_count}/{total} predictions follow GT"

    def test_gradient_flow_to_box_head(self, model_and_config):
        """
        Verify gradients flow properly to box head during training.

        This checks that:
        1. Box head parameters receive gradients
        2. Gradients are non-zero
        3. Parameters actually update
        """
        model, config = model_and_config
        model.train()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Get initial box head parameters
        initial_params = {}
        for name, param in model.named_parameters():
            if 'box_head' in name.lower() or 'rgb_head' in name.lower():
                initial_params[name] = param.clone().detach()

        assert len(initial_params) > 0, "No box head parameters found!"

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Single training step
        rgb_t, rgb_s, sonar_t, sonar_s, gt = self.create_synthetic_pair(0.3, 0.3)

        rgb_t = rgb_t.to(device)
        rgb_s = rgb_s.to(device)
        sonar_t = sonar_t.to(device)
        sonar_s = sonar_s.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        outputs = model(rgb_t, rgb_s, sonar_t, sonar_s, return_losses=False)
        loss = torch.nn.functional.l1_loss(outputs['rgb_boxes'], gt)
        loss.backward()

        # Check gradients exist and are non-zero
        grad_info = []
        for name, param in model.named_parameters():
            if name in initial_params:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_info.append((name, grad_norm))
                else:
                    grad_info.append((name, 0.0))

        print("\nBox head gradient norms:")
        for name, norm in grad_info:
            print(f"  {name}: {norm:.6f}")

        # At least some gradients should be non-zero
        non_zero_grads = sum(1 for _, norm in grad_info if norm > 1e-8)
        assert non_zero_grads > 0, "No gradients flowing to box head!"

        # Apply update
        optimizer.step()

        # Check parameters changed
        changed_params = 0
        for name, param in model.named_parameters():
            if name in initial_params:
                if not torch.allclose(param, initial_params[name], atol=1e-8):
                    changed_params += 1

        print(f"\nParameters updated: {changed_params}/{len(initial_params)}")
        assert changed_params > 0, "Box head parameters did not update!"


class TestCenterBiasDetection:
    """Tests specifically for detecting center bias in predictions."""

    @pytest.fixture
    def trained_model_path(self):
        """Path to a trained checkpoint (if available)."""
        import os
        checkpoint_paths = [
            'outputs/train/checkpoints/checkpoint_last.pth',
            'outputs/train/checkpoints/checkpoint_best.pth',
        ]
        for path in checkpoint_paths:
            if os.path.exists(path):
                return path
        return None

    def test_detect_center_bias_in_checkpoint(self, trained_model_path):
        """
        Test a trained checkpoint for center bias.

        Skip if no checkpoint available.
        """
        if trained_model_path is None:
            pytest.skip("No trained checkpoint available")

        from aquasot3d.utils.config import load_config, _config_to_dict
        from aquasot3d.models.tracker import AquaSOT3DTracker

        config = load_config('configs/train.yaml')

        # Build model
        rgb_backbone_cfg = _config_to_dict(config.model.backbone.rgb)
        sonar_backbone_cfg = _config_to_dict(config.model.backbone.sonar)
        fusion_cfg = _config_to_dict(config.model.fusion)

        def clean_head_config(head_cfg):
            cfg_dict = _config_to_dict(head_cfg)
            cfg_dict.pop('type', None)
            cfg_dict.pop('in_channels', None)
            return cfg_dict

        box_head_cfg = clean_head_config(config.model.heads.get('rgb_box', {}))

        model = AquaSOT3DTracker(
            rgb_backbone_config=rgb_backbone_cfg,
            sonar_backbone_config=sonar_backbone_cfg,
            fusion_config=fusion_cfg,
            box_head_config=box_head_cfg,
        )

        # Load checkpoint
        checkpoint = torch.load(trained_model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Test multiple positions
        test_positions = [
            (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
            (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7),
        ]

        center = np.array([0.5, 0.5])
        predictions = []

        with torch.no_grad():
            for cx, cy in test_positions:
                # Create synthetic data
                rgb_t = torch.randn(1, 3, 128, 128).to(device)
                rgb_s = torch.randn(1, 3, 256, 256).to(device)
                sonar_t = torch.randn(1, 1, 128, 128).to(device)
                sonar_s = torch.randn(1, 1, 256, 256).to(device)

                outputs = model(rgb_t, rgb_s, sonar_t, sonar_s, return_losses=False)
                pred = outputs['rgb_boxes'][0, :2].cpu().numpy()
                predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        dist_to_center = np.linalg.norm(mean_pred - center)

        print(f"\nPrediction statistics:")
        print(f"  Mean prediction: {mean_pred}")
        print(f"  Std prediction: {std_pred}")
        print(f"  Distance of mean to center: {dist_to_center:.4f}")

        # Check for center bias
        # If std is very low and mean is near center, model has center bias
        has_center_bias = (std_pred[0] < 0.05 and std_pred[1] < 0.05 and
                          dist_to_center < 0.1)

        if has_center_bias:
            print("\n[WARNING] Model appears to have CENTER BIAS!")
            print("Predictions cluster around center regardless of input.")

        assert not has_center_bias, \
            "Model has center bias - always predicting near [0.5, 0.5]"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])