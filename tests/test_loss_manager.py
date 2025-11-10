"""
Unit Tests for Loss Manager

Tests for multi-task loss computation, weighting, and scheduling.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.training.loss_manager import LossManager, build_loss_manager


class TestLossManager:
    """Test suite for LossManager"""

    def test_initialization(self):
        """Test LossManager initialization"""
        manager = LossManager()
        assert manager is not None
        assert 'rgb_cls' in manager.loss_weights
        assert 'fusion_box' in manager.loss_weights

    def test_custom_weights(self):
        """Test custom loss weights"""
        custom_weights = {
            'rgb_cls': 2.0,
            'rgb_box': 1.5,
            'fusion_cls': 3.0
        }
        manager = LossManager(loss_weights=custom_weights)
        assert manager.loss_weights['rgb_cls'] == 2.0
        assert manager.loss_weights['rgb_box'] == 1.5

    def test_compute_losses_basic(self):
        """Test basic loss computation"""
        manager = LossManager()

        # Create dummy predictions and targets
        batch_size = 4
        predictions = {
            'rgb_box': torch.randn(batch_size, 4),
            'sonar_box': torch.randn(batch_size, 4),
            'fusion_box': torch.randn(batch_size, 4)
        }
        targets = {
            'rgb_box': torch.randn(batch_size, 4),
            'sonar_box': torch.randn(batch_size, 4),
            'fusion_box': torch.randn(batch_size, 4)
        }

        total_loss, loss_dict = manager.compute_losses(predictions, targets)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() > 0
        assert 'total' in loss_dict
        assert 'rgb_box' in loss_dict

    def test_classification_loss(self):
        """Test classification loss computation"""
        manager = LossManager()

        predictions = {
            'rgb_cls': torch.randn(4, 1),
            'fusion_cls': torch.randn(4, 1)
        }
        targets = {
            'rgb_cls': torch.randint(0, 2, (4, 1)),
            'fusion_cls': torch.randint(0, 2, (4, 1))
        }

        total_loss, loss_dict = manager.compute_losses(predictions, targets)

        assert 'rgb_cls' in loss_dict
        assert 'fusion_cls' in loss_dict
        assert total_loss.item() > 0

    def test_consistency_losses(self):
        """Test consistency loss handling"""
        manager = LossManager()

        predictions = {
            'consistency': {
                'component': torch.tensor(0.5),
                'representation': torch.tensor(0.3)
            }
        }
        targets = {}

        total_loss, loss_dict = manager.compute_losses(predictions, targets)

        assert 'component_consistency' in loss_dict
        assert 'representation_consistency' in loss_dict
        assert loss_dict['component_consistency'] == 0.5

    def test_weak3d_cycle_loss(self):
        """Test weak 3D cycle loss handling"""
        manager = LossManager()

        predictions = {
            'weak3d': {
                'cyclic_loss': torch.tensor(0.8)
            }
        }
        targets = {}

        total_loss, loss_dict = manager.compute_losses(predictions, targets)

        assert 'weak3d_cycle' in loss_dict
        assert loss_dict['weak3d_cycle'] == 0.8

    def test_temporal_smoothness_loss(self):
        """Test temporal smoothness loss"""
        manager = LossManager()

        predictions = {
            'temporal': {
                'smoothness_loss': torch.tensor(0.1)
            }
        }
        targets = {}

        total_loss, loss_dict = manager.compute_losses(predictions, targets)

        assert 'temporal_smooth' in loss_dict
        assert loss_dict['temporal_smooth'] == 0.1

    def test_linear_schedule(self):
        """Test linear weight scheduling"""
        target_weights = {'fusion_box': 1.5, 'weak3d_cycle': 1.0}
        manager = LossManager(
            weight_schedule='linear',
            schedule_config={'max_epoch': 60, 'target_weights': target_weights}
        )

        # Test at different epochs
        weights_epoch_0 = manager._get_current_weights()
        manager.current_epoch = 30
        weights_epoch_30 = manager._get_current_weights()
        manager.current_epoch = 60
        weights_epoch_60 = manager._get_current_weights()

        assert weights_epoch_30['fusion_box'] != weights_epoch_0['fusion_box']
        assert abs(weights_epoch_60['fusion_box'] - 1.5) < 0.01

    def test_step_schedule(self):
        """Test step-wise weight scheduling"""
        weight_changes = {
            20: {'fusion_box': 1.2},
            40: {'fusion_box': 1.5}
        }
        manager = LossManager(
            weight_schedule='step',
            schedule_config={'milestones': [20, 40], 'weight_changes': weight_changes}
        )

        manager.current_epoch = 10
        weights_epoch_10 = manager._get_current_weights()

        manager.current_epoch = 25
        weights_epoch_25 = manager._get_current_weights()

        manager.current_epoch = 45
        weights_epoch_45 = manager._get_current_weights()

        assert weights_epoch_10['fusion_box'] == 1.0  # Initial
        assert weights_epoch_25['fusion_box'] == 1.2  # After milestone 20
        assert weights_epoch_45['fusion_box'] == 1.5  # After milestone 40

    def test_cosine_schedule(self):
        """Test cosine weight scheduling"""
        target_weights = {'fusion_box': 1.5}
        manager = LossManager(
            weight_schedule='cosine',
            schedule_config={'max_epoch': 60, 'target_weights': target_weights}
        )

        manager.current_epoch = 0
        weights_start = manager._get_current_weights()

        manager.current_epoch = 30
        weights_mid = manager._get_current_weights()

        manager.current_epoch = 60
        weights_end = manager._get_current_weights()

        # Cosine should vary smoothly
        assert weights_start['fusion_box'] != weights_mid['fusion_box']
        assert weights_mid['fusion_box'] != weights_end['fusion_box']

    def test_loss_history(self):
        """Test loss history tracking"""
        manager = LossManager()

        predictions = {'rgb_box': torch.randn(4, 4)}
        targets = {'rgb_box': torch.randn(4, 4)}

        # Compute losses for multiple epochs
        for epoch in range(5):
            manager.compute_losses(predictions, targets, epoch=epoch)

        history = manager.get_loss_history()
        assert len(history) == 5
        assert all('epoch' in entry for entry in history)
        assert all('total' in entry for entry in history)

    def test_reset_history(self):
        """Test loss history reset"""
        manager = LossManager()

        predictions = {'rgb_box': torch.randn(4, 4)}
        targets = {'rgb_box': torch.randn(4, 4)}

        manager.compute_losses(predictions, targets)
        assert len(manager.get_loss_history()) > 0

        manager.reset_history()
        assert len(manager.get_loss_history()) == 0

    def test_state_dict(self):
        """Test state dict save/load"""
        manager = LossManager()
        manager.current_epoch = 10

        state = manager.state_dict()
        assert state['current_epoch'] == 10
        assert 'loss_weights' in state

        new_manager = LossManager()
        new_manager.load_state_dict(state)
        assert new_manager.current_epoch == 10

    def test_build_loss_manager(self):
        """Test factory function"""
        config = {
            'loss_weights': {'rgb_cls': 2.0},
            'weight_schedule': 'linear'
        }
        manager = build_loss_manager(config)

        assert manager is not None
        assert manager.loss_weights['rgb_cls'] == 2.0
        assert manager.weight_schedule == 'linear'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
