"""
Test training pipeline following OSTrack structure.

This module tests:
1. Training configuration and settings
2. Actor forward pass and loss computation
3. Trainer initialization and single step
4. Optimizer and scheduler creation
5. Full training loop (single iteration smoke test)
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from collections import OrderedDict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ==============================================================================
# Configuration Tests
# ==============================================================================

class TestTrainingConfiguration:
    """Test configuration loading and validation."""

    def test_config_module_import(self):
        """Test that config module can be imported."""
        from lib.config.aquasot.config import cfg
        assert cfg is not None

    def test_config_has_train_section(self):
        """Test that config has TRAIN section."""
        from lib.config.aquasot.config import cfg
        assert hasattr(cfg, 'TRAIN')
        assert hasattr(cfg.TRAIN, 'LR')
        assert hasattr(cfg.TRAIN, 'EPOCH')
        assert hasattr(cfg.TRAIN, 'BATCH_SIZE')

    def test_config_has_model_section(self):
        """Test that config has MODEL section."""
        from lib.config.aquasot.config import cfg
        assert hasattr(cfg, 'MODEL')
        assert hasattr(cfg.MODEL, 'BACKBONE')
        assert hasattr(cfg.MODEL.BACKBONE, 'RGB')
        assert hasattr(cfg.MODEL.BACKBONE, 'SONAR')

    def test_config_has_data_section(self):
        """Test that config has DATA section."""
        from lib.config.aquasot.config import cfg
        assert hasattr(cfg, 'DATA')
        assert hasattr(cfg.DATA, 'RGB')
        assert hasattr(cfg.DATA, 'SONAR')

    def test_update_config_from_file(self):
        """Test config update from YAML file."""
        from lib.config.aquasot.config import cfg, update_config_from_file
        config_file = os.path.join(project_root, 'experiments/aquasot/vitb_256_dual.yaml')
        if os.path.exists(config_file):
            update_config_from_file(config_file)
            # Config should still be valid after update
            assert cfg.TRAIN.LR > 0


# ==============================================================================
# Settings Tests
# ==============================================================================

class TestSettings:
    """Test training settings module."""

    def test_settings_creation(self):
        """Test Settings object creation."""
        from lib.train.admin import Settings
        settings = Settings()
        assert settings is not None

    def test_settings_has_env(self):
        """Test Settings has environment."""
        from lib.train.admin import Settings
        settings = Settings()
        assert hasattr(settings, 'env')

    def test_env_settings(self):
        """Test environment settings."""
        from lib.train.admin import env_settings
        env = env_settings()
        assert hasattr(env, 'workspace_dir')
        assert hasattr(env, 'data_dir')


# ==============================================================================
# Actor Tests
# ==============================================================================

class TestActor:
    """Test training actor."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        from lib.models.aquasot import build_aquasot
        from lib.config.aquasot.config import cfg

        # Disable pretrained weights for testing
        cfg.MODEL.PRETRAIN_FILE = ""
        cfg.MODEL.PRETRAIN_FILE_SONAR = ""
        # Create small model for testing
        cfg.MODEL.BACKBONE.RGB.DEPTH = 2  # Reduce depth for faster testing
        cfg.MODEL.BACKBONE.SONAR.DEPTH = 2
        model = build_aquasot(cfg, training=False)
        return model

    @pytest.fixture
    def mock_data(self):
        """Create mock training data."""
        batch_size = 2
        # Note: rgb_anno and sonar_anno use [x, y, w, h] format (normalized)
        return {
            'rgb_template': torch.randn(1, batch_size, 3, 128, 128),
            'rgb_search': torch.randn(1, batch_size, 3, 256, 256),
            'sonar_template': torch.randn(1, batch_size, 1, 128, 128),
            'sonar_search': torch.randn(1, batch_size, 1, 256, 256),
            'rgb_anno': torch.tensor([[[0.3, 0.3, 0.3, 0.3]] * batch_size]),  # [N_s, B, 4]
            'sonar_anno': torch.tensor([[[0.3, 0.3, 0.3, 0.3]] * batch_size]),
            'epoch': 1,
        }

    def test_actor_import(self):
        """Test that actor can be imported."""
        from lib.train.actors import AquaSOTActor, BaseActor
        assert AquaSOTActor is not None
        assert BaseActor is not None

    def test_actor_creation(self, mock_model):
        """Test actor creation."""
        from lib.train.actors import AquaSOTActor
        from lib.train.admin import Settings
        from lib.config.aquasot.config import cfg
        from lib.utils import giou_loss, FocalLoss
        from torch.nn.functional import l1_loss

        settings = Settings()
        settings.device = 'cpu'

        objective = {
            'giou': giou_loss,
            'l1': l1_loss,
            'focal': FocalLoss(),
        }
        loss_weight = {
            'giou': 2.0,
            'l1': 5.0,
            'focal': 1.0,
        }

        actor = AquaSOTActor(
            net=mock_model,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )
        assert actor is not None

    def test_actor_forward_pass(self, mock_model, mock_data):
        """Test actor forward pass."""
        from lib.train.actors import AquaSOTActor
        from lib.train.admin import Settings
        from lib.config.aquasot.config import cfg
        from lib.utils import giou_loss, FocalLoss
        from torch.nn.functional import l1_loss

        settings = Settings()
        settings.device = 'cpu'

        objective = {
            'giou': giou_loss,
            'l1': l1_loss,
            'focal': FocalLoss(),
        }
        loss_weight = {
            'giou': 2.0,
            'l1': 5.0,
            'focal': 1.0,
        }

        actor = AquaSOTActor(
            net=mock_model,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )
        actor.train(True)

        # Forward pass should return loss and stats
        loss, stats = actor(mock_data)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert isinstance(stats, dict)
        assert 'Loss/total' in stats


# ==============================================================================
# Trainer Tests
# ==============================================================================

class TestTrainer:
    """Test training trainer."""

    def test_trainer_import(self):
        """Test that trainers can be imported."""
        from lib.train.trainers import LTRTrainer, BaseTrainer
        assert LTRTrainer is not None
        assert BaseTrainer is not None

    def test_ltr_loader_import(self):
        """Test that LTRLoader can be imported."""
        from lib.train.data import LTRLoader
        assert LTRLoader is not None


# ==============================================================================
# Base Functions Tests
# ==============================================================================

class TestBaseFunctions:
    """Test base training functions."""

    def test_update_settings(self):
        """Test settings update from config."""
        from lib.train.base_functions import update_settings
        from lib.train.admin import Settings
        from lib.config.aquasot.config import cfg

        settings = Settings()
        update_settings(settings, cfg)

        assert hasattr(settings, 'print_interval')
        assert hasattr(settings, 'batchsize')
        assert hasattr(settings, 'scheduler_type')

    def test_get_optimizer_scheduler(self):
        """Test optimizer and scheduler creation."""
        from lib.train.base_functions import get_optimizer_scheduler
        from lib.config.aquasot.config import cfg

        # Create simple model for testing
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 5)
        )
        # Add backbone attribute for parameter grouping
        model.backbone = nn.Linear(5, 5)

        optimizer, scheduler = get_optimizer_scheduler(model, cfg)

        assert optimizer is not None
        assert scheduler is not None
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_build_dataloaders(self):
        """Test dataloader creation."""
        from lib.train.base_functions import build_dataloaders, update_settings
        from lib.train.admin import Settings
        from lib.config.aquasot.config import cfg

        settings = Settings()
        settings.local_rank = -1
        update_settings(settings, cfg)

        train_loader, val_loader = build_dataloaders(cfg, settings)

        assert train_loader is not None
        assert val_loader is not None
        assert train_loader.name == 'train'
        assert val_loader.name == 'val'


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    @pytest.fixture
    def setup_training_env(self):
        """Set up minimal training environment."""
        from lib.config.aquasot.config import cfg
        from lib.train.admin import Settings

        # Disable pretrained weights for testing
        cfg.MODEL.PRETRAIN_FILE = ""
        cfg.MODEL.PRETRAIN_FILE_SONAR = ""
        # Reduce model size for testing
        cfg.MODEL.BACKBONE.RGB.DEPTH = 2
        cfg.MODEL.BACKBONE.SONAR.DEPTH = 2
        cfg.TRAIN.BATCH_SIZE = 2
        cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 10
        cfg.DATA.VAL.SAMPLE_PER_EPOCH = 5

        settings = Settings()
        settings.local_rank = -1
        settings.device = 'cpu'

        return cfg, settings

    def test_full_training_setup(self, setup_training_env):
        """Test full training setup without running."""
        cfg, settings = setup_training_env

        from lib.train.base_functions import update_settings, build_dataloaders, get_optimizer_scheduler
        from lib.train.actors import AquaSOTActor
        from lib.train.trainers import LTRTrainer
        from lib.models.aquasot import build_aquasot
        from lib.utils import giou_loss, FocalLoss
        from torch.nn.functional import l1_loss

        # Update settings
        update_settings(settings, cfg)

        # Build dataloaders
        train_loader, val_loader = build_dataloaders(cfg, settings)
        assert len(train_loader) > 0

        # Build model
        model = build_aquasot(cfg, training=False)
        assert model is not None

        # Create actor
        objective = {
            'giou': giou_loss,
            'l1': l1_loss,
            'focal': FocalLoss(),
        }
        loss_weight = {
            'giou': cfg.TRAIN.GIOU_WEIGHT,
            'l1': cfg.TRAIN.L1_WEIGHT,
            'focal': cfg.TRAIN.FOCAL_WEIGHT,
        }
        actor = AquaSOTActor(
            net=model,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )
        assert actor is not None

        # Create optimizer
        optimizer, scheduler = get_optimizer_scheduler(model, cfg)
        assert optimizer is not None

    def test_single_training_step(self, setup_training_env):
        """Test single training step execution."""
        cfg, settings = setup_training_env

        from lib.train.base_functions import update_settings, get_optimizer_scheduler
        from lib.train.actors import AquaSOTActor
        from lib.models.aquasot import build_aquasot
        from lib.utils import giou_loss, FocalLoss
        from torch.nn.functional import l1_loss

        update_settings(settings, cfg)

        # Build small model
        model = build_aquasot(cfg, training=False)

        # Create actor
        objective = {
            'giou': giou_loss,
            'l1': l1_loss,
            'focal': FocalLoss(),
        }
        loss_weight = {
            'giou': 2.0,
            'l1': 5.0,
            'focal': 1.0,
        }
        actor = AquaSOTActor(
            net=model,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )

        # Create optimizer
        optimizer, _ = get_optimizer_scheduler(model, cfg)

        # Create dummy batch
        batch_size = 2
        # Note: rgb_anno and sonar_anno use [x, y, w, h] format (normalized)
        data = {
            'rgb_template': torch.randn(1, batch_size, 3, 128, 128),
            'rgb_search': torch.randn(1, batch_size, 3, 256, 256),
            'sonar_template': torch.randn(1, batch_size, 1, 128, 128),
            'sonar_search': torch.randn(1, batch_size, 1, 256, 256),
            'rgb_anno': torch.tensor([[[0.3, 0.3, 0.3, 0.3]] * batch_size]),  # [N_s, B, 4]
            'sonar_anno': torch.tensor([[[0.3, 0.3, 0.3, 0.3]] * batch_size]),
            'epoch': 1,
        }

        # Run single training step
        actor.train(True)
        optimizer.zero_grad()
        loss, stats = actor(data)
        loss.backward()
        optimizer.step()

        # Loss should be finite
        assert torch.isfinite(loss)
        assert loss.item() > 0


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
