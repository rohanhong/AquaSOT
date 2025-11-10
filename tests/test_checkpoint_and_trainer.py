"""
Unit Tests for Checkpoint Manager and Trainer

Tests for checkpoint management, model saving/loading, and training loop.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from pathlib import Path

from aquasot3d.training.checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint
)
from aquasot3d.training.trainer import Trainer, build_trainer
from aquasot3d.training.loss_manager import LossManager
from aquasot3d.training.optimizer import build_optimizer
from aquasot3d.training.scheduler import build_scheduler


class SimpleModel(nn.Module):
    """
    Simple model for testing that mimics AquaSOT3DTracker interface.

    Expected by Trainer:
    - forward(rgb_template, rgb_search, sonar_template, sonar_search)
    - Returns dict with 'rgb_box', 'sonar_box', 'fusion_box' predictions
    """
    def __init__(self):
        super().__init__()
        # Simple feature extractors
        self.rgb_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.sonar_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        # Box prediction heads (output 4 values: cx, cy, w, h)
        self.rgb_box_head = nn.Linear(3, 4)
        self.sonar_box_head = nn.Linear(1, 4)
        self.fusion_box_head = nn.Linear(4, 4)  # Fused features

    def forward(self, rgb_template, rgb_search, sonar_template, sonar_search):
        """
        Forward pass matching Tracker interface.

        Args:
            rgb_template: (B, 3, H, W)
            rgb_search: (B, 3, H, W)
            sonar_template: (B, 1, H, W)
            sonar_search: (B, 1, H, W)

        Returns:
            dict with 'rgb_box', 'sonar_box', 'fusion_box' predictions
        """
        # Extract features
        rgb_feat = self.rgb_encoder(rgb_search)  # (B, 3)
        sonar_feat = self.sonar_encoder(sonar_search)  # (B, 1)

        # Predict boxes
        rgb_box = self.rgb_box_head(rgb_feat)  # (B, 4)
        sonar_box = self.sonar_box_head(sonar_feat)  # (B, 4)

        # Fusion (concatenate and predict)
        fusion_feat = torch.cat([rgb_feat, sonar_feat], dim=1)  # (B, 4)
        fusion_box = self.fusion_box_head(fusion_feat)  # (B, 4)

        return {
            'rgb_box': rgb_box,
            'sonar_box': sonar_box,
            'fusion_box': fusion_box
        }


class TestCheckpointManager:
    """Test suite for CheckpointManager"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialization(self, temp_dir):
        """Test CheckpointManager initialization"""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            max_to_keep=5
        )

        assert manager.checkpoint_dir == Path(temp_dir)
        assert manager.max_to_keep == 5
        assert manager.checkpoint_dir.exists()

    def test_save_checkpoint(self, temp_dir):
        """Test saving checkpoint"""
        manager = CheckpointManager(checkpoint_dir=temp_dir)
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        checkpoint_path = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={'loss': 0.5, 'fusion_auc': 0.75}
        )

        assert checkpoint_path is not None
        assert checkpoint_path.exists()

    def test_load_checkpoint(self, temp_dir):
        """Test loading checkpoint"""
        manager = CheckpointManager(checkpoint_dir=temp_dir)
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        # Save checkpoint
        manager.save(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={'loss': 0.5}
        )

        # Create new model and optimizer
        new_model = SimpleModel()
        new_optimizer = optim.AdamW(new_model.parameters(), lr=0.001)

        # Load checkpoint
        checkpoint = manager.load(
            'checkpoint_epoch_10.pth',
            model=new_model,
            optimizer=new_optimizer
        )

        assert checkpoint['epoch'] == 10
        assert 'loss' in checkpoint['metrics']

    def test_best_model_tracking(self, temp_dir):
        """Test best model tracking"""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            metric_name='fusion_auc',
            mode='max'
        )
        model = SimpleModel()

        # Save checkpoints with increasing metrics
        manager.save(model, epoch=1, metrics={'fusion_auc': 0.70})
        manager.save(model, epoch=2, metrics={'fusion_auc': 0.75})
        manager.save(model, epoch=3, metrics={'fusion_auc': 0.72})

        # Best should be epoch 2
        assert manager.best_epoch == 2
        assert abs(manager.best_metric - 0.75) < 1e-6

        best_path = manager.get_best_checkpoint_path()
        assert best_path is not None
        assert best_path.exists()

    def test_max_checkpoints_to_keep(self, temp_dir):
        """Test checkpoint cleanup"""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            max_to_keep=3
        )
        model = SimpleModel()

        # Save 5 checkpoints
        for epoch in range(5):
            manager.save(model, epoch=epoch)

        # Should only keep 3 + 1 (best)
        checkpoints = list(Path(temp_dir).glob('checkpoint_epoch_*.pth'))
        assert len(checkpoints) <= 3

    def test_resume_training(self, temp_dir):
        """Test resuming training"""
        manager = CheckpointManager(checkpoint_dir=temp_dir)
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        # Save checkpoint
        manager.save(model, optimizer, epoch=10)

        # Resume
        new_model = SimpleModel()
        new_optimizer = optim.AdamW(new_model.parameters(), lr=0.001)

        start_epoch = manager.resume(new_model, new_optimizer)

        assert start_epoch == 11  # Should resume from next epoch

    def test_save_best_only(self, temp_dir):
        """Test save_best_only mode"""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            save_best_only=True,
            metric_name='loss',
            mode='min'
        )
        model = SimpleModel()

        # Save checkpoints
        manager.save(model, epoch=0, metrics={'loss': 1.0})  # Should save (first)
        manager.save(model, epoch=1, metrics={'loss': 1.5})  # Should not save (worse)
        manager.save(model, epoch=2, metrics={'loss': 0.8})  # Should save (best)

        # Should only have best checkpoint
        checkpoints = list(Path(temp_dir).glob('checkpoint_epoch_*.pth'))
        assert len(checkpoints) <= 2  # epoch 0 and epoch 2

    def test_simple_save_load(self, temp_dir):
        """Test simple save/load functions"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        filepath = Path(temp_dir) / 'model.pth'

        # Save
        save_checkpoint(
            filepath,
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics={'loss': 0.3}
        )

        assert filepath.exists()

        # Load
        new_model = SimpleModel()
        new_optimizer = optim.AdamW(new_model.parameters(), lr=0.001)

        checkpoint = load_checkpoint(
            filepath,
            model=new_model,
            optimizer=new_optimizer
        )

        assert checkpoint['epoch'] == 5
        assert 'metrics' in checkpoint


class TestTrainer:
    """Test suite for Trainer"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dummy_data(self):
        """Create dummy dataset"""
        # Create simple dataset
        rgb_template = torch.randn(32, 3, 224, 224)
        rgb_search = torch.randn(32, 3, 224, 224)
        sonar_template = torch.randn(32, 1, 128, 128)
        sonar_search = torch.randn(32, 1, 128, 128)
        rgb_box = torch.randn(32, 4)
        sonar_box = torch.randn(32, 4)

        # Create dataset (returning dict)
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 32

            def __getitem__(self, idx):
                return {
                    'rgb_template': rgb_template[idx],
                    'rgb_search': rgb_search[idx],
                    'sonar_template': sonar_template[idx],
                    'sonar_search': sonar_search[idx],
                    'rgb_box': rgb_box[idx],
                    'sonar_box': sonar_box[idx],
                    'fusion_box': rgb_box[idx]  # Use RGB box as fusion target
                }

        dataset = DummyDataset()
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

        return train_loader, val_loader

    def test_trainer_initialization(self, dummy_data, temp_dir):
        """Test Trainer initialization"""
        train_loader, val_loader = dummy_data

        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=1e-3)
        scheduler = build_scheduler(optimizer, 'cosine', max_epoch=2)
        loss_manager = LossManager()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=2,
            device='cpu',
            use_amp=False
        )

        assert trainer is not None
        assert trainer.max_epochs == 2
        assert not trainer.use_amp

    def test_train_epoch(self, dummy_data, temp_dir):
        """Test training for one epoch"""
        train_loader, val_loader = dummy_data

        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=1e-3)
        scheduler = build_scheduler(optimizer, 'cosine', max_epoch=1)
        loss_manager = LossManager()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            max_epochs=1,
            device='cpu',
            use_amp=False,
            grad_clip_norm=1.0
        )

        # Train one epoch
        metrics = trainer.train_epoch(epoch=0)

        assert isinstance(metrics, dict)
        assert 'total' in metrics
        assert metrics['total'] >= 0

    def test_validate(self, dummy_data, temp_dir):
        """Test validation loop"""
        train_loader, val_loader = dummy_data

        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=1e-3)
        scheduler = build_scheduler(optimizer, 'cosine', max_epoch=1)
        loss_manager = LossManager()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=1,
            device='cpu',
            use_amp=False
        )

        # Validate
        metrics = trainer.validate(epoch=0)

        assert isinstance(metrics, dict)
        assert any(k.startswith('val_') for k in metrics.keys())

    def test_full_training(self, dummy_data, temp_dir):
        """Test full training loop"""
        train_loader, val_loader = dummy_data

        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=1e-3)
        scheduler = build_scheduler(optimizer, 'cosine', max_epoch=2)
        loss_manager = LossManager()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=2,
            device='cpu',
            use_amp=False,
            eval_interval=1
        )

        # Train
        trainer.train(start_epoch=0)

        # Check that training completed
        assert trainer.current_epoch == 1  # 0-indexed, last epoch

    def test_build_trainer(self, dummy_data, temp_dir):
        """Test trainer factory function"""
        train_loader, val_loader = dummy_data

        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=1e-3)
        scheduler = build_scheduler(optimizer, 'cosine', max_epoch=2)

        config = {
            'max_epochs': 2,
            'device': 'cpu',
            'use_amp': False,
            'checkpoint_dir': temp_dir
        }

        trainer = build_trainer(
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            config
        )

        assert trainer is not None
        assert trainer.max_epochs == 2

    def test_gradient_accumulation(self, dummy_data, temp_dir):
        """Test gradient accumulation"""
        train_loader, val_loader = dummy_data

        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=1e-3)
        scheduler = build_scheduler(optimizer, 'cosine', max_epoch=1)
        loss_manager = LossManager()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            max_epochs=1,
            device='cpu',
            use_amp=False,
            gradient_accumulation_steps=2
        )

        # Train one epoch with gradient accumulation
        metrics = trainer.train_epoch(epoch=0)

        assert isinstance(metrics, dict)

    def test_trainer_state_dict(self, dummy_data, temp_dir):
        """Test trainer state save/load"""
        train_loader, val_loader = dummy_data

        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=1e-3)
        scheduler = build_scheduler(optimizer, 'cosine', max_epoch=2)
        loss_manager = LossManager()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            max_epochs=2,
            device='cpu',
            use_amp=False
        )

        trainer.current_epoch = 5
        trainer.global_step = 100

        state = trainer.state_dict()
        assert state['current_epoch'] == 5
        assert state['global_step'] == 100

        new_trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            max_epochs=2,
            device='cpu',
            use_amp=False
        )

        new_trainer.load_state_dict(state)
        assert new_trainer.current_epoch == 5
        assert new_trainer.global_step == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
