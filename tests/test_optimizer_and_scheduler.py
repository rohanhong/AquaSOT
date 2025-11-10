"""
Unit Tests for Optimizer and Scheduler

Tests for optimizer configuration, parameter grouping, and learning rate scheduling.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from aquasot3d.training.optimizer import (
    build_optimizer,
    get_parameter_groups,
    clip_gradients,
    freeze_parameters,
    unfreeze_parameters
)
from aquasot3d.training.scheduler import (
    build_scheduler,
    CosineAnnealingWarmupRestarts,
    PolynomialLRDecay,
    WarmupScheduler,
    get_current_lr,
    adjust_learning_rate
)


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 20)
        )
        self.fusion = nn.Linear(20, 10)
        self.head = nn.Linear(10, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fusion(x)
        x = self.head(x)
        return x


class TestOptimizer:
    """Test suite for optimizer utilities"""

    def test_build_optimizer_sgd(self):
        """Test SGD optimizer creation"""
        model = SimpleModel()
        optimizer = build_optimizer(model, 'sgd', base_lr=0.01, momentum=0.9)

        assert isinstance(optimizer, optim.SGD)
        assert len(optimizer.param_groups) > 0

    def test_build_optimizer_adam(self):
        """Test Adam optimizer creation"""
        model = SimpleModel()
        optimizer = build_optimizer(model, 'adam', base_lr=0.001)

        assert isinstance(optimizer, optim.Adam)

    def test_build_optimizer_adamw(self):
        """Test AdamW optimizer creation"""
        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=2e-4)

        assert isinstance(optimizer, optim.AdamW)

    def test_parameter_grouping(self):
        """Test automatic parameter grouping"""
        model = SimpleModel()
        optimizer = build_optimizer(model, 'adamw', base_lr=2e-4)

        # Should have multiple parameter groups (backbone, fusion, head)
        assert len(optimizer.param_groups) >= 3

        # Check different learning rates
        lrs = [group['lr'] for group in optimizer.param_groups]
        assert len(set(lrs)) > 1  # Should have different LRs

    def test_custom_parameter_groups(self):
        """Test custom parameter groups"""
        model = SimpleModel()

        custom_groups = [
            {'params': model.backbone.parameters(), 'lr': 1e-4},
            {'params': model.fusion.parameters(), 'lr': 2e-4},
            {'params': model.head.parameters(), 'lr': 2e-4}
        ]

        optimizer = build_optimizer(
            model,
            'adamw',
            base_lr=2e-4,
            parameter_groups=custom_groups
        )

        assert len(optimizer.param_groups) == 3
        assert optimizer.param_groups[0]['lr'] == 1e-4
        assert optimizer.param_groups[1]['lr'] == 2e-4

    def test_gradient_clipping(self):
        """Test gradient clipping"""
        model = SimpleModel()
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)

        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Forward and backward
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

        # Clip gradients
        grad_norm = clip_gradients(model, max_norm=1.0)

        assert isinstance(grad_norm, torch.Tensor)
        assert grad_norm.item() >= 0

    def test_freeze_parameters(self):
        """Test parameter freezing"""
        model = SimpleModel()

        # Count trainable parameters before freezing
        trainable_before = sum(p.requires_grad for p in model.parameters())

        # Freeze backbone
        freeze_parameters(model, ['backbone'])

        # Count trainable parameters after freezing
        trainable_after = sum(p.requires_grad for p in model.parameters())

        assert trainable_after < trainable_before

        # Check that backbone parameters are frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze_parameters(self):
        """Test parameter unfreezing"""
        model = SimpleModel()

        # Freeze all
        freeze_parameters(model, ['backbone', 'fusion', 'head'])

        # Unfreeze specific
        unfreeze_parameters(model, ['fusion', 'head'])

        # Check
        for param in model.backbone.parameters():
            assert not param.requires_grad
        for param in model.fusion.parameters():
            assert param.requires_grad
        for param in model.head.parameters():
            assert param.requires_grad

    def test_get_parameter_groups(self):
        """Test parameter group extraction"""
        model = SimpleModel()
        groups = get_parameter_groups(model, base_lr=2e-4)

        assert len(groups) >= 3
        assert all('params' in g for g in groups)
        assert all('lr' in g for g in groups)


class TestScheduler:
    """Test suite for learning rate schedulers"""

    def test_build_cosine_scheduler(self):
        """Test cosine annealing scheduler"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        scheduler = build_scheduler(
            optimizer,
            'cosine',
            max_epoch=60,
            warmup_epochs=5
        )

        assert isinstance(scheduler, CosineAnnealingWarmupRestarts)

    def test_build_step_scheduler(self):
        """Test step scheduler"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        scheduler = build_scheduler(
            optimizer,
            'step',
            step_size=20,
            gamma=0.1
        )

        assert scheduler is not None

    def test_build_multistep_scheduler(self):
        """Test multi-step scheduler"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        scheduler = build_scheduler(
            optimizer,
            'multistep',
            milestones=[20, 40],
            gamma=0.1
        )

        assert scheduler is not None

    def test_build_polynomial_scheduler(self):
        """Test polynomial scheduler"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        scheduler = build_scheduler(
            optimizer,
            'polynomial',
            max_epoch=60,
            power=0.9
        )

        assert isinstance(scheduler, PolynomialLRDecay)

    def test_cosine_warmup_scheduler(self):
        """Test cosine annealing with warmup"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            T_0=50,
            warmup_epochs=5,
            eta_min=1e-6
        )

        # Get initial LR (during warmup)
        initial_lr = get_current_lr(optimizer)

        # Step through warmup
        for _ in range(5):
            scheduler.step()

        warmup_end_lr = get_current_lr(optimizer)

        # Step through cosine annealing
        for _ in range(25):
            scheduler.step()

        mid_lr = get_current_lr(optimizer)

        # LR should change during warmup and annealing
        assert warmup_end_lr[0] > initial_lr[0]
        assert mid_lr[0] < warmup_end_lr[0]

    def test_polynomial_decay(self):
        """Test polynomial learning rate decay"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        scheduler = PolynomialLRDecay(
            optimizer,
            max_epoch=60,
            power=0.9,
            eta_min=1e-6
        )

        initial_lr = get_current_lr(optimizer)[0]

        # Step through epochs
        for _ in range(30):
            scheduler.step()

        mid_lr = get_current_lr(optimizer)[0]

        # LR should decay
        assert mid_lr < initial_lr

    def test_warmup_scheduler_wrapper(self):
        """Test warmup scheduler wrapper"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        scheduler = WarmupScheduler(
            base_scheduler,
            warmup_epochs=5,
            warmup_start_lr=1e-6
        )

        initial_lr = get_current_lr(optimizer)[0]

        # Step through warmup
        for _ in range(5):
            scheduler.step()

        warmup_end_lr = get_current_lr(optimizer)[0]

        assert warmup_end_lr > initial_lr

    def test_get_current_lr(self):
        """Test getting current learning rate"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        lrs = get_current_lr(optimizer)

        assert isinstance(lrs, list)
        assert len(lrs) == len(optimizer.param_groups)
        assert lrs[0] == 2e-4

    def test_adjust_learning_rate(self):
        """Test manual LR adjustment"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        # Adjust LR
        new_lr = adjust_learning_rate(
            optimizer,
            epoch=30,
            max_epoch=60,
            base_lr=2e-4,
            schedule_type='cosine'
        )

        # Check that LR was updated
        current_lr = get_current_lr(optimizer)[0]
        assert abs(current_lr - new_lr) < 1e-10

    def test_scheduler_state_dict(self):
        """Test scheduler state save/load"""
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)

        scheduler = build_scheduler(optimizer, 'cosine', max_epoch=60)

        # Step scheduler
        for _ in range(10):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()

        # Create new scheduler and load state
        new_optimizer = optim.AdamW(model.parameters(), lr=2e-4)
        new_scheduler = build_scheduler(new_optimizer, 'cosine', max_epoch=60)
        new_scheduler.load_state_dict(state)

        # States should match
        assert new_scheduler.last_epoch == scheduler.last_epoch


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
