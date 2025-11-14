"""
Regression test for Audit 2025111414 Finding 4: eval_interval checkpoint logic

Validates that when eval_interval > 1, best checkpoint saving is correctly
gated on validation epochs to avoid "metric not found" errors.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import torch
import os
import tempfile
import shutil
from pathlib import Path


def test_trainer_checkpoint_logic_with_eval_interval():
    """
    Test that best checkpoint logic is gated on val_metrics availability.

    This is a critical regression test for Audit 2025111414 Finding 4.
    Without proper gating, non-validation epochs will trigger "metric not found"
    errors and fail to save best checkpoints.
    """
    from aquasot3d.training.trainer import Trainer
    from aquasot3d.training.checkpoint import CheckpointManager

    # Create temporary checkpoint directory
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            def train(self, mode=True):
                self.training = mode
                return self

            def eval(self):
                return self.train(False)

        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Create mock dataloaders (minimal data)
        train_data = [(torch.randn(2, 10), torch.randn(2, 10)) for _ in range(4)]
        val_data = [(torch.randn(2, 10), torch.randn(2, 10)) for _ in range(2)]

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints_to_keep=3,
            save_best=True,
            metric_name='val_total',
            mode='min'
        )

        # Create trainer with eval_interval=3
        # This means validation runs at epochs 3, 6, 9, ... and final epoch
        eval_interval = 3
        max_epochs = 5

        # Mock loss manager
        class MockLossManager:
            def __init__(self):
                self.loss_weights = {'train_loss': 1.0}

            def compute_loss(self, *args, **kwargs):
                return {
                    'total': torch.tensor(1.0),
                    'train_loss': torch.tensor(1.0)
                }

            def get_loss_weights(self):
                return self.loss_weights

        loss_manager = MockLossManager()

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_data,
            val_loader=val_data,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            checkpoint_manager=checkpoint_manager,
            max_epochs=max_epochs,
            eval_interval=eval_interval,
            device='cpu',
            use_amp=False
        )

        # Override train_epoch and validate to return mock metrics
        def mock_train_epoch(epoch):
            return {'train_loss': 0.5 - epoch * 0.05}

        def mock_validate(epoch):
            return {'val_total': 0.8 - epoch * 0.1}

        trainer.train_epoch = mock_train_epoch
        trainer.validate = mock_validate

        # Track which epochs attempt to save best checkpoint
        best_checkpoint_attempts = []
        original_save = checkpoint_manager.save

        def tracked_save(*args, **kwargs):
            if kwargs.get('is_best', False):
                # Record epoch when best checkpoint is saved
                epoch = kwargs.get('epoch', -1)
                best_checkpoint_attempts.append(epoch)
            return original_save(*args, **kwargs)

        checkpoint_manager.save = tracked_save

        # Run training
        trainer.train(start_epoch=0)

        # Verify best checkpoint saving logic
        # With eval_interval=3 and max_epochs=5:
        # - Validation runs at epochs: 2 (3rd epoch), 4 (final epoch) [0-indexed]
        # - Best checkpoint should ONLY be saved on these epochs

        # Check that best checkpoint was attempted ONLY on validation epochs
        expected_best_epochs = [2, 4]  # 0-indexed epochs where validation runs

        assert len(best_checkpoint_attempts) > 0, \
            "Best checkpoint should be saved at least once"

        for attempt_epoch in best_checkpoint_attempts:
            assert attempt_epoch in expected_best_epochs, \
                f"Best checkpoint saved on epoch {attempt_epoch}, but validation only runs on {expected_best_epochs}"

        # Verify checkpoint files exist
        checkpoint_files = list(Path(checkpoint_dir).glob('*.pth'))
        assert len(checkpoint_files) > 0, "At least one checkpoint should be saved"

        # Check for best checkpoint
        best_checkpoint = Path(checkpoint_dir) / 'checkpoint_best.pth'
        assert best_checkpoint.exists(), "Best checkpoint should exist"

        print(f"✅ Test passed: Best checkpoint saved on epochs {best_checkpoint_attempts}")
        print(f"   Expected validation epochs: {expected_best_epochs}")

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_checkpoint_logic_code_inspection():
    """
    Code inspection test to verify the checkpoint gating logic exists.

    This test reads the trainer.py source to ensure the fix is present.
    """
    trainer_path = 'src/aquasot3d/training/trainer.py'
    assert os.path.exists(trainer_path), "Trainer file must exist"

    with open(trainer_path, 'r') as f:
        content = f.read()

    # Verify the critical fix is present
    assert 'if self.save_best and val_metrics is not None:' in content, \
        "Best checkpoint saving must be gated on val_metrics is not None"

    # Verify the validation interval logic is present
    assert '(epoch + 1) % self.eval_interval == 0' in content, \
        "Validation must respect eval_interval"

    # Verify both conditions work together
    lines = content.split('\n')

    # Find the validation gating line
    val_gate_line = None
    checkpoint_gate_line = None

    for i, line in enumerate(lines):
        if '(epoch + 1) % self.eval_interval == 0' in line:
            val_gate_line = i
        if 'if self.save_best and val_metrics is not None:' in line:
            checkpoint_gate_line = i

    assert val_gate_line is not None, "Validation gating must exist"
    assert checkpoint_gate_line is not None, "Checkpoint gating must exist"
    assert checkpoint_gate_line > val_gate_line, \
        "Checkpoint gating must come after validation gating in code flow"

    print(f"✅ Code inspection passed:")
    print(f"   Validation gate at line {val_gate_line}")
    print(f"   Checkpoint gate at line {checkpoint_gate_line}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
