"""
Base Trainer class.

From OSTrack with minimal modifications.
"""

import os
import torch


class BaseTrainer:
    """Base trainer class.

    Contains common training logic including checkpointing.
    """

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        Args:
            actor: Actor for forward pass and loss computation
            loaders: List of data loaders [train_loader, val_loader]
            optimizer: Optimizer
            settings: Training settings
            lr_scheduler: Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', 'cuda')
        if self.device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.settings = settings

    def train(self, max_epochs, load_latest=False, fail_safe=True):
        """Training loop.

        Args:
            max_epochs: Maximum number of epochs
            load_latest: Whether to load latest checkpoint
            fail_safe: Whether to catch exceptions and save checkpoint
        """
        if load_latest:
            self.load_checkpoint()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            try:
                self.train_epoch()
            except Exception as e:
                if fail_safe:
                    print(f'Training failed at epoch {epoch}: {e}')
                    self.save_checkpoint()
                raise e

            if self.lr_scheduler is not None:
                if self.settings.scheduler_type != 'cosine':
                    self.lr_scheduler.step()
                else:
                    # For cosine scheduler, step per iteration
                    pass

            # Save checkpoint
            if self.settings.local_rank in [-1, 0]:
                # Always save latest checkpoint (every epoch)
                self.save_checkpoint(save_latest_only=True)

                # Save numbered checkpoint at intervals
                if epoch % self.settings.checkpoint_interval == 0:
                    self.save_checkpoint(save_latest_only=False)

    def train_epoch(self):
        """Train one epoch. To be implemented by subclass."""
        raise NotImplementedError

    def save_checkpoint(self, save_latest_only=False):
        """Save training checkpoint.

        Args:
            save_latest_only: If True, only save checkpoint_latest.pth.
                              If False, save both numbered and latest checkpoints.
        """
        state = {
            'epoch': self.epoch,
            'net': self.actor.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
        }
        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()

        checkpoint_dir = self.settings.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save numbered checkpoint (at intervals)
        if not save_latest_only:
            file_path = os.path.join(checkpoint_dir, f'checkpoint_ep{self.epoch:04d}.pth')
            torch.save(state, file_path)
            print(f'Saved checkpoint: {file_path}')

        # Always save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(state, latest_path)

    def load_checkpoint(self, checkpoint=None):
        """Load training checkpoint.

        Args:
            checkpoint: Path to checkpoint or None for latest
        """
        if checkpoint is None:
            checkpoint = os.path.join(self.settings.checkpoint_dir, 'checkpoint_latest.pth')

        if not os.path.exists(checkpoint):
            print(f'No checkpoint found at {checkpoint}')
            return

        state = torch.load(checkpoint, map_location='cpu')

        self.actor.net.load_state_dict(state['net'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch']
        self.stats = state.get('stats', {})

        if self.lr_scheduler is not None and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])

        print(f'Loaded checkpoint from epoch {self.epoch}: {checkpoint}')
