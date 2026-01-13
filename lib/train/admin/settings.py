"""
Training settings for AquaSOT-3D.

Based on OSTrack's settings module.
"""

from .environment import env_settings


class Settings:
    """Training settings container.

    Contains paths to datasets, checkpoints, and training parameters.
    """

    def __init__(self):
        self.set_default()

    def set_default(self):
        """Set default settings."""
        self.env = env_settings()
        self.use_gpu = True
        self.local_rank = -1
        self.device = 'cuda'

        # Will be set by training script
        self.script_name = None
        self.config_name = None
        self.project_path = None
        self.save_dir = None
        self.cfg_file = None
        self.log_file = None

        # Training parameters (overwritten by config)
        self.print_interval = 10
        self.checkpoint_interval = 5
        self.grad_clip_norm = 0.1
        self.batchsize = 16
        self.scheduler_type = 'step'
