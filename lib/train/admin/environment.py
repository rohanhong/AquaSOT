"""
Environment settings for AquaSOT-3D training.

Defines paths to datasets and other environment-specific settings.
"""

import os


class EnvSettings:
    """Environment settings for training."""

    def __init__(self):
        # Base paths
        self.workspace_dir = os.path.expanduser('~/Workspace/AquaSOT-3D')
        self.data_dir = os.path.join(self.workspace_dir, 'data')

        # AquaSOT dataset paths
        self.aquasot_dir = os.path.join(self.data_dir, 'AquaSOT')

        # Pretrained model paths
        self.pretrained_dir = os.path.join(self.workspace_dir, 'pretrained')

        # Output paths
        self.output_dir = os.path.join(self.workspace_dir, 'output')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')


def env_settings():
    """Get environment settings."""
    return EnvSettings()
