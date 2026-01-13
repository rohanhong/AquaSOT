"""
TensorBoard logging utilities.
"""

import os
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    """TensorBoard writer for training logs."""

    def __init__(self, directory, loader_names):
        """
        Args:
            directory: TensorBoard log directory
            loader_names: List of data loader names
        """
        self.directory = directory
        self.loader_names = loader_names
        self.writer = SummaryWriter(log_dir=directory)

    def write_info(self, script_name, description):
        """Write experiment info."""
        self.writer.add_text('Script', script_name)
        self.writer.add_text('Description', description)

    def write_epoch(self, stats, epoch):
        """Write epoch statistics.

        Args:
            stats: OrderedDict of loader -> stat_name -> value
            epoch: Current epoch
        """
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for stat_name, stat_value in loader_stats.items():
                if hasattr(stat_value, 'avg'):
                    val = stat_value.avg
                elif hasattr(stat_value, 'val'):
                    val = stat_value.val
                else:
                    val = stat_value

                tag = f'{loader_name}/{stat_name}'
                self.writer.add_scalar(tag, val, epoch)

    def close(self):
        """Close the writer."""
        self.writer.close()
