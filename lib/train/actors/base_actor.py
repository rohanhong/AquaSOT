"""
Base Actor class for training.

From OSTrack with minimal modifications.
"""

from torch import nn


class BaseActor(nn.Module):
    """Base class for actor.

    The actor handles the forward pass and loss computation during training.
    """

    def __init__(self, net, objective):
        """
        Args:
            net: The network to train
            objective: Dict of loss functions
        """
        super().__init__()
        self.net = net
        self.objective = objective

    def __call__(self, data):
        """Forward pass and loss computation.

        To be implemented by subclasses.

        Args:
            data: Training batch data

        Returns:
            loss: Total loss
            status: Dict of loss components for logging
        """
        raise NotImplementedError

    def train(self, mode=True):
        """Set training mode."""
        self.net.train(mode)
        return self

    def to(self, device):
        """Move to device."""
        self.net = self.net.to(device)
        return self
