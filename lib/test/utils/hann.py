"""
Hann window functions for motion constraint.

Reference: OSTrack (lib/test/utils/hann.py)
"""

import torch
import math


def hann1d(sz: int, centered: bool = True) -> torch.Tensor:
    """1D cosine window.

    Args:
        sz: Window size
        centered: If True, window is centered

    Returns:
        1D Hann window tensor
    """
    if centered:
        return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz // 2 + 1).float()))
    return torch.cat([w, w[1:sz - sz // 2].flip((0,))])


def hann2d(sz: torch.Tensor, centered: bool = True) -> torch.Tensor:
    """2D cosine window.

    Args:
        sz: [H, W] window size tensor
        centered: If True, window is centered

    Returns:
        2D Hann window tensor [1, 1, H, W]
    """
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * \
           hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)
