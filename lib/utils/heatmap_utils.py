"""
Heatmap generation utilities.
"""

import torch
import math


def generate_heatmap(bboxes, search_size, stride):
    """Generate Gaussian heatmap from bounding boxes.

    Args:
        bboxes: [N, B, 4] (x, y, w, h) normalized to [0, 1]
        search_size: Search region size (e.g., 256)
        stride: Feature stride (e.g., 16)

    Returns:
        heatmaps: [N, B, H, W] Gaussian heatmaps
    """
    feat_size = search_size // stride
    heatmaps = []

    for bbox in bboxes:
        heatmap = generate_single_heatmap(bbox, feat_size)
        heatmaps.append(heatmap)

    return heatmaps


def generate_single_heatmap(bbox, feat_size):
    """Generate Gaussian heatmap for a batch of boxes.

    Args:
        bbox: [B, 4] (x, y, w, h) normalized
        feat_size: Feature map size

    Returns:
        heatmap: [B, H, W]
    """
    batch_size = bbox.shape[0]
    device = bbox.device

    # Convert to center coordinates
    cx = (bbox[:, 0] + bbox[:, 2] / 2) * feat_size  # [B]
    cy = (bbox[:, 1] + bbox[:, 3] / 2) * feat_size  # [B]
    w = bbox[:, 2] * feat_size  # [B]
    h = bbox[:, 3] * feat_size  # [B]

    # Compute sigma based on box size
    sigma_x = w / 6.0  # [B]
    sigma_y = h / 6.0  # [B]
    sigma_x = torch.clamp(sigma_x, min=1.0)
    sigma_y = torch.clamp(sigma_y, min=1.0)

    # Create coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(feat_size, device=device, dtype=torch.float32),
        torch.arange(feat_size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]

    # Reshape center and sigma for broadcasting
    cx = cx.view(-1, 1, 1)  # [B, 1, 1]
    cy = cy.view(-1, 1, 1)  # [B, 1, 1]
    sigma_x = sigma_x.view(-1, 1, 1)  # [B, 1, 1]
    sigma_y = sigma_y.view(-1, 1, 1)  # [B, 1, 1]

    # Compute Gaussian
    heatmap = torch.exp(
        -((x_grid - cx) ** 2 / (2 * sigma_x ** 2) +
          (y_grid - cy) ** 2 / (2 * sigma_y ** 2))
    )

    return heatmap
