"""
Dual Consistency Loss for G2-DCMA Fusion

This module implements two types of consistency constraints for cross-modal fusion:
1. Component Consistency: Ensures RGB and Sonar branches predict consistent 3D components (e.g., X coordinate)
2. Representation Consistency: Uses InfoNCE contrastive loss to align cross-modal representations

These consistency constraints enable calibration-free learning by enforcing
geometric and semantic alignment between modalities without explicit calibration.

Components:
- ComponentConsistencyLoss: L1/L2 distance between modality-specific predictions
- InfoNCELoss: Contrastive learning for cross-modal representation alignment
- DualConsistencyLoss: Combined component + representation consistency

Reference:
    AquaSOT-3D Technical Specification Section 3(d): Dual Consistency
    SimCLR (ICML 2020): https://arxiv.org/abs/2002.05709 (InfoNCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from aquasot3d.core.base_module import BaseModule


class ComponentConsistencyLoss(BaseModule):
    """
    Component Consistency Loss between RGB and Sonar predictions.

    Enforces consistency between modality-specific predictions of shared 3D
    components (e.g., X coordinate). This constraint helps learn a unified
    3D representation without explicit calibration.

    Args:
        loss_type: Type of distance metric ('l1', 'l2', 'smooth_l1')
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        reduction: str = 'mean'
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        # Loss function selection
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(
        self,
        pred_rgb: torch.Tensor,
        pred_sonar: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute component consistency loss.

        Args:
            pred_rgb: RGB prediction (e.g., X coordinate) [B, ...]
            pred_sonar: Sonar prediction (e.g., X coordinate) [B, ...]
            valid_mask: Optional mask for valid predictions [B, ...]

        Returns:
            loss: Component consistency loss scalar
        """
        # Compute distance
        loss = self.loss_fn(pred_rgb, pred_sonar)

        # Apply valid mask if provided
        if valid_mask is not None:
            if self.reduction == 'none':
                loss = loss * valid_mask
                loss = loss.sum() / (valid_mask.sum() + 1e-6)
            else:
                # Recompute with mask
                diff = torch.abs(pred_rgb - pred_sonar) if self.loss_type == 'l1' else (pred_rgb - pred_sonar) ** 2
                diff = diff * valid_mask
                loss = diff.sum() / (valid_mask.sum() + 1e-6)

        return loss


class InfoNCELoss(BaseModule):
    """
    InfoNCE (Noise Contrastive Estimation) Loss for cross-modal representation learning.

    Pulls together representations from the same frame (positive pairs) and pushes
    apart representations from different frames (negative pairs). This encourages
    the network to learn modality-invariant representations.

    Args:
        temperature: Temperature parameter for softmax (smaller = harder negatives)
        normalize: Whether to L2-normalize embeddings before computing similarity
        similarity_metric: Similarity metric ('cosine', 'dot')
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        similarity_metric: str = 'cosine'
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.similarity_metric = similarity_metric

    def compute_similarity(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise similarity between two sets of embeddings.

        Args:
            embeddings_a: First set of embeddings [B, D]
            embeddings_b: Second set of embeddings [B, D]

        Returns:
            similarity: Pairwise similarity matrix [B, B]
        """
        if self.normalize:
            embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
            embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        if self.similarity_metric == 'cosine':
            # Cosine similarity (already normalized if self.normalize=True)
            similarity = torch.matmul(embeddings_a, embeddings_b.T)
        elif self.similarity_metric == 'dot':
            # Dot product
            similarity = torch.matmul(embeddings_a, embeddings_b.T)
        else:
            raise ValueError(f"Unknown similarity_metric: {self.similarity_metric}")

        return similarity

    def forward(
        self,
        embeddings_rgb: torch.Tensor,
        embeddings_sonar: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE contrastive loss.

        Args:
            embeddings_rgb: RGB embeddings [B, D]
            embeddings_sonar: Sonar embeddings [B, D]
            labels: Optional frame labels for positive pair identification [B]
                   If None, assumes batch contains positive pairs at same indices

        Returns:
            dict containing:
                - loss: InfoNCE loss scalar
                - rgb_to_sonar_acc: Accuracy of retrieving sonar from RGB
                - sonar_to_rgb_acc: Accuracy of retrieving RGB from sonar
        """
        B, D = embeddings_rgb.shape
        device = embeddings_rgb.device

        # Compute similarity matrix
        similarity = self.compute_similarity(embeddings_rgb, embeddings_sonar)  # [B, B]
        similarity = similarity / self.temperature

        # Create positive pair mask
        if labels is None:
            # Assume diagonal pairs are positive (same index = same frame)
            positive_mask = torch.eye(B, device=device, dtype=torch.bool)
        else:
            # Create mask based on labels
            labels = labels.contiguous().view(-1, 1)
            positive_mask = torch.eq(labels, labels.T)

        # Compute InfoNCE loss for RGB -> Sonar direction
        # For each RGB embedding, the positive is the corresponding Sonar embedding
        rgb_to_sonar_logits = similarity  # [B, B]
        rgb_to_sonar_labels = torch.arange(B, device=device)  # Diagonal are positives

        # Cross-entropy loss (treats row as query, columns as candidates)
        rgb_to_sonar_loss = F.cross_entropy(
            rgb_to_sonar_logits,
            rgb_to_sonar_labels,
            reduction='mean'
        )

        # Compute InfoNCE loss for Sonar -> RGB direction
        sonar_to_rgb_logits = similarity.T  # [B, B]
        sonar_to_rgb_labels = torch.arange(B, device=device)

        sonar_to_rgb_loss = F.cross_entropy(
            sonar_to_rgb_logits,
            sonar_to_rgb_labels,
            reduction='mean'
        )

        # Total loss (symmetric)
        loss = (rgb_to_sonar_loss + sonar_to_rgb_loss) / 2.0

        # Compute retrieval accuracy for monitoring
        with torch.no_grad():
            rgb_to_sonar_pred = rgb_to_sonar_logits.argmax(dim=1)
            rgb_to_sonar_acc = (rgb_to_sonar_pred == rgb_to_sonar_labels).float().mean()

            sonar_to_rgb_pred = sonar_to_rgb_logits.argmax(dim=1)
            sonar_to_rgb_acc = (sonar_to_rgb_pred == sonar_to_rgb_labels).float().mean()

        return {
            'loss': loss,
            'rgb_to_sonar_loss': rgb_to_sonar_loss,
            'sonar_to_rgb_loss': sonar_to_rgb_loss,
            'rgb_to_sonar_acc': rgb_to_sonar_acc,
            'sonar_to_rgb_acc': sonar_to_rgb_acc
        }


class DualConsistencyLoss(BaseModule):
    """
    Combined Dual Consistency Loss (Component + Representation).

    Integrates both component-level consistency (e.g., X coordinate alignment)
    and representation-level consistency (contrastive learning) for holistic
    cross-modal alignment.

    Args:
        component_weight: Weight for component consistency loss
        representation_weight: Weight for InfoNCE representation loss
        component_loss_type: Loss type for component consistency
        infonce_temperature: Temperature for InfoNCE
    """

    def __init__(
        self,
        component_weight: float = 1.0,
        representation_weight: float = 1.0,
        component_loss_type: str = 'l1',
        infonce_temperature: float = 0.07
    ):
        super().__init__()
        self.component_weight = component_weight
        self.representation_weight = representation_weight

        # Component consistency loss
        self.component_loss = ComponentConsistencyLoss(
            loss_type=component_loss_type,
            reduction='mean'
        )

        # Representation consistency loss (InfoNCE)
        self.infonce_loss = InfoNCELoss(
            temperature=infonce_temperature,
            normalize=True
        )

    def forward(
        self,
        pred_rgb_component: torch.Tensor,
        pred_sonar_component: torch.Tensor,
        embeddings_rgb: torch.Tensor,
        embeddings_sonar: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dual consistency loss.

        Args:
            pred_rgb_component: RGB component prediction (e.g., X) [B, ...]
            pred_sonar_component: Sonar component prediction (e.g., X) [B, ...]
            embeddings_rgb: RGB feature embeddings [B, D]
            embeddings_sonar: Sonar feature embeddings [B, D]
            valid_mask: Optional mask for valid predictions [B, ...]
            frame_labels: Optional frame labels for InfoNCE [B]

        Returns:
            dict containing:
                - total_loss: Weighted sum of component and representation losses
                - component_loss: Component consistency loss
                - infonce_loss: InfoNCE representation loss
                - rgb_to_sonar_acc: Cross-modal retrieval accuracy
                - sonar_to_rgb_acc: Cross-modal retrieval accuracy
        """
        # Component consistency loss
        comp_loss = self.component_loss(
            pred_rgb_component,
            pred_sonar_component,
            valid_mask
        )

        # Representation consistency loss (InfoNCE)
        infonce_output = self.infonce_loss(
            embeddings_rgb,
            embeddings_sonar,
            frame_labels
        )

        # Total loss
        total_loss = (
            self.component_weight * comp_loss +
            self.representation_weight * infonce_output['loss']
        )

        return {
            'total_loss': total_loss,
            'component_loss': comp_loss,
            'infonce_loss': infonce_output['loss'],
            'rgb_to_sonar_loss': infonce_output['rgb_to_sonar_loss'],
            'sonar_to_rgb_loss': infonce_output['sonar_to_rgb_loss'],
            'rgb_to_sonar_acc': infonce_output['rgb_to_sonar_acc'],
            'sonar_to_rgb_acc': infonce_output['sonar_to_rgb_acc']
        }


class ProjectionHead(BaseModule):
    """
    Projection head for mapping features to embedding space for contrastive learning.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        num_layers: Number of projection layers
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.projection = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project features to embedding space.

        Args:
            features: Input features [B, C] or [B, C, H, W]

        Returns:
            embeddings: Projected embeddings [B, output_dim]
        """
        # Global average pooling if spatial dimensions exist
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        # Check if input dimension matches expected dimension
        actual_dim = features.shape[1]
        expected_dim = self.projection[0].in_features
        if actual_dim != expected_dim:
            # Create dynamic projection layer to adapt input dimension
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(actual_dim, expected_dim).to(features.device)
            features = self.input_proj(features)

        embeddings = self.projection(features)
        return embeddings


def build_dual_consistency_loss(config: dict) -> DualConsistencyLoss:
    """
    Factory function to build DualConsistencyLoss from config.

    Args:
        config: Configuration dictionary

    Returns:
        DualConsistencyLoss instance
    """
    return DualConsistencyLoss(
        component_weight=config.get('component_weight', 1.0),
        representation_weight=config.get('representation_weight', 1.0),
        component_loss_type=config.get('component_loss_type', 'l1'),
        infonce_temperature=config.get('infonce_temperature', 0.07)
    )


def build_projection_head(config: dict) -> ProjectionHead:
    """
    Factory function to build ProjectionHead from config.

    Args:
        config: Configuration dictionary

    Returns:
        ProjectionHead instance
    """
    return ProjectionHead(
        input_dim=config.get('input_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        output_dim=config.get('output_dim', 128),
        num_layers=config.get('num_layers', 2)
    )
