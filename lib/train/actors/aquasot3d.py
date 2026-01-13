"""
AquaSOT-3D Actor for training.

Handles forward pass and loss computation for dual-modal tracking.
Following OSTrack Actor pattern.
"""

import torch
import torch.nn.functional as F

from .base_actor import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.heatmap_utils import generate_heatmap


class AquaSOT3DActor(BaseActor):
    """Actor for training AquaSOT-3D model."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        """
        Args:
            net: AquaSOT3D model
            objective: Dict of loss functions {giou, l1, focal}
            loss_weight: Dict of loss weights
            settings: Training settings
            cfg: Configuration object
        """
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg

    def __call__(self, data):
        """Forward pass and loss computation.

        Args:
            data: Dict containing:
                - rgb_template: [N_t, B, 3, H, W]
                - rgb_search: [N_s, B, 3, H, W]
                - sonar_template: [N_t, B, 1, H, W]
                - sonar_search: [N_s, B, 1, H, W]
                - rgb_anno: [N_s, B, 4] (x, y, w, h)
                - sonar_anno: [N_s, B, 4]

        Returns:
            loss: Total loss
            status: Dict of loss components
        """
        # Forward pass
        out_dict = self.forward_pass(data)

        # Compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        """Execute forward pass.

        Args:
            data: Training batch

        Returns:
            out_dict: Model outputs
        """
        # Get templates and search regions
        # Shape: [N, B, C, H, W] -> [B, C, H, W]
        rgb_template = data['rgb_template'][0]
        rgb_search = data['rgb_search'][0]
        sonar_template = data['sonar_template'][0]
        sonar_search = data['sonar_search'][0]

        # Move to GPU
        device = next(self.net.parameters()).device
        rgb_template = rgb_template.to(device)
        rgb_search = rgb_search.to(device)
        sonar_template = sonar_template.to(device)
        sonar_search = sonar_search.to(device)

        # Forward pass through model
        out_dict = self.net(
            rgb_template=rgb_template,
            rgb_search=rgb_search,
            sonar_template=sonar_template,
            sonar_search=sonar_search
        )

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        """Compute all losses.

        Args:
            pred_dict: Model predictions
            gt_dict: Ground truth data

        Returns:
            loss: Total weighted loss
            status: Dict of individual losses for logging
        """
        # Get device from predictions
        device = pred_dict['rgb_boxes'].device

        # Get GT boxes (x, y, w, h format) and move to GPU
        gt_bbox_rgb = gt_dict['rgb_anno'][-1].to(device)  # [B, 4]
        gt_bbox_sonar = gt_dict['sonar_anno'][-1].to(device)  # [B, 4]

        # Generate GT heatmaps
        search_size = self.cfg.DATA.SEARCH.SIZE
        stride = self.cfg.MODEL.BACKBONE.RGB.STRIDE

        # Move annotations to device before generating heatmaps
        rgb_anno_device = [anno.to(device) for anno in gt_dict['rgb_anno']]
        sonar_anno_device = [anno.to(device) for anno in gt_dict['sonar_anno']]

        gt_gaussian_rgb = generate_heatmap(
            rgb_anno_device, search_size, stride
        )[-1].unsqueeze(1)  # [B, 1, H, W]

        gt_gaussian_sonar = generate_heatmap(
            sonar_anno_device, search_size, stride
        )[-1].unsqueeze(1)

        # Compute RGB losses
        rgb_giou, rgb_l1, rgb_focal, rgb_iou = self._compute_modality_loss(
            pred_dict['rgb_boxes'],
            pred_dict['rgb_score_map'],
            gt_bbox_rgb,
            gt_gaussian_rgb
        )

        # Compute Sonar losses
        sonar_giou, sonar_l1, sonar_focal, sonar_iou = self._compute_modality_loss(
            pred_dict['sonar_boxes'],
            pred_dict['sonar_score_map'],
            gt_bbox_sonar,
            gt_gaussian_sonar
        )

        # Weighted sum
        loss = (
            self.loss_weight['giou'] * (rgb_giou + sonar_giou) +
            self.loss_weight['l1'] * (rgb_l1 + sonar_l1) +
            self.loss_weight['focal'] * (rgb_focal + sonar_focal)
        )

        if return_status:
            status = {
                "Loss/total": loss.item(),
                "Loss/rgb_giou": rgb_giou.item(),
                "Loss/rgb_l1": rgb_l1.item(),
                "Loss/rgb_focal": rgb_focal.item(),
                "Loss/sonar_giou": sonar_giou.item(),
                "Loss/sonar_l1": sonar_l1.item(),
                "Loss/sonar_focal": sonar_focal.item(),
                "IoU/rgb": rgb_iou.item(),
                "IoU/sonar": sonar_iou.item(),
            }
            return loss, status
        else:
            return loss

    def _compute_modality_loss(self, pred_boxes, pred_score_map, gt_bbox, gt_gaussian):
        """Compute losses for a single modality.

        Args:
            pred_boxes: [B, 4] (cx, cy, w, h) normalized
            pred_score_map: [B, 1, H, W]
            gt_bbox: [B, 4] (x, y, w, h) normalized
            gt_gaussian: [B, 1, H, W]

        Returns:
            giou_loss, l1_loss, focal_loss, mean_iou
        """
        # Check for NaN
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs NaN! Stop Training")

        # Convert to xyxy format
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # [B, 4]
        gt_boxes_xyxy = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0)  # [B, 4]

        # GIoU loss
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_xyxy, gt_boxes_xyxy)
        except Exception:
            giou_loss = torch.tensor(0.0, device=pred_boxes.device)
            iou = torch.tensor(0.0, device=pred_boxes.device)

        # L1 loss
        l1_loss = self.objective['l1'](pred_boxes_xyxy, gt_boxes_xyxy)

        # Focal loss for score map
        if pred_score_map is not None:
            focal_loss = self.objective['focal'](pred_score_map, gt_gaussian)
        else:
            focal_loss = torch.tensor(0.0, device=pred_boxes.device)

        mean_iou = iou.detach().mean()

        return giou_loss, l1_loss, focal_loss, mean_iou
