"""
AquaSOT: Dual-stream RGB-Sonar Tracker.

This is the main tracker model following OSTrack's simplified design.
Total: ~150 lines (vs. 1349 in old implementation).
"""

import torch
import torch.nn as nn

from .vit import build_vit_rgb, build_vit_sonar
from .fusion import build_fusion
from lib.models.layers.head import build_box_head


class AquaSOT(nn.Module):
    """Dual-stream RGB-Sonar tracking model.

    Architecture:
        RGB Template + Search -> ViT RGB -> RGB Features
        Sonar Template + Search -> ViT Sonar -> Sonar Features
        RGB Features + Sonar Features -> DCMA Fusion
        Fused RGB -> CenterHead -> RGB Bbox
        Fused Sonar -> CenterHead -> Sonar Bbox
    """

    def __init__(self, rgb_backbone, sonar_backbone, fusion, box_head, head_type="CENTER"):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.sonar_backbone = sonar_backbone
        self.fusion = fusion
        self.box_head = box_head
        self.head_type = head_type

        # Feature dimensions
        self.embed_dim = rgb_backbone.embed_dim
        self.feat_sz = box_head.feat_sz
        self.feat_len_s = self.feat_sz ** 2  # Search region feature length

    def forward(self, rgb_template, rgb_search, sonar_template, sonar_search,
                gt_score_map_rgb=None, gt_score_map_sonar=None):
        """
        Args:
            rgb_template: [B, 3, H_t, W_t] - RGB template
            rgb_search: [B, 3, H_s, W_s] - RGB search region
            sonar_template: [B, 1, H_t, W_t] - Sonar template
            sonar_search: [B, 1, H_s, W_s] - Sonar search region
            gt_score_map_rgb: Optional GT score map for RGB (training)
            gt_score_map_sonar: Optional GT score map for Sonar (training)

        Returns:
            dict with:
                - rgb_boxes: [B, 4] (cx, cy, w, h) normalized
                - sonar_boxes: [B, 4]
                - rgb_score_map: [B, 1, H, W]
                - sonar_score_map: [B, 1, H, W]
                - (other auxiliary outputs)
        """
        # Extract backbone features
        rgb_feat, rgb_aux = self.rgb_backbone(rgb_template, rgb_search)
        sonar_feat, sonar_aux = self.sonar_backbone(sonar_template, sonar_search)

        # Cross-modal fusion
        fused_rgb, fused_sonar = self.fusion(rgb_feat, sonar_feat)

        # Get search region features and predict boxes
        rgb_out = self.forward_head(fused_rgb, gt_score_map_rgb)
        sonar_out = self.forward_head(fused_sonar, gt_score_map_sonar)

        return {
            'rgb_boxes': rgb_out['pred_boxes'],
            'sonar_boxes': sonar_out['pred_boxes'],
            'rgb_score_map': rgb_out['score_map'],
            'sonar_score_map': sonar_out['score_map'],
            'rgb_size_map': rgb_out['size_map'],
            'sonar_size_map': sonar_out['size_map'],
            'rgb_offset_map': rgb_out['offset_map'],
            'sonar_offset_map': sonar_out['offset_map'],
        }

    def forward_head(self, cat_feature, gt_score_map=None):
        """Extract search features and run prediction head.

        Args:
            cat_feature: [B, L_z + L_s, C] - Combined template+search features
            gt_score_map: Optional GT score map

        Returns:
            dict with pred_boxes, score_map, size_map, offset_map
        """
        # Extract search region features (last feat_len_s tokens)
        enc_opt = cat_feature[:, -self.feat_len_s:]

        # Reshape to spatial format
        opt_feat = enc_opt.permute(0, 2, 1).reshape(
            -1, self.embed_dim, self.feat_sz, self.feat_sz
        )

        # Run head
        score_map, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)

        return {
            'pred_boxes': bbox,
            'score_map': score_map,
            'size_map': size_map,
            'offset_map': offset_map,
        }

    def set_online(self, rgb_template, sonar_template):
        """Cache template features for online tracking.

        Args:
            rgb_template: [1, 3, H_t, W_t]
            sonar_template: [1, 1, H_t, W_t]
        """
        # Store template images for later use
        self._rgb_template = rgb_template
        self._sonar_template = sonar_template

    def track(self, rgb_search, sonar_search):
        """Online tracking with cached template.

        Args:
            rgb_search: [1, 3, H_s, W_s]
            sonar_search: [1, 1, H_s, W_s]

        Returns:
            dict with rgb_boxes, sonar_boxes, etc.
        """
        with torch.no_grad():
            return self.forward(
                self._rgb_template, rgb_search,
                self._sonar_template, sonar_search
            )


def build_aquasot(cfg, training=True):
    """Build AquaSOT model from config.

    Args:
        cfg: Configuration object
        training: Whether to load pretrained weights

    Returns:
        AquaSOT model
    """
    # Build backbones
    pretrained = cfg.MODEL.PRETRAIN_FILE if training else None
    rgb_backbone = build_vit_rgb(cfg, pretrained=pretrained)
    sonar_backbone = build_vit_sonar(cfg, pretrained=pretrained)

    # Build fusion
    fusion = build_fusion(cfg)

    # Build head
    hidden_dim = cfg.MODEL.BACKBONE.RGB.EMBED_DIM
    box_head = build_box_head(cfg, hidden_dim)

    # Create model
    model = AquaSOT(
        rgb_backbone=rgb_backbone,
        sonar_backbone=sonar_backbone,
        fusion=fusion,
        box_head=box_head,
        head_type=cfg.MODEL.HEAD.TYPE
    )

    return model
