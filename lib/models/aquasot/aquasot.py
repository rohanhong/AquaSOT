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
        Fused RGB -> RGB CenterHead -> RGB Bbox
        Fused Sonar -> Sonar CenterHead -> Sonar Bbox

    Note:
        RGB and Sonar use separate box heads to learn modality-specific
        prediction patterns. This is critical because:
        1. Sonar FOV > RGB FOV, so same target moves slower in Sonar
        2. Shared head causes Sonar predictions to drift toward RGB patterns
    """

    def __init__(self, rgb_backbone, sonar_backbone, fusion,
                 rgb_box_head, sonar_box_head, head_type="CENTER"):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.sonar_backbone = sonar_backbone
        self.fusion = fusion
        self.rgb_box_head = rgb_box_head
        self.sonar_box_head = sonar_box_head
        self.head_type = head_type

        # Feature dimensions
        self.embed_dim = rgb_backbone.embed_dim
        self.feat_sz = rgb_box_head.feat_sz
        self.feat_len_s = self.feat_sz ** 2  # Search region feature length

    def forward(self, rgb_template, rgb_search, sonar_template, sonar_search,
                rgb_online_template=None, sonar_online_template=None,
                gt_score_map_rgb=None, gt_score_map_sonar=None):
        """
        Args:
            rgb_template: [B, 3, H_t, W_t] - RGB first-frame template
            rgb_search: [B, 3, H_s, W_s] - RGB search region
            sonar_template: [B, 1, H_t, W_t] - Sonar first-frame template
            sonar_search: [B, 1, H_s, W_s] - Sonar search region
            rgb_online_template: [B, N, 3, H_t, W_t] or [B*N, 3, H_t, W_t] - RGB online templates (optional)
            sonar_online_template: [B, N, 1, H_t, W_t] or [B*N, 1, H_t, W_t] - Sonar online templates (optional)
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
        # Detect modality dropout (input is all zeros) to bypass invalid cross-attention
        # This enables single-modal inference without retraining
        # Threshold 1e-4 is conservative to avoid float precision edge cases
        rgb_valid = rgb_search.abs().sum() > 1e-4
        sonar_valid = sonar_search.abs().sum() > 1e-4

        # Extract backbone features (with optional online templates)
        rgb_feat, rgb_aux = self.rgb_backbone(rgb_template, rgb_search, rgb_online_template)
        sonar_feat, sonar_aux = self.sonar_backbone(sonar_template, sonar_search, sonar_online_template)

        # Cross-modal fusion (skip cross-attention for dropped modality)
        fused_rgb, fused_sonar = self.fusion(rgb_feat, sonar_feat,
                                              rgb_valid=rgb_valid, sonar_valid=sonar_valid)

        # Get search region features and predict boxes (separate heads)
        rgb_out = self.forward_head(fused_rgb, self.rgb_box_head, gt_score_map_rgb)
        sonar_out = self.forward_head(fused_sonar, self.sonar_box_head, gt_score_map_sonar)

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

    def forward_head(self, cat_feature, box_head, gt_score_map=None):
        """Extract search features and run prediction head.

        Args:
            cat_feature: [B, L_z + L_s, C] - Combined template+search features
            box_head: CenterPredictor head to use (rgb_box_head or sonar_box_head)
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

        # Run modality-specific head
        score_map, bbox, size_map, offset_map = box_head(opt_feat, gt_score_map)

        return {
            'pred_boxes': bbox,
            'score_map': score_map,
            'size_map': size_map,
            'offset_map': offset_map,
        }

    def set_online(self, rgb_template, sonar_template,
                   rgb_online_template=None, sonar_online_template=None):
        """Cache template features for online tracking.

        Args:
            rgb_template: [1, 3, H_t, W_t] - First-frame RGB template
            sonar_template: [1, 1, H_t, W_t] - First-frame Sonar template
            rgb_online_template: [1, N, 3, H_t, W_t] or [N, 3, H_t, W_t] - Online RGB templates (optional)
            sonar_online_template: [1, N, 1, H_t, W_t] or [N, 1, H_t, W_t] - Online Sonar templates (optional)
        """
        # Store template images for later use
        self._rgb_template = rgb_template
        self._sonar_template = sonar_template
        self._rgb_online_template = rgb_online_template
        self._sonar_online_template = sonar_online_template

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
                self._sonar_template, sonar_search,
                self._rgb_online_template, self._sonar_online_template
            )


def build_aquasot(cfg, training=True):
    """Build AquaSOT model from config.

    Args:
        cfg: Configuration object
        training: Whether building for training (affects pretrained weights loading
                  and fusion weights: training=1.0, inference=0.3)

    Returns:
        AquaSOT model
    """
    # Build backbones
    pretrained = cfg.MODEL.PRETRAIN_FILE if training else None
    rgb_backbone = build_vit_rgb(cfg, pretrained=pretrained)
    sonar_backbone = build_vit_sonar(cfg, pretrained=pretrained)

    # Build fusion (training uses 1.0 weights, inference uses 0.3 weights)
    fusion = build_fusion(cfg, training=training)

    # Build separate heads for RGB and Sonar
    # This is critical because Sonar FOV > RGB FOV, causing different
    # motion patterns. Shared head causes Sonar to drift toward RGB patterns.
    hidden_dim = cfg.MODEL.BACKBONE.RGB.EMBED_DIM
    rgb_box_head = build_box_head(cfg, hidden_dim)
    sonar_box_head = build_box_head(cfg, hidden_dim)

    # Create model
    model = AquaSOT(
        rgb_backbone=rgb_backbone,
        sonar_backbone=sonar_backbone,
        fusion=fusion,
        rgb_box_head=rgb_box_head,
        sonar_box_head=sonar_box_head,
        head_type=cfg.MODEL.HEAD.TYPE
    )

    return model
