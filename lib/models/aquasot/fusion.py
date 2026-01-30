"""
Cross-Modal Fusion module for AquaSOT.

Cross-Modal Attention (CMA) for bidirectional feature fusion between RGB and Sonar.

Note: Despite the class name 'DCMA' (Deformable Cross-Modal Attention), this module
implements standard cross-attention, not deformable attention. The 'deformable' aspect
refers to the flexible cross-modal interaction pattern, not the Deformable-DETR style
sparse sampling mechanism. For true deformable attention with learnable sampling
offsets and reference points, see external/Deformable-DETR/models/ops/modules/.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossModalAttention(nn.Module):
    """Standard cross-modal attention.

    Computes attention between two modality features.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key_value):
        """
        Args:
            query: [B, L_q, C] - Query modality features
            key_value: [B, L_kv, C] - Key-Value modality features

        Returns:
            out: [B, L_q, C] - Attended features
        """
        B, L_q, C = query.shape
        L_kv = key_value.shape[1]

        q = self.q_proj(query).reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L_q, C)
        out = self.out_proj(out)

        return out


class DCMALayer(nn.Module):
    """Deformable Cross-Modal Attention Layer.

    Performs bidirectional cross-attention between RGB and Sonar features.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        # RGB -> Sonar attention
        self.rgb_to_sonar = CrossModalAttention(embed_dim, num_heads, dropout)
        # Sonar -> RGB attention
        self.sonar_to_rgb = CrossModalAttention(embed_dim, num_heads, dropout)

        # FFN
        self.ffn_rgb = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_sonar = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # Norms
        self.norm_rgb1 = nn.LayerNorm(embed_dim)
        self.norm_rgb2 = nn.LayerNorm(embed_dim)
        self.norm_sonar1 = nn.LayerNorm(embed_dim)
        self.norm_sonar2 = nn.LayerNorm(embed_dim)

    def forward(self, rgb_feat, sonar_feat, rgb_fusion_weight=0.3, sonar_fusion_weight=0.3,
                rgb_valid=True, sonar_valid=True):
        """
        Args:
            rgb_feat: [B, L, C] - RGB features
            sonar_feat: [B, L, C] - Sonar features
            rgb_fusion_weight: Weight for RGB cross-modal attention (0=no fusion, 1=full)
            sonar_fusion_weight: Weight for Sonar cross-modal attention (0=no fusion, 1=full)
            rgb_valid: Whether RGB input was valid (not dropped by modality dropout)
            sonar_valid: Whether Sonar input was valid (not dropped by modality dropout)

        Returns:
            rgb_out: [B, L, C] - Fused RGB features
            sonar_out: [B, L, C] - Fused Sonar features
        """
        # Cross-attention with configurable fusion strength
        # IMPORTANT: Compute both attentions using ORIGINAL features first,
        # then apply updates. This preserves exact behavior when both modalities valid.
        #
        # When one modality is dropped (invalid), we use SELF-ATTENTION as a substitute
        # for cross-attention. This preserves the feature distribution that the head
        # learned during training, avoiding the distribution shift that would occur
        # if we simply skipped the attention operation.

        # RGB attention: RGB query attends to Sonar (or self if Sonar invalid)
        # Note: Self-attention for RGB works because RGB features are robust (ImageNet pretrained)
        if sonar_valid:
            rgb_attn = self.rgb_to_sonar(self.norm_rgb1(rgb_feat), sonar_feat)
        else:
            # Sonar dropped: self-attention still works for RGB due to robust features
            rgb_attn = self.rgb_to_sonar(self.norm_rgb1(rgb_feat), rgb_feat)

        # Sonar attention: Sonar query attends to RGB (or cross-module self-attention if RGB invalid)
        # Note: Self-attention using sonar_to_rgb FAILS because its K/V projections are
        # trained for RGB features, producing garbage when applied to Sonar features.
        # SOLUTION: Use rgb_to_sonar's K/V projections (trained for Sonar features).
        if rgb_valid:
            sonar_attn = self.sonar_to_rgb(self.norm_sonar1(sonar_feat), rgb_feat)
        else:
            # RGB dropped: use rgb_to_sonar module for Sonar self-attention
            # Rationale: rgb_to_sonar has K/V projections trained for SONAR features,
            # so applying them to Sonar features produces meaningful output.
            # The Q projection mismatch (trained for RGB) is tolerable - it only
            # affects attention pattern, while V projection ensures correct output distribution.
            sonar_attn = self.rgb_to_sonar(self.norm_sonar1(sonar_feat), sonar_feat)

        # Apply fusion updates
        rgb_feat = rgb_feat + rgb_fusion_weight * rgb_attn
        sonar_feat = sonar_feat + sonar_fusion_weight * sonar_attn

        # FFN (always applied to refine features)
        rgb_out = rgb_feat + self.ffn_rgb(self.norm_rgb2(rgb_feat))
        sonar_out = sonar_feat + self.ffn_sonar(self.norm_sonar2(sonar_feat))

        return rgb_out, sonar_out


class DCMA(nn.Module):
    """Cross-Modal Attention module.

    Multi-layer bidirectional cross-attention between RGB and Sonar modalities.
    Uses standard scaled dot-product attention (not deformable attention).
    """

    def __init__(self, embed_dim=768, num_heads=8, num_layers=2, dropout=0.1,
                 rgb_fusion_weight=0.3, sonar_fusion_weight=0.3):
        super().__init__()

        self.layers = nn.ModuleList([
            DCMALayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        # Configurable fusion weights (0=no cross-modal influence, 1=full fusion)
        self.rgb_fusion_weight = rgb_fusion_weight
        self.sonar_fusion_weight = sonar_fusion_weight

    def forward(self, rgb_feat, sonar_feat, rgb_valid=True, sonar_valid=True):
        """
        Args:
            rgb_feat: [B, L, C] - RGB features from backbone
            sonar_feat: [B, L, C] - Sonar features from backbone
            rgb_valid: Whether RGB input was valid (not dropped by modality dropout)
            sonar_valid: Whether Sonar input was valid (not dropped by modality dropout)

        Returns:
            rgb_out: [B, L, C] - Fused RGB features
            sonar_out: [B, L, C] - Fused Sonar features
        """
        for layer in self.layers:
            rgb_feat, sonar_feat = layer(
                rgb_feat, sonar_feat,
                rgb_fusion_weight=self.rgb_fusion_weight,
                sonar_fusion_weight=self.sonar_fusion_weight,
                rgb_valid=rgb_valid,
                sonar_valid=sonar_valid
            )

        return rgb_feat, sonar_feat


class IdentityFusion(nn.Module):
    """Identity fusion (no cross-modal interaction).

    Note: Currently unused. The no_fusion ablation mode is implemented via
    DCMA with fusion weights set to 0.0 (--eval_mode no_fusion), which
    preserves the FFN layers for feature refinement. This class is retained
    as an extension point for future experiments.
    """

    def __init__(self, embed_dim=768, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rgb_feat, sonar_feat, rgb_valid=True, sonar_valid=True):
        # rgb_valid/sonar_valid ignored - no cross-modal interaction anyway
        return rgb_feat, sonar_feat


def build_fusion(cfg):
    """Build fusion module from config."""
    fusion_type = cfg.MODEL.FUSION.TYPE.lower()

    if fusion_type == 'dcma':
        # Get fusion weights from config (default to 0.3 for weak fusion)
        rgb_fusion_weight = getattr(cfg.MODEL.FUSION, 'RGB_FUSION_WEIGHT', 0.3)
        sonar_fusion_weight = getattr(cfg.MODEL.FUSION, 'SONAR_FUSION_WEIGHT', 0.3)
        return DCMA(
            embed_dim=cfg.MODEL.FUSION.EMBED_DIM,
            num_heads=cfg.MODEL.FUSION.NUM_HEADS,
            num_layers=cfg.MODEL.FUSION.NUM_LAYERS,
            dropout=cfg.MODEL.FUSION.DROPOUT,
            rgb_fusion_weight=rgb_fusion_weight,
            sonar_fusion_weight=sonar_fusion_weight
        )
    elif fusion_type == 'identity':
        return IdentityFusion(embed_dim=cfg.MODEL.FUSION.EMBED_DIM)
    else:
        raise ValueError(f"Fusion type {fusion_type} not supported")
