"""
Sonar Encoder with Log-Polar Preprocessing

This module implements a specialized encoder for Forward-Looking Sonar (FLS)
images. It integrates differentiable log-polar transformation and extracts
features suitable for tracking and weak 3D depth estimation.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Literal
import timm

from aquasot3d.core import BACKBONE_REGISTRY, BaseModule
from aquasot3d.dataloader.transforms import LogPolarTransform, SonarGeometryHandler


@BACKBONE_REGISTRY.register("Sonar_Encoder")
class SonarEncoder(BaseModule):
    """
    Sonar feature encoder with log-polar preprocessing.

    Processes fan-shaped sonar images through differentiable log-polar
    transformation followed by a lightweight encoder (ViT or CNN).

    Args:
        encoder_type: Type of encoder ('vit' or 'conv')
        model_name: timm model name for encoder
        pretrained: Whether to use pretrained encoder
        use_logpolar: Whether to apply log-polar transformation
        learnable_logpolar: Whether log-polar parameters are learnable
        img_size: Expected sonar image size (height, width)
        embed_dim: Embedding dimension (default: 384 for lightweight models)
        depth: Number of encoder layers (default: 6 for lightweight)
        num_heads: Number of attention heads (for ViT, default: 6)
        sonar_fov_angle: Sonar field-of-view angle in degrees (default: 130)
        max_range_pixels: Maximum effective range in pixels
        extract_radial_profile: Whether to extract radial activation profile

    Returns:
        Dictionary containing:
            - 'features': Encoded sonar features
            - 'logpolar_features': Features in log-polar domain (optional)
            - 'radial_profile': Radial activation profile for depth proxy (optional)
            - 'geometry_info': Sonar geometry information

    Example:
        >>> encoder = SonarEncoder(encoder_type='vit', model_name='vit_tiny_patch16_224')
        >>> sonar_template = torch.randn(4, 1, 682, 376)  # Grayscale sonar
        >>> sonar_search = torch.randn(4, 1, 682, 376)
        >>> outputs = encoder(sonar_template, sonar_search)
        >>> print(outputs['features'].shape)
    """

    def __init__(
        self,
        encoder_type: Literal['vit', 'conv'] = 'vit',
        model_name: str = 'vit_tiny_patch16_224',
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        use_logpolar: bool = True,
        learnable_logpolar: bool = True,
        img_size: Tuple[int, int] = (376, 682),
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        sonar_fov_angle: float = 130.0,
        max_range_pixels: Optional[int] = None,
        extract_radial_profile: bool = True,
        **kwargs
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.model_name = model_name
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.use_logpolar = use_logpolar
        self.learnable_logpolar = learnable_logpolar
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.sonar_fov_angle = sonar_fov_angle
        self.extract_radial_profile = extract_radial_profile

        if max_range_pixels is None:
            max_range_pixels = int(img_size[0] * 0.95)
        self.max_range_pixels = max_range_pixels

        # Sonar geometry handler
        self.geometry_handler = SonarGeometryHandler(
            sonar_height=img_size[0],
            sonar_width=img_size[1],
            max_range=max_range_pixels,
            fov_angle=sonar_fov_angle,
        )

        # Log-polar transformation (differentiable)
        # CRITICAL FIX (Audit 2025111407): Forward custom FOV/max_range/sensor_pos to LogPolarTransform
        # This ensures masks/radial weights and log-polar sampling use consistent coordinate systems
        if use_logpolar:
            self.logpolar_transform = LogPolarTransform(
                sonar_height=img_size[0],
                sonar_width=img_size[1],
                output_height=224,  # Match ViT input size
                output_width=224,   # Match ViT input size
                fov_angle=sonar_fov_angle,  # Forward custom FOV
                max_range=max_range_pixels,  # Forward custom max range
                sensor_pos=self.geometry_handler.sensor_pos,  # Forward sensor position
                learnable_scales=learnable_logpolar,
                init_alpha=1.0,
                init_beta=1.0,
            )
        else:
            self.logpolar_transform = nn.Identity()

        # Input channel adapter (sonar is typically grayscale)
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)

        # Encoder network
        if encoder_type == 'vit':
            self.encoder = self._create_vit_encoder(model_name, pretrained)
            # Get actual embed_dim from loaded model (timm models may differ from config)
            if hasattr(self.encoder, 'embed_dim'):
                self.embed_dim = self.encoder.embed_dim
            elif hasattr(self.encoder, 'num_features'):
                self.embed_dim = self.encoder.num_features
        elif encoder_type == 'conv':
            self.encoder = self._create_conv_encoder(model_name, pretrained)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Radial profile extraction head (for depth proxy)
        if extract_radial_profile:
            self.radial_head = RadialProfileExtractor(
                feature_dim=self.embed_dim,  # Use actual model embed_dim
                num_bins=50,
            )

        # Load pretrained checkpoint if specified (Stage 2 Sonar-only weights)
        if pretrained_path is not None:
            self._load_pretrained_checkpoint(pretrained_path)

        self.logger.info(
            f"Initialized SonarEncoder: encoder_type={encoder_type}, "
            f"model={model_name}, use_logpolar={use_logpolar}, "
            f"learnable_logpolar={learnable_logpolar}, "
            f"pretrained_path={pretrained_path}"
        )

    def _create_vit_encoder(self, model_name: str, pretrained: bool) -> nn.Module:
        """Create lightweight ViT encoder for sonar."""
        try:
            encoder = timm.create_model(
                model_name,
                pretrained=pretrained,
                img_size=224,  # Will be adapted
                num_classes=0,
                global_pool='',
            )
            return encoder
        except Exception as e:
            self.logger.warning(f"Failed to load {model_name}, using default tiny ViT: {e}")
            # Fallback to a simple ViT
            return self._create_default_vit()

    def _create_conv_encoder(self, model_name: str, pretrained: bool) -> nn.Module:
        """Create lightweight CNN encoder for sonar."""
        try:
            encoder = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                num_classes=0,
            )
            return encoder
        except Exception as e:
            self.logger.warning(f"Failed to load {model_name}, using default CNN: {e}")
            return self._create_default_cnn()

    def _create_default_vit(self) -> nn.Module:
        """Create default lightweight ViT as fallback."""
        from timm.models.vision_transformer import VisionTransformer
        return VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
        )

    def _create_default_cnn(self) -> nn.Module:
        """Create default lightweight CNN as fallback."""
        return LightweightSonarCNN(
            in_channels=3,
            embed_dim=self.embed_dim,
        )

    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        """
        Load pretrained sonar encoder weights from Stage 2 checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pth)
        """
        import os
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter Sonar encoder keys
            # Expected format: 'backbone.sonar_encoder.encoder.*'
            sonar_encoder_keys = {}
            for key, value in state_dict.items():
                if 'sonar_encoder.encoder.' in key:
                    # Remove prefix to match local encoder module
                    new_key = key.split('sonar_encoder.encoder.')[-1]
                    sonar_encoder_keys[new_key] = value
                elif 'sonar_encoder.input_adapter.' in key:
                    # Also load input adapter weights
                    new_key = key.split('sonar_encoder.')[-1]
                    sonar_encoder_keys[new_key] = value
                elif 'sonar_encoder.logpolar_transform.' in key:
                    # Also load logpolar transform weights if learnable
                    new_key = key.split('sonar_encoder.')[-1]
                    sonar_encoder_keys[new_key] = value
                elif key.startswith('encoder.'):
                    # Direct encoder key (if saved without model wrapper)
                    new_key = key[8:]  # Remove 'encoder.' prefix
                    sonar_encoder_keys[new_key] = value

            if not sonar_encoder_keys:
                self.logger.warning(
                    f"No Sonar encoder keys found in checkpoint {checkpoint_path}. "
                    f"Available keys: {list(state_dict.keys())[:5]}..."
                )
                return

            # Load encoder state dict
            encoder_keys = {k.replace('input_adapter.', '').replace('logpolar_transform.', ''): v
                           for k, v in sonar_encoder_keys.items()
                           if not k.startswith('input_adapter.') and not k.startswith('logpolar_transform.')}

            if encoder_keys:
                missing_keys, unexpected_keys = self.encoder.load_state_dict(
                    encoder_keys, strict=False
                )
                if missing_keys:
                    self.logger.debug(f"Missing encoder keys: {missing_keys[:5]}...")
                if unexpected_keys:
                    self.logger.debug(f"Unexpected encoder keys: {unexpected_keys[:5]}...")

            # Load input adapter and logpolar transform
            adapter_keys = {k.replace('input_adapter.', ''): v
                           for k, v in sonar_encoder_keys.items()
                           if k.startswith('input_adapter.')}
            if adapter_keys:
                self.input_adapter.load_state_dict(adapter_keys, strict=False)

            logpolar_keys = {k.replace('logpolar_transform.', ''): v
                            for k, v in sonar_encoder_keys.items()
                            if k.startswith('logpolar_transform.')}
            if logpolar_keys and self.use_logpolar and self.learnable_logpolar:
                self.logpolar_transform.load_state_dict(logpolar_keys, strict=False)

            self.logger.info(
                f"Loaded pretrained Sonar encoder from {checkpoint_path}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to load pretrained checkpoint {checkpoint_path}: {e}"
            )
            raise

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder network.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Encoded features
        """
        if self.encoder_type == 'vit':
            # ViT forward
            if hasattr(self.encoder, 'patch_embed'):
                x = self.encoder.patch_embed(x)
            if hasattr(self.encoder, 'cls_token'):
                B = x.shape[0]
                cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            if hasattr(self.encoder, 'pos_embed'):
                x = x + self.encoder.pos_embed
            if hasattr(self.encoder, 'blocks'):
                for blk in self.encoder.blocks:
                    x = blk(x)
            if hasattr(self.encoder, 'norm'):
                x = self.encoder.norm(x)
            return x
        elif self.encoder_type == 'conv':
            # CNN forward
            if isinstance(self.encoder, nn.ModuleList):
                # features_only model returns list
                features = self.encoder(x)
                return features[-1]  # Use final stage
            else:
                return self.encoder(x)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        return_separate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for template-search paradigm.

        Args:
            template: Template sonar frame (B, 1, H, W)
            search: Search region sonar frame (B, 1, H, W)
            return_separate: If True, process separately

        Returns:
            Dictionary containing encoded features and geometry info
        """
        B = template.shape[0]

        # Apply geometry handler (fan mask and radial weights)
        template_geo = self.geometry_handler(template)
        search_geo = self.geometry_handler(search)
        template_masked = template_geo['masked']
        search_masked = search_geo['masked']

        # Apply log-polar transformation if enabled
        if self.use_logpolar:
            template_lp = self.logpolar_transform(template_masked)
            search_lp = self.logpolar_transform(search_masked)
        else:
            # Resize to match encoder input size (224x224 for ViT)
            import torch.nn.functional as F
            template_lp = F.interpolate(template_masked, size=(224, 224), mode='bilinear', align_corners=False)
            search_lp = F.interpolate(search_masked, size=(224, 224), mode='bilinear', align_corners=False)

        # Adapt input channels (1 -> 3)
        template_rgb = self.input_adapter(template_lp)
        search_rgb = self.input_adapter(search_lp)

        if return_separate:
            # Process separately
            template_feat = self.forward_encoder(template_rgb)
            search_feat = self.forward_encoder(search_rgb)

            outputs = {
                'template_features': template_feat,
                'search_features': search_feat,
            }
        else:
            # Process together
            combined = torch.cat([template_rgb, search_rgb], dim=0)
            combined_feat = self.forward_encoder(combined)

            template_feat = combined_feat[:B]
            search_feat = combined_feat[B:]

            outputs = {
                'features': combined_feat,
                'template_features': template_feat,
                'search_features': search_feat,
            }

        # Extract radial profile for depth proxy
        if self.extract_radial_profile:
            template_radial = self.geometry_handler.get_radial_profile(
                template_masked, num_bins=50
            )
            search_radial = self.geometry_handler.get_radial_profile(
                search_masked, num_bins=50
            )

            outputs['template_radial_profile'] = template_radial
            outputs['search_radial_profile'] = search_radial

        # Add geometry information
        outputs['geometry_info'] = {
            'template_weights': template_geo['weighted'],
            'search_weights': search_geo['weighted'],
            'fov_angle': self.sonar_fov_angle,
            'max_range': self.max_range_pixels,
        }

        return outputs

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.embed_dim


class LightweightSonarCNN(nn.Module):
    """
    Lightweight CNN encoder for sonar images.

    Simple convolutional architecture as fallback when timm models are unavailable.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, embed_dim, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """Create a layer with residual blocks."""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


class RadialProfileExtractor(nn.Module):
    """
    Extract radial activation profile for depth proxy estimation.

    Processes sonar features to extract a 1D profile along the radial dimension,
    which serves as a proxy for depth/distance information.
    """

    def __init__(self, feature_dim: int, num_bins: int = 50):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bins = num_bins

        # Learnable projection for radial profile
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_bins),
        )

    def forward(self, features: torch.Tensor, radial_bins: torch.Tensor) -> torch.Tensor:
        """
        Extract radial profile from features.

        Args:
            features: Encoded features (B, N, C) or (B, C, H, W)
            radial_bins: Radial bin assignments (B, num_bins)

        Returns:
            Radial profile (B, num_bins)
        """
        # Global average pooling if spatial features
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])  # (B, C)
        elif features.dim() == 3:
            features = features.mean(dim=1)  # (B, C)

        # Project to radial profile
        profile = self.projection(features)  # (B, num_bins)

        return profile


def create_sonar_encoder(
    encoder_type: str = 'vit',
    model_name: str = 'vit_tiny_patch16_224',
    **kwargs
) -> SonarEncoder:
    """
    Factory function to create Sonar encoder.

    Args:
        encoder_type: Type of encoder ('vit' or 'conv')
        model_name: timm model name
        **kwargs: Additional arguments for SonarEncoder

    Returns:
        SonarEncoder instance

    Example:
        >>> encoder = create_sonar_encoder('vit', 'vit_tiny_patch16_224')
    """
    return SonarEncoder(encoder_type=encoder_type, model_name=model_name, **kwargs)
