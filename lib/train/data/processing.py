"""
Data processing for AquaSOT-3D training.

Based on OSTrack's STARKProcessing, extended for dual-modal RGB-Sonar data.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from . import processing_utils as prutils


def stack_tensors(x):
    """Stack list of tensors into a single tensor."""
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """Base class for data processing."""

    def __init__(self, transform=transforms.ToTensor(), joint_transform=None):
        """
        Args:
            transform: Transforms to apply to images
            joint_transform: Transforms to apply jointly to template and search
        """
        self.transform = transform
        self.joint_transform = joint_transform

    def __call__(self, data):
        raise NotImplementedError


class MultiModalProcessing(BaseProcessing):
    """Processing for dual-modal RGB-Sonar tracking data.

    Handles:
    1. Jittering bounding boxes for data augmentation
    2. Cropping template and search regions
    3. Applying transforms to both RGB and Sonar modalities
    4. Normalizing coordinates to [0, 1]
    """

    def __init__(
        self,
        search_area_factor,
        output_sz,
        center_jitter_factor,
        scale_jitter_factor,
        mode='sequence',
        rgb_transform=None,
        sonar_transform=None,
        joint_transform=None,
        settings=None,
    ):
        """
        Args:
            search_area_factor: dict with 'template' and 'search' keys
            output_sz: dict with 'template' and 'search' keys
            center_jitter_factor: dict with 'template' and 'search' keys
            scale_jitter_factor: dict with 'template' and 'search' keys
            mode: 'pair' or 'sequence'
            rgb_transform: Transforms for RGB images
            sonar_transform: Transforms for sonar images
            joint_transform: Joint transforms for template/search
            settings: Training settings
        """
        super().__init__(joint_transform=joint_transform)

        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.rgb_transform = rgb_transform
        self.sonar_transform = sonar_transform
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """Apply jitter to bounding box.

        Args:
            box: Input box [x, y, w, h]
            mode: 'template' or 'search'

        Returns:
            Jittered box
        """
        jittered_size = box[2:4] * torch.exp(
            torch.randn(2) * self.scale_jitter_factor[mode]
        )
        max_offset = jittered_size.prod().sqrt() * torch.tensor(
            self.center_jitter_factor[mode]
        ).float()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data):
        """Process the data.

        Args:
            data: dict with:
                - rgb_template_images: List of RGB template images
                - rgb_search_images: List of RGB search images
                - sonar_template_images: List of sonar template images
                - sonar_search_images: List of sonar search images
                - rgb_template_anno: RGB template bboxes
                - rgb_search_anno: RGB search bboxes
                - sonar_template_anno: Sonar template bboxes
                - sonar_search_anno: Sonar search bboxes

        Returns:
            Processed data dict with cropped/normalized images and boxes
        """
        # Process each modality and role (template/search)
        for modality in ['rgb', 'sonar']:
            for role in ['template', 'search']:
                images_key = f'{modality}_{role}_images'
                anno_key = f'{modality}_{role}_anno'

                if images_key not in data or anno_key not in data:
                    continue

                images = data[images_key]
                annos = data[anno_key]

                # Ensure tensor format for annotations
                annos = [torch.tensor(a) if not isinstance(a, torch.Tensor) else a
                         for a in annos]

                # Check bbox validity
                w = torch.stack(annos)[:, 2]
                h = torch.stack(annos)[:, 3]
                crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[role])

                if (crop_sz < 1).any():
                    data['valid'] = False
                    return data

                # Apply jitter to get extraction boxes
                jittered_anno = [self._get_jittered_box(a, role) for a in annos]

                # Crop and resize
                factor = self.search_area_factor[role]
                out_sz = self.output_sz[role]

                crops, boxes, att_mask, _ = prutils.jittered_center_crop(
                    images, jittered_anno, annos, factor, out_sz
                )

                # Apply transforms
                if modality == 'rgb' and self.rgb_transform is not None:
                    crops = [self.rgb_transform(c) for c in crops]
                elif modality == 'sonar' and self.sonar_transform is not None:
                    crops = [self.sonar_transform(c) for c in crops]
                else:
                    # Default: convert to tensor
                    if modality == 'rgb':
                        crops = [torch.from_numpy(c).permute(2, 0, 1).float() / 255.0
                                 for c in crops]
                    else:
                        # Sonar: single channel
                        crops = [torch.from_numpy(c).unsqueeze(0).float() / 255.0
                                 if isinstance(c, np.ndarray) else c.unsqueeze(0).float() / 255.0
                                 for c in crops]

                # Check attention mask validity
                for mask in att_mask:
                    if mask.all():
                        data['valid'] = False
                        return data

                    # Check downsampled mask
                    feat_size = out_sz // 16
                    mask_tensor = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
                    mask_down = F.interpolate(
                        mask_tensor[None, None].float(),
                        size=feat_size
                    ).to(torch.bool)[0]
                    if mask_down.all():
                        data['valid'] = False
                        return data

                # Store processed data
                data[images_key] = crops
                data[anno_key] = boxes
                data[f'{modality}_{role}_att'] = [
                    torch.from_numpy(m) if isinstance(m, np.ndarray) else m
                    for m in att_mask
                ]

        data['valid'] = True

        # Stack tensors if in sequence mode
        if self.mode == 'sequence':
            for key in list(data.keys()):
                if isinstance(data[key], list) and len(data[key]) > 0:
                    if isinstance(data[key][0], torch.Tensor):
                        data[key] = torch.stack(data[key])
        else:
            # Pair mode: take first element
            for key in list(data.keys()):
                if isinstance(data[key], list) and len(data[key]) > 0:
                    data[key] = data[key][0]

        return data


class DualModalProcessing(MultiModalProcessing):
    """Simplified processing that outputs in actor-expected format.

    Converts the processed data to the format expected by AquaSOT3DActor:
        - rgb_template: [N, B, C, H, W]
        - rgb_search: [N, B, C, H, W]
        - sonar_template: [N, B, 1, H, W]
        - sonar_search: [N, B, 1, H, W]
        - rgb_anno: [N, B, 4]
        - sonar_anno: [N, B, 4]
    """

    def __call__(self, data):
        """Process and reformat data for actor."""
        # First apply base processing
        data = super().__call__(data)

        if not data.get('valid', True):
            return data

        # Reformat to actor-expected keys
        result = {
            'valid': data['valid'],
            'dataset': data.get('dataset', ''),
        }

        # Map to actor keys
        key_mapping = {
            'rgb_template_images': 'rgb_template',
            'rgb_search_images': 'rgb_search',
            'sonar_template_images': 'sonar_template',
            'sonar_search_images': 'sonar_search',
            'rgb_search_anno': 'rgb_anno',
            'sonar_search_anno': 'sonar_anno',
        }

        for src_key, dst_key in key_mapping.items():
            if src_key in data:
                val = data[src_key]
                # Ensure correct shape [N, ...] for stacking in collate
                if isinstance(val, torch.Tensor):
                    if val.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                        val = val.unsqueeze(0)
                    elif val.dim() == 1:  # [4] -> [1, 4]
                        val = val.unsqueeze(0)
                result[dst_key] = val

        return result
