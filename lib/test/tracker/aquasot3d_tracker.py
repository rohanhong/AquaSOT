"""
AquaSOT-3D tracker wrapper for evaluation.

Based on OSTrack's tracker wrapper pattern.
"""

import os
import torch
import numpy as np
import cv2

from .base_tracker import Tracker
from lib.models.aquasot3d import build_aquasot3d
from lib.config.aquasot3d.config import cfg, update_config_from_file
from lib.test.utils.hann import hann2d


class AquaSOT3DTracker(Tracker):
    """AquaSOT-3D tracker for evaluation.

    Wraps the AquaSOT-3D model for running on benchmarks.

    Args:
        params: Tracker parameters (config path, checkpoint path, etc.)
        dataset_name: Name of dataset being evaluated
    """

    def __init__(self, params, dataset_name=None):
        super().__init__(name='AquaSOT3D', is_deterministic=True)
        self.params = params
        self.dataset_name = dataset_name

        # Load config
        cfg_file = params.get('cfg_file', None)
        if cfg_file is not None and os.path.exists(cfg_file):
            update_config_from_file(cfg_file)

        # Override config with params
        if 'search_area_factor' in params:
            cfg.DATA.SEARCH.FACTOR = params['search_area_factor']
        if 'template_area_factor' in params:
            cfg.DATA.TEMPLATE.FACTOR = params['template_area_factor']
        if 'search_size' in params:
            cfg.DATA.SEARCH.SIZE = params['search_size']
        if 'template_size' in params:
            cfg.DATA.TEMPLATE.SIZE = params['template_size']

        # Build model
        self.network = build_aquasot3d(cfg, training=False)

        # Load checkpoint
        checkpoint = params.get('checkpoint', None)
        if checkpoint is not None and os.path.exists(checkpoint):
            self._load_checkpoint(checkpoint)

        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = self.network.to(self.device)
        self.network.eval()

        # Initialize state
        self.state = None
        self.frame_id = 0

        # Preprocessing params
        self.template_factor = cfg.DATA.TEMPLATE.FACTOR
        self.search_factor = cfg.DATA.SEARCH.FACTOR
        self.template_size = cfg.DATA.TEMPLATE.SIZE
        self.search_size = cfg.DATA.SEARCH.SIZE

        # Mean/std for RGB normalization
        self.mean_rgb = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.std_rgb = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

        # Mean/std for Sonar normalization
        self.mean_sonar = 0.5
        self.std_sonar = 0.5

        # Hann window for motion constraint (reduces tracking drift)
        stride = cfg.MODEL.BACKBONE.RGB.STRIDE
        self.feat_sz = self.search_size // stride
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True
        ).to(self.device)

    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP training)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        missing, unexpected = self.network.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[AquaSOT3DTracker] Missing keys: {missing}')
        if unexpected:
            print(f'[AquaSOT3DTracker] Unexpected keys: {unexpected}')

    def init(self, image, box):
        """Initialize tracker with first frame.

        Args:
            image: First frame (dict with 'rgb' and 'sonar' keys) or RGB image
            box: Initial bounding box [x, y, w, h] (can be dict with modality keys)
        """
        # Handle multi-modal or single-modal input
        if isinstance(image, dict):
            rgb_image = image.get('rgb')
            sonar_image = image.get('sonar')
        else:
            rgb_image = image
            sonar_image = None

        # Handle box format
        if isinstance(box, dict):
            rgb_box = np.array(box.get('rgb', box.get('bbox', [0, 0, 1, 1])))
            sonar_box = np.array(box.get('sonar', rgb_box))
        else:
            rgb_box = np.array(box)
            sonar_box = rgb_box.copy()

        # Convert [x, y, w, h] to [cx, cy, w, h]
        rgb_box_cxcywh = self._xywh_to_cxcywh(rgb_box)
        sonar_box_cxcywh = self._xywh_to_cxcywh(sonar_box)

        # Crop and preprocess template
        rgb_template, _ = self._sample_target(
            rgb_image, rgb_box_cxcywh, self.template_factor, self.template_size
        )
        rgb_template = self._preprocess_rgb(rgb_template)

        if sonar_image is not None:
            sonar_template, _ = self._sample_target(
                sonar_image, sonar_box_cxcywh, self.template_factor, self.template_size
            )
            sonar_template = self._preprocess_sonar(sonar_template)
        else:
            # Create dummy sonar template
            sonar_template = torch.zeros(1, 1, self.template_size, self.template_size)

        # Move to device
        rgb_template = rgb_template.to(self.device)
        sonar_template = sonar_template.to(self.device)

        # Set template in model
        self.network.set_online(rgb_template, sonar_template)

        # Save state
        self.state = {
            'rgb_box': rgb_box_cxcywh,
            'sonar_box': sonar_box_cxcywh,
            'rgb_image_shape': rgb_image.shape[:2] if rgb_image is not None else None,
            'sonar_image_shape': sonar_image.shape[:2] if sonar_image is not None else None,
        }
        self.frame_id = 0

    def update(self, image):
        """Track target in current frame.

        Args:
            image: Current frame (dict with 'rgb' and 'sonar' keys) or RGB image

        Returns:
            box: Predicted bounding box [x, y, w, h]
        """
        self.frame_id += 1

        # Handle multi-modal or single-modal input
        if isinstance(image, dict):
            rgb_image = image.get('rgb')
            sonar_image = image.get('sonar')
        else:
            rgb_image = image
            sonar_image = None

        # Crop and preprocess search region
        rgb_search, rgb_resize_factor = self._sample_target(
            rgb_image, self.state['rgb_box'], self.search_factor, self.search_size
        )
        rgb_search = self._preprocess_rgb(rgb_search)

        if sonar_image is not None:
            sonar_search, sonar_resize_factor = self._sample_target(
                sonar_image, self.state['sonar_box'], self.search_factor, self.search_size
            )
            sonar_search = self._preprocess_sonar(sonar_search)
        else:
            sonar_search = torch.zeros(1, 1, self.search_size, self.search_size)
            sonar_resize_factor = rgb_resize_factor

        # Move to device
        rgb_search = rgb_search.to(self.device)
        sonar_search = sonar_search.to(self.device)

        # Run tracking
        with torch.no_grad():
            out = self.network.track(rgb_search, sonar_search)

        # Apply Hann window to score map for motion constraint
        pred_score_map = out['rgb_score_map']
        response = self.output_window * pred_score_map

        # Recalculate bbox using windowed response
        pred_box = self.network.box_head.cal_bbox(
            response, out['rgb_size_map'], out['rgb_offset_map']
        )
        pred_box = pred_box.cpu().numpy()[0]  # [cx, cy, w, h] normalized

        # Map back to original image coordinates
        pred_box = self._map_box_back(
            pred_box, self.state['rgb_box'], rgb_resize_factor, self.search_size
        )

        # Update state
        self.state['rgb_box'] = pred_box.copy()

        # Also update sonar state if we have sonar output
        if 'sonar_boxes' in out and sonar_image is not None:
            sonar_pred = out['sonar_boxes'].cpu().numpy()[0]
            sonar_pred = self._map_box_back(
                sonar_pred, self.state['sonar_box'], sonar_resize_factor, self.search_size
            )
            self.state['sonar_box'] = sonar_pred

        # Convert to [x, y, w, h] for output
        return self._cxcywh_to_xywh(pred_box)

    def _xywh_to_cxcywh(self, box):
        """Convert [x, y, w, h] to [cx, cy, w, h]."""
        return np.array([
            box[0] + box[2] / 2,
            box[1] + box[3] / 2,
            box[2],
            box[3]
        ])

    def _cxcywh_to_xywh(self, box):
        """Convert [cx, cy, w, h] to [x, y, w, h]."""
        return np.array([
            box[0] - box[2] / 2,
            box[1] - box[3] / 2,
            box[2],
            box[3]
        ])

    def _sample_target(self, image, target_bb, search_area_factor, output_sz):
        """Sample target region from image.

        Args:
            image: Input image (H, W, C) or (H, W)
            target_bb: Target box [cx, cy, w, h]
            search_area_factor: Factor to expand target region
            output_sz: Output size

        Returns:
            crop: Cropped and resized region
            resize_factor: Resize factor applied
        """
        if image is None:
            return np.zeros((output_sz, output_sz, 3), dtype=np.uint8), 1.0

        x, y, w, h = target_bb

        # Safety: constrain box dimensions to reasonable values
        img_h, img_w = image.shape[:2]
        max_dim = max(img_h, img_w) * 2
        w = np.clip(w, 1.0, max_dim)
        h = np.clip(h, 1.0, max_dim)

        crop_sz = np.ceil(np.sqrt(w * h) * search_area_factor)

        # Safety: limit crop size
        crop_sz = min(crop_sz, max_dim)

        if crop_sz < 1:
            crop_sz = output_sz

        # Compute crop box
        x1 = round(x - crop_sz / 2)
        y1 = round(y - crop_sz / 2)
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        # Handle padding for out-of-bound regions
        x1_pad = max(0, -x1)
        y1_pad = max(0, -y1)
        x2_pad = max(0, x2 - image.shape[1])
        y2_pad = max(0, y2 - image.shape[0])

        # Clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # Crop
        crop = image[int(y1):int(y2), int(x1):int(x2)]

        # Pad if needed
        if x1_pad > 0 or y1_pad > 0 or x2_pad > 0 or y2_pad > 0:
            if len(image.shape) == 3:
                pad_width = ((int(y1_pad), int(y2_pad)), (int(x1_pad), int(x2_pad)), (0, 0))
            else:
                pad_width = ((int(y1_pad), int(y2_pad)), (int(x1_pad), int(x2_pad)))
            crop = np.pad(crop, pad_width, mode='constant', constant_values=0)

        # Resize
        resize_factor = output_sz / crop_sz
        crop_resized = cv2.resize(
            crop, (output_sz, output_sz), interpolation=cv2.INTER_LINEAR
        )

        return crop_resized, resize_factor

    def _preprocess_rgb(self, image):
        """Preprocess RGB image.

        Args:
            image: RGB image (H, W, 3) uint8

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        image = (image - self.mean_rgb) / self.std_rgb

        # Convert to tensor (1, 3, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

        return tensor

    def _preprocess_sonar(self, image):
        """Preprocess sonar image.

        Args:
            image: Sonar image (H, W) or (H, W, 1) uint8

        Returns:
            Preprocessed tensor (1, 1, H, W)
        """
        if len(image.shape) == 3:
            image = image[:, :, 0]

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply sonar normalization
        image = (image - self.mean_sonar) / self.std_sonar

        # Convert to tensor (1, 1, H, W)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

        return tensor

    def _map_box_back(self, pred_box, state_box, resize_factor, output_sz):
        """Map predicted box back to original image coordinates.

        Args:
            pred_box: Predicted box [cx, cy, w, h] normalized to [0, 1]
            state_box: Previous state box [cx, cy, w, h] in image coords
            resize_factor: Resize factor used for cropping
            output_sz: Output size of search region

        Returns:
            Mapped box [cx, cy, w, h] in image coordinates
        """
        # De-normalize
        pred_box = pred_box * output_sz

        # Get crop center
        cx_prev, cy_prev = state_box[0], state_box[1]
        half_side = 0.5 * output_sz / resize_factor

        # Map to image coordinates
        cx = pred_box[0] / resize_factor + (cx_prev - half_side)
        cy = pred_box[1] / resize_factor + (cy_prev - half_side)
        w = pred_box[2] / resize_factor
        h = pred_box[3] / resize_factor

        # Constrain box size to prevent runaway growth (max 10x initial size)
        w_prev, h_prev = state_box[2], state_box[3]
        max_ratio = 10.0
        w = np.clip(w, w_prev / max_ratio, w_prev * max_ratio)
        h = np.clip(h, h_prev / max_ratio, h_prev * max_ratio)

        # Ensure minimum size
        w = max(w, 10.0)
        h = max(h, 10.0)

        return np.array([cx, cy, w, h])


def get_tracker_class():
    """Return tracker class for external loading."""
    return AquaSOT3DTracker
