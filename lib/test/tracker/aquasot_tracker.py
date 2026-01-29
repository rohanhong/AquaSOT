"""
AquaSOT tracker wrapper for evaluation.

Based on OSTrack's tracker wrapper pattern.
"""

import os
import torch
import numpy as np
import cv2

from .base_tracker import Tracker
from lib.models.aquasot import build_aquasot
from lib.config.aquasot.config import cfg, update_config_from_file
from lib.test.utils.hann import hann2d
from lib.utils.modality_dropout import apply_modality_dropout


class AquaSOTTracker(Tracker):
    """AquaSOT tracker for evaluation.

    Wraps the AquaSOT model for running on benchmarks.

    Args:
        params: Tracker parameters (config path, checkpoint path, etc.)
        dataset_name: Name of dataset being evaluated
    """

    def __init__(self, params, dataset_name=None):
        super().__init__(name='AquaSOT', is_deterministic=True)
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
        self.network = build_aquasot(cfg, training=False)

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

        # Template update parameters (MixFormer-style three-tier strategy)
        self.template_update_enabled = cfg.TEST.TEMPLATE_UPDATE.ENABLED
        self.template_update_strategy = cfg.TEST.TEMPLATE_UPDATE.STRATEGY
        self.online_size = cfg.TEST.TEMPLATE_UPDATE.ONLINE_SIZE
        self.update_interval = cfg.TEST.TEMPLATE_UPDATE.UPDATE_INTERVAL
        self.score_threshold = cfg.TEST.TEMPLATE_UPDATE.SCORE_THRESHOLD
        self.high_confidence_threshold = cfg.TEST.TEMPLATE_UPDATE.HIGH_CONFIDENCE_THRESHOLD
        self.max_score_decay = cfg.TEST.TEMPLATE_UPDATE.MAX_SCORE_DECAY

        # Target loss detection and recovery (dual-modal independent state machine)
        self.target_loss_enabled = cfg.TEST.TARGET_LOSS.ENABLED
        if self.target_loss_enabled:
            from .target_loss_handler import DualModalTargetLossHandler
            self.loss_handler = DualModalTargetLossHandler(cfg)
        else:
            self.loss_handler = None

        # Dynamic search factor based on tracking state
        # Reference: DiMP uses search_area_scale=4.5-5.0, NA-OSTrack uses fixed K=3
        self.dynamic_search_enabled = (
            self.target_loss_enabled and
            cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR.ENABLED
        )
        if self.dynamic_search_enabled:
            dsf = cfg.TEST.TARGET_LOSS.DYNAMIC_SEARCH_FACTOR
            # RGB-specific search factors
            self.rgb_search_factors = {
                'STABLE': dsf.RGB_STABLE,
                'UNCERTAIN': dsf.RGB_UNCERTAIN,
                'LOST': dsf.RGB_LOST,
            }
            # Sonar-specific search factors
            self.sonar_search_factors = {
                'STABLE': dsf.SONAR_STABLE,
                'UNCERTAIN': dsf.SONAR_UNCERTAIN,
                'LOST': dsf.SONAR_LOST,
            }

        # Previous frame tracking states (for dynamic search factor)
        self.prev_rgb_state = 'STABLE'
        self.prev_sonar_state = 'STABLE'

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

        # Backward compatibility: migrate shared box_head to separate heads
        # Old checkpoints have 'box_head.*', new model expects 'rgb_box_head.*' and 'sonar_box_head.*'
        migrated_state_dict = {}
        has_old_box_head = any(k.startswith('box_head.') for k in state_dict.keys())
        has_new_box_heads = any(k.startswith('rgb_box_head.') for k in state_dict.keys())

        if has_old_box_head and not has_new_box_heads:
            print('[AquaSOTTracker] Migrating old checkpoint: box_head -> rgb_box_head, sonar_box_head')
            for key, value in state_dict.items():
                if key.startswith('box_head.'):
                    # Copy to both RGB and Sonar heads
                    rgb_key = key.replace('box_head.', 'rgb_box_head.')
                    sonar_key = key.replace('box_head.', 'sonar_box_head.')
                    migrated_state_dict[rgb_key] = value.clone()
                    migrated_state_dict[sonar_key] = value.clone()
                else:
                    migrated_state_dict[key] = value
            state_dict = migrated_state_dict

        missing, unexpected = self.network.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[AquaSOTTracker] Missing keys: {missing}')
        if unexpected:
            print(f'[AquaSOTTracker] Unexpected keys: {unexpected}')

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

        # Handle box format (dict for dual-modal, array for single-modal)
        if isinstance(box, dict):
            rgb_box = box.get('rgb') if 'rgb' in box else box.get('bbox')
            sonar_box = box.get('sonar') if 'sonar' in box else rgb_box
            if rgb_box is None and sonar_box is None:
                raise ValueError("init_box dict must contain 'rgb', 'sonar', or 'bbox' key")
            # Use available box as fallback for the other modality
            if rgb_box is None:
                rgb_box = sonar_box
            elif sonar_box is None:
                sonar_box = rgb_box
            rgb_box = np.array(rgb_box)
            sonar_box = np.array(sonar_box)
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

        # Apply modality dropout for ablation testing
        # When ENABLED=False (default), this is a no-op for backward compatibility
        rgb_template, sonar_template, _ = apply_modality_dropout(
            rgb_template, sonar_template, cfg
        )

        # Initialize three-tier template structure (MixFormer style)
        # Tier 1: First-frame template (fixed, never updated)
        self.rgb_template = rgb_template.clone()
        self.sonar_template = sonar_template.clone()

        # Tier 2: Online template collection (FIFO queue)
        self.rgb_online_template = rgb_template.clone()
        self.sonar_online_template = sonar_template.clone()

        # Tier 3: Best-frame cache (highest score in current interval)
        self.rgb_online_max_template = rgb_template.clone()
        self.sonar_online_max_template = sonar_template.clone()

        # Template update tracking
        self.rgb_max_pred_score = -1.0
        self.sonar_max_pred_score = -1.0
        self.online_forget_id = 0
        self.last_update_frame = 0

        # Set templates in model
        if self.template_update_enabled and self.online_size > 0:
            self.network.set_online(
                self.rgb_template, self.sonar_template,
                self.rgb_online_template, self.sonar_online_template
            )
        else:
            self.network.set_online(rgb_template, sonar_template)

        # Save state
        self.state = {
            'rgb_box': rgb_box_cxcywh,
            'sonar_box': sonar_box_cxcywh,
            'rgb_image_shape': rgb_image.shape[:2] if rgb_image is not None else None,
            'sonar_image_shape': sonar_image.shape[:2] if sonar_image is not None else None,
        }
        self.frame_id = 0

        # Initialize target loss handler with first frame info
        if self.target_loss_enabled and self.loss_handler is not None:
            rgb_shape = rgb_image.shape[:2] if rgb_image is not None else (1080, 1920)
            # IMPORTANT: Use Sonar-specific default resolution, not RGB
            # Sonar has unified resolution 682x376 after preprocessing
            sonar_shape = sonar_image.shape[:2] if sonar_image is not None else (376, 682)
            self.loss_handler.initialize(
                rgb_init_box=rgb_box_cxcywh,
                sonar_init_box=sonar_box_cxcywh,
                rgb_image_shape=rgb_shape,
                sonar_image_shape=sonar_shape
            )

        # Reset tracking states for dynamic search factor
        self.prev_rgb_state = 'STABLE'
        self.prev_sonar_state = 'STABLE'

    def _get_dynamic_search_factor(self, modality, state_name):
        """Get search factor based on tracking state.

        When tracking is UNCERTAIN or LOST, expand search region to increase
        chances of re-acquiring the target.

        Reference: DiMP uses search_area_scale=4.5-5.0, NA-OSTrack uses fixed K=3

        Args:
            modality: 'rgb' or 'sonar'
            state_name: Tracking state name ('STABLE', 'UNCERTAIN', 'LOST')

        Returns:
            float: Search factor for the given modality and state
        """
        if not self.dynamic_search_enabled:
            return self.search_factor

        if modality == 'rgb':
            return self.rgb_search_factors.get(state_name, self.search_factor)
        else:  # sonar
            return self.sonar_search_factors.get(state_name, self.search_factor)

    def update(self, image):
        """Track target in current frame.

        Args:
            image: Current frame (dict with 'rgb' and 'sonar' keys) or RGB image

        Returns:
            dict: Dual-modal prediction results with keys:
                - target_bbox: Primary bbox [x, y, w, h] (RGB)
                - rgb_bbox: RGB prediction [x, y, w, h]
                - sonar_bbox: Sonar prediction [x, y, w, h]
                - rgb_score: RGB confidence score
                - sonar_score: Sonar confidence score
        """
        self.frame_id += 1

        # Handle multi-modal or single-modal input
        if isinstance(image, dict):
            rgb_image = image.get('rgb')
            sonar_image = image.get('sonar')
        else:
            rgb_image = image
            sonar_image = None

        # Get dynamic search factors based on previous frame tracking state
        # This allows expanding search region when tracking is uncertain or lost
        rgb_search_factor = self._get_dynamic_search_factor('rgb', self.prev_rgb_state)
        sonar_search_factor = self._get_dynamic_search_factor('sonar', self.prev_sonar_state)

        # Crop and preprocess search region with dynamic factors
        rgb_search, rgb_resize_factor = self._sample_target(
            rgb_image, self.state['rgb_box'], rgb_search_factor, self.search_size
        )
        rgb_search = self._preprocess_rgb(rgb_search)

        if sonar_image is not None:
            sonar_search, sonar_resize_factor = self._sample_target(
                sonar_image, self.state['sonar_box'], sonar_search_factor, self.search_size
            )
            sonar_search = self._preprocess_sonar(sonar_search)
        else:
            sonar_search = torch.zeros(1, 1, self.search_size, self.search_size)
            sonar_resize_factor = rgb_resize_factor

        # Move to device
        rgb_search = rgb_search.to(self.device)
        sonar_search = sonar_search.to(self.device)

        # Apply modality dropout for ablation testing
        # When ENABLED=False (default), this is a no-op for backward compatibility
        rgb_search, sonar_search, _ = apply_modality_dropout(
            rgb_search, sonar_search, cfg
        )

        # Run tracking
        with torch.no_grad():
            out = self.network.track(rgb_search, sonar_search)

        # Extract confidence scores from score maps (peak value)
        rgb_score = out['rgb_score_map'].flatten(1).max(dim=1)[0].item()
        sonar_score = out['sonar_score_map'].flatten(1).max(dim=1)[0].item()

        # Apply Hann window to RGB score map for motion constraint
        rgb_response = self.output_window * out['rgb_score_map']

        # Recalculate RGB bbox using windowed response
        rgb_pred_box = self.network.rgb_box_head.cal_bbox(
            rgb_response, out['rgb_size_map'], out['rgb_offset_map']
        )
        rgb_pred_box = rgb_pred_box.cpu().numpy()[0]  # [cx, cy, w, h] normalized

        # Map RGB bbox back to original image coordinates
        rgb_pred_box = self._map_box_back(
            rgb_pred_box, self.state['rgb_box'], rgb_resize_factor, self.search_size
        )

        # Process Sonar prediction
        if sonar_image is not None:
            # Apply Hann window to Sonar score map
            sonar_response = self.output_window * out['sonar_score_map']
            sonar_pred_box = self.network.sonar_box_head.cal_bbox(
                sonar_response, out['sonar_size_map'], out['sonar_offset_map']
            )
            sonar_pred_box = sonar_pred_box.cpu().numpy()[0]
            sonar_pred_box = self._map_box_back(
                sonar_pred_box, self.state['sonar_box'], sonar_resize_factor, self.search_size
            )
        else:
            sonar_pred_box = rgb_pred_box.copy()

        # Target loss detection and recovery (dual-modal independent state machine)
        if self.target_loss_enabled and self.loss_handler is not None:
            from .target_loss_handler import TrackingState
            loss_result = self.loss_handler.check(
                rgb_pred_box=rgb_pred_box,
                sonar_pred_box=sonar_pred_box,
                rgb_score=rgb_score,
                sonar_score=sonar_score,
                rgb_prev_box=self.state['rgb_box'],
                sonar_prev_box=self.state['sonar_box']
            )

            # Update RGB state only if RGB modality approves
            if loss_result['rgb_should_update']:
                self.state['rgb_box'] = rgb_pred_box.copy()
            else:
                rgb_pred_box = loss_result['rgb_box']

            # Update Sonar state only if Sonar modality approves
            if loss_result['sonar_should_update']:
                self.state['sonar_box'] = sonar_pred_box.copy()
            else:
                sonar_pred_box = loss_result['sonar_box']

            # Select primary output based on modality states and scores
            primary_pred_box = loss_result['primary_box']

            # Template update: only update each modality when its state is STABLE
            if self.template_update_enabled:
                rgb_stable = loss_result['rgb_state'] == TrackingState.STABLE
                sonar_stable = loss_result['sonar_state'] == TrackingState.STABLE
                self._update_template_selective(
                    image, rgb_pred_box, sonar_pred_box,
                    rgb_score, sonar_score,
                    update_rgb=rgb_stable, update_sonar=sonar_stable
                )

            # Store tracking states for return
            rgb_state_name = loss_result['rgb_state'].name
            sonar_state_name = loss_result['sonar_state'].name
        else:
            # Original behavior (no target loss detection)
            self.state['rgb_box'] = rgb_pred_box.copy()
            self.state['sonar_box'] = sonar_pred_box.copy()
            primary_pred_box = rgb_pred_box

            if self.template_update_enabled:
                self._update_template(image, rgb_pred_box, sonar_pred_box, rgb_score, sonar_score)

            rgb_state_name = 'STABLE'
            sonar_state_name = 'STABLE'

        # Update previous tracking states for next frame's dynamic search factor
        self.prev_rgb_state = rgb_state_name
        self.prev_sonar_state = sonar_state_name

        # Return dual-modal results with tracking states
        return {
            'target_bbox': self._cxcywh_to_xywh(primary_pred_box),
            'rgb_bbox': self._cxcywh_to_xywh(rgb_pred_box),
            'sonar_bbox': self._cxcywh_to_xywh(sonar_pred_box),
            'rgb_score': rgb_score,
            'sonar_score': sonar_score,
            'rgb_state': rgb_state_name,
            'sonar_state': sonar_state_name,
        }

    def _update_template(self, image, rgb_box, sonar_box, rgb_score, sonar_score):
        """Update templates using MixFormer three-tier strategy.

        Args:
            image: Current frame (dict or image)
            rgb_box: RGB prediction [cx, cy, w, h]
            sonar_box: Sonar prediction [cx, cy, w, h]
            rgb_score: RGB confidence score
            sonar_score: Sonar confidence score
        """
        # Get images
        if isinstance(image, dict):
            rgb_image = image.get('rgb')
            sonar_image = image.get('sonar')
        else:
            rgb_image = image
            sonar_image = None

        # Decay max scores (MixFormer uses decay to gradually lower threshold)
        self.rgb_max_pred_score *= self.max_score_decay
        self.sonar_max_pred_score *= self.max_score_decay

        # Update RGB best-frame cache if current score is better
        if rgb_score > self.score_threshold and rgb_score > self.rgb_max_pred_score:
            rgb_template_new, _ = self._sample_target(
                rgb_image, rgb_box, self.template_factor, self.template_size
            )
            self.rgb_online_max_template = self._preprocess_rgb(rgb_template_new).to(self.device)
            self.rgb_max_pred_score = rgb_score

        # Update Sonar best-frame cache if current score is better
        if sonar_image is not None:
            if sonar_score > self.score_threshold and sonar_score > self.sonar_max_pred_score:
                sonar_template_new, _ = self._sample_target(
                    sonar_image, sonar_box, self.template_factor, self.template_size
                )
                self.sonar_online_max_template = self._preprocess_sonar(sonar_template_new).to(self.device)
                self.sonar_max_pred_score = sonar_score

        # Push best-frame to online queue at update interval
        if self.frame_id % self.update_interval == 0 and self.frame_id > 0:
            self._push_to_online_queue()

    def _update_template_selective(self, image, rgb_box, sonar_box,
                                    rgb_score, sonar_score,
                                    update_rgb=True, update_sonar=True):
        """Selectively update templates based on modality tracking states.

        Only updates templates for modalities that are in STABLE state,
        preventing template pollution when target is lost.

        Args:
            image: Current frame (dict or image)
            rgb_box: RGB prediction [cx, cy, w, h]
            sonar_box: Sonar prediction [cx, cy, w, h]
            rgb_score: RGB confidence score
            sonar_score: Sonar confidence score
            update_rgb: Whether to update RGB template (True if RGB is STABLE)
            update_sonar: Whether to update Sonar template (True if Sonar is STABLE)
        """
        # Get images
        if isinstance(image, dict):
            rgb_image = image.get('rgb')
            sonar_image = image.get('sonar')
        else:
            rgb_image = image
            sonar_image = None

        # Decay max scores (MixFormer uses decay to gradually lower threshold)
        self.rgb_max_pred_score *= self.max_score_decay
        self.sonar_max_pred_score *= self.max_score_decay

        # Use higher confidence threshold for selective template updates
        # This prevents template pollution during uncertain tracking
        update_threshold = self.high_confidence_threshold

        # Update RGB best-frame cache only if RGB is STABLE and score is high
        if update_rgb and rgb_score > update_threshold and rgb_score > self.rgb_max_pred_score:
            rgb_template_new, _ = self._sample_target(
                rgb_image, rgb_box, self.template_factor, self.template_size
            )
            self.rgb_online_max_template = self._preprocess_rgb(rgb_template_new).to(self.device)
            self.rgb_max_pred_score = rgb_score

        # Update Sonar best-frame cache only if Sonar is STABLE and score is high
        if update_sonar and sonar_image is not None:
            if sonar_score > update_threshold and sonar_score > self.sonar_max_pred_score:
                sonar_template_new, _ = self._sample_target(
                    sonar_image, sonar_box, self.template_factor, self.template_size
                )
                self.sonar_online_max_template = self._preprocess_sonar(sonar_template_new).to(self.device)
                self.sonar_max_pred_score = sonar_score

        # Push best-frame to online queue at update interval
        if self.frame_id % self.update_interval == 0 and self.frame_id > 0:
            self._push_to_online_queue()

    def _push_to_online_queue(self):
        """Push best-frame templates to online template queue (FIFO)."""
        if self.online_size == 1:
            # Single template mode: direct replacement
            self.rgb_online_template = self.rgb_online_max_template.clone()
            self.sonar_online_template = self.sonar_online_max_template.clone()
        elif self.rgb_online_template.shape[0] < self.online_size:
            # Queue not full: append new template
            self.rgb_online_template = torch.cat([
                self.rgb_online_template, self.rgb_online_max_template
            ], dim=0)
            self.sonar_online_template = torch.cat([
                self.sonar_online_template, self.sonar_online_max_template
            ], dim=0)
        else:
            # Queue full: FIFO replacement
            self.rgb_online_template[self.online_forget_id] = self.rgb_online_max_template[0]
            self.sonar_online_template[self.online_forget_id] = self.sonar_online_max_template[0]
            self.online_forget_id = (self.online_forget_id + 1) % self.online_size

        # Update model with new templates
        self.network.set_online(
            self.rgb_template, self.sonar_template,
            self.rgb_online_template, self.sonar_online_template
        )

        # Reset best-frame cache for next interval
        self.rgb_max_pred_score = -1.0
        self.sonar_max_pred_score = -1.0
        self.rgb_online_max_template = self.rgb_template.clone()
        self.sonar_online_max_template = self.sonar_template.clone()
        self.last_update_frame = self.frame_id

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
    return AquaSOTTracker
