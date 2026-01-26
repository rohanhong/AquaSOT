"""
Base tracker interface for AquaSOT.

Based on got10k-toolkit Tracker class.
"""

from abc import ABC, abstractmethod


class Tracker(ABC):
    """Base class for all trackers.

    All trackers should inherit from this class and implement:
        - init(image, box): Initialize tracker with first frame
        - update(image): Track target in subsequent frames

    Args:
        name: Tracker name for identification
        is_deterministic: Whether tracker produces deterministic results
    """

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic

    @abstractmethod
    def init(self, image, box):
        """Initialize tracker with first frame and bounding box.

        Args:
            image: First frame (H, W, C) numpy array or dict for multi-modal
            box: Initial bounding box [x, y, w, h]
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, image):
        """Track target in current frame.

        Args:
            image: Current frame (H, W, C) numpy array or dict for multi-modal

        Returns:
            box: Predicted bounding box [x, y, w, h]
        """
        raise NotImplementedError

    def track(self, img_files, box, visualize=False):
        """Run tracker on a sequence.

        Args:
            img_files: List of image file paths or list of dicts for multi-modal
            box: Initial bounding box [x, y, w, h]
            visualize: Whether to visualize tracking results

        Returns:
            dict: Tracking results with keys:
                - boxes: Array of primary boxes (N, 4), same as rgb_boxes
                - rgb_boxes: Array of RGB predictions (N, 4)
                - sonar_boxes: Array of Sonar predictions (N, 4)
                - rgb_scores: Array of RGB confidence scores (N,)
                - sonar_scores: Array of Sonar confidence scores (N,)
                - times: Array of frame processing times (N,)
        """
        import numpy as np
        import time

        frame_num = len(img_files)
        rgb_boxes = np.zeros((frame_num, 4))
        sonar_boxes = np.zeros((frame_num, 4))
        rgb_scores = np.zeros(frame_num)
        sonar_scores = np.zeros(frame_num)
        times = np.zeros(frame_num)

        # First frame uses initial box (handle dict format for dual-modal)
        if isinstance(box, dict):
            rgb_boxes[0] = box.get('rgb') if 'rgb' in box else box.get('bbox')
            sonar_boxes[0] = box.get('sonar') if 'sonar' in box else rgb_boxes[0]
            if rgb_boxes[0] is None and sonar_boxes[0] is None:
                raise ValueError("init_box dict must contain 'rgb', 'sonar', or 'bbox' key")
            # Use available box as fallback for the other modality
            if rgb_boxes[0] is None:
                rgb_boxes[0] = sonar_boxes[0]
            elif sonar_boxes[0] is None:
                sonar_boxes[0] = rgb_boxes[0]
        else:
            rgb_boxes[0] = box
            sonar_boxes[0] = box
        rgb_scores[0] = 1.0
        sonar_scores[0] = 1.0

        for f, img_file in enumerate(img_files):
            # Load image
            image = self._load_image(img_file)

            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                result = self.update(image)
                # Handle dict return (new format) or array return (legacy)
                if isinstance(result, dict):
                    rgb_boxes[f, :] = result['rgb_bbox']
                    sonar_boxes[f, :] = result['sonar_bbox']
                    rgb_scores[f] = result['rgb_score']
                    sonar_scores[f] = result['sonar_score']
                else:
                    # Legacy: single bbox return
                    rgb_boxes[f, :] = result
                    sonar_boxes[f, :] = result
                    rgb_scores[f] = 1.0
                    sonar_scores[f] = 1.0
            times[f] = time.time() - start_time

            if visualize:
                # Get tracking states if available (first frame always STABLE)
                if f == 0:
                    rgb_state = 'STABLE'
                    sonar_state = 'STABLE'
                else:
                    rgb_state = result.get('rgb_state', 'STABLE') if isinstance(result, dict) else 'STABLE'
                    sonar_state = result.get('sonar_state', 'STABLE') if isinstance(result, dict) else 'STABLE'
                self._visualize(
                    image, rgb_boxes[f], sonar_boxes[f],
                    rgb_scores[f], sonar_scores[f],
                    rgb_state, sonar_state
                )

        return {
            'boxes': rgb_boxes,  # Primary boxes (backward compatible)
            'rgb_boxes': rgb_boxes,
            'sonar_boxes': sonar_boxes,
            'rgb_scores': rgb_scores,
            'sonar_scores': sonar_scores,
            'times': times,
        }

    def _load_image(self, img_file):
        """Load image from file path or return dict for multi-modal.

        Args:
            img_file: Image file path (str) or dict with paths

        Returns:
            Loaded image or dict of images
        """
        import cv2

        if isinstance(img_file, str):
            image = cv2.imread(img_file)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        elif isinstance(img_file, dict):
            # Multi-modal: load each modality
            images = {}
            for key, path in img_file.items():
                if path is not None:
                    img = cv2.imread(path)
                    if img is not None:
                        if key == 'sonar':
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images[key] = img
            return images
        return img_file

    def _visualize(self, image, rgb_box, sonar_box=None,
                   rgb_score=1.0, sonar_score=1.0,
                   rgb_state='STABLE', sonar_state='STABLE'):
        """Visualize tracking result with dual-modal side-by-side display.

        Args:
            image: Current frame (dict with 'rgb' and 'sonar' keys) or RGB image
            rgb_box: RGB predicted bounding box [x, y, w, h]
            sonar_box: Sonar predicted bounding box [x, y, w, h] (optional)
            rgb_score: RGB confidence score
            sonar_score: Sonar confidence score
            rgb_state: RGB tracking state ('STABLE', 'UNCERTAIN', 'LOST')
            sonar_state: Sonar tracking state ('STABLE', 'UNCERTAIN', 'LOST')
        """
        import cv2
        import numpy as np

        # State color mapping
        state_colors = {
            'STABLE': (0, 255, 0),     # Green
            'UNCERTAIN': (0, 255, 255), # Yellow
            'LOST': (0, 0, 255),        # Red
        }

        # Extract RGB and Sonar images
        if isinstance(image, dict):
            rgb_image = image.get('rgb')
            sonar_image = image.get('sonar')
        else:
            rgb_image = image
            sonar_image = None

        # Use rgb_box for sonar_box if not provided
        if sonar_box is None:
            sonar_box = rgb_box

        # Prepare RGB visualization
        if rgb_image is not None:
            rgb_vis = rgb_image.copy()
            if len(rgb_vis.shape) == 2:
                rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_GRAY2BGR)
            else:
                rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)

            # Draw RGB bounding box with state color
            rgb_color = state_colors.get(rgb_state, (0, 255, 0))
            x, y, w, h = [int(v) for v in rgb_box]
            cv2.rectangle(rgb_vis, (x, y), (x + w, y + h), rgb_color, 2)

            # Add RGB label and state
            cv2.putText(rgb_vis, f"RGB [{rgb_state}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rgb_color, 2)
            cv2.putText(rgb_vis, f"Score: {rgb_score:.3f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            rgb_vis = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(rgb_vis, "RGB: N/A", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # Prepare Sonar visualization and display side-by-side
        if sonar_image is not None:
            sonar_vis = sonar_image.copy()
            if len(sonar_vis.shape) == 2:
                sonar_vis = cv2.cvtColor(sonar_vis, cv2.COLOR_GRAY2BGR)

            # Draw Sonar bounding box with state color
            sonar_color = state_colors.get(sonar_state, (0, 255, 255))
            x, y, w, h = [int(v) for v in sonar_box]
            cv2.rectangle(sonar_vis, (x, y), (x + w, y + h), sonar_color, 2)

            # Add Sonar label and state
            cv2.putText(sonar_vis, f"Sonar [{sonar_state}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sonar_color, 2)
            cv2.putText(sonar_vis, f"Score: {sonar_score:.3f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Resize to match heights for side-by-side display
            target_h = max(rgb_vis.shape[0], sonar_vis.shape[0])
            if rgb_vis.shape[0] != target_h:
                scale = target_h / rgb_vis.shape[0]
                new_w = int(rgb_vis.shape[1] * scale)
                rgb_vis = cv2.resize(rgb_vis, (new_w, target_h))
            if sonar_vis.shape[0] != target_h:
                scale = target_h / sonar_vis.shape[0]
                new_w = int(sonar_vis.shape[1] * scale)
                sonar_vis = cv2.resize(sonar_vis, (new_w, target_h))

            # Combine side-by-side
            combined = np.hstack([rgb_vis, sonar_vis])
            cv2.imshow(self.name, combined)
        else:
            # Only RGB available
            cv2.imshow(self.name, rgb_vis)

        cv2.waitKey(1)
