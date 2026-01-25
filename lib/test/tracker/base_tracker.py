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

        # First frame uses initial box
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
                self._visualize(image, rgb_boxes[f])

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

    def _visualize(self, image, box):
        """Visualize tracking result.

        Args:
            image: Current frame
            box: Predicted bounding box [x, y, w, h]
        """
        import cv2

        if isinstance(image, dict):
            image = image.get('rgb', list(image.values())[0])

        if image is None:
            return

        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(self.name, vis_image)
        cv2.waitKey(1)
