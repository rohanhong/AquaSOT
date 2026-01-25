"""
Image transforms for AquaSOT training.

Based on OSTrack's transforms.py with additions for sonar data.
"""

import random
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as tvisf


class Transform:
    """Container for a sequence of transforms.

    Applies transforms in order, supporting joint transformation of
    images, bounding boxes, masks, and attention masks.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)

        if len(var_names) == 1:
            return out[var_names[0]]
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [
            {k: v for k, v in zip(var_names, vals)}
            for vals in zip(*[inputs[vn] for vn in var_names])
        ]
        for arg_name, arg_val in inputs.items():
            if arg_name != 'joint' and arg_name in self._valid_args:
                if isinstance(arg_val, list):
                    for inp, av in zip(split_inputs, arg_val):
                        inp[arg_name] = av
                else:
                    for inp in split_inputs:
                        inp[arg_name] = arg_val
        return split_inputs


class TransformBase:
    """Base class for individual transforms."""

    def __init__(self):
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = {}
        for var_name, var in input_vars.items():
            if var is not None:
                transform_func = getattr(self, 'transform_' + var_name)
                if var_name in ['coords', 'bbox']:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [transform_func(x, *params) for x in var]
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        im = inputs.get('image')
        if im is None:
            im = inputs.get('mask')
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        return None

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        return coords

    def transform_bbox(self, bbox, image_shape, *rand_params):
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox

        coord = bbox.clone().view(-1, 2).t().flip(0)
        x1, x2 = coord[1, 0], coord[1, 0] + coord[1, 1]
        y1, y2 = coord[0, 0], coord[0, 0] + coord[0, 1]
        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])
        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        return torch.cat((tl, sz), dim=-1).reshape(bbox.shape)

    def transform_mask(self, mask, *rand_params):
        return mask

    def transform_att(self, att, *rand_params):
        return att


class ToTensor(TransformBase):
    """Convert numpy array to tensor."""

    def transform_image(self, image):
        if image.ndim == 2:
            image = image[:, :, None]
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        return image

    def transform_mask(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        return mask

    def transform_att(self, att):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        return att


class ToTensorAndJitter(TransformBase):
    """Convert to tensor with brightness jitter."""

    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(
            max(0, 1 - self.brightness_jitter),
            1 + self.brightness_jitter
        )

    def transform_image(self, image, brightness_factor):
        if image.ndim == 2:
            image = image[:, :, None]
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if self.normalize:
            return image.float().mul(brightness_factor / 255.0).clamp(0.0, 1.0)
        return image.float().mul(brightness_factor).clamp(0.0, 255.0)

    def transform_mask(self, mask, brightness_factor):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        return mask

    def transform_att(self, att, brightness_factor):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        return att


class Normalize(TransformBase):
    """Normalize image with mean and std."""

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        return tvisf.normalize(image, self.mean, self.std, self.inplace)


class ToGrayscale(TransformBase):
    """Convert RGB to grayscale with probability."""

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_grayscale):
        if do_grayscale:
            if torch.is_tensor(image):
                # Convert tensor to grayscale
                weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
                gray = (image * weights).sum(dim=0, keepdim=True)
                return gray.expand(3, -1, -1)
            # NumPy array
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
        return image


class RandomHorizontalFlip(TransformBase):
    """Random horizontal flip."""

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((-1,))
            return np.fliplr(image).copy()
        return image

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1, :] = (image_shape[1] - 1) - coords[1, :]
            return coords_flip
        return coords

    def transform_mask(self, mask, do_flip):
        if do_flip:
            if torch.is_tensor(mask):
                return mask.flip((-1,))
            return np.fliplr(mask).copy()
        return mask

    def transform_att(self, att, do_flip):
        if do_flip:
            if torch.is_tensor(att):
                return att.flip((-1,))
            return np.fliplr(att).copy()
        return att


class RandomHorizontalFlipNorm(RandomHorizontalFlip):
    """Horizontal flip with normalized coordinates [0, 1]."""

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1, :] = 1 - coords[1, :]
            return coords_flip
        return coords


# ============================================================================
# Additional RGB transforms
# ============================================================================

class ColorJitter(TransformBase):
    """Color jitter for RGB images (contrast, saturation, hue).

    Applied after ToTensor, expects tensor in [0, 1] range.
    """

    def __init__(self, brightness=0.0, contrast=0.3, saturation=0.3, hue=0.2):
        super().__init__()
        self.brightness = brightness  # Not used here (handled by ToTensorAndJitter)
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def roll(self):
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        return (contrast_factor, saturation_factor, hue_factor)

    def transform_image(self, image, contrast_factor, saturation_factor, hue_factor):
        if not torch.is_tensor(image):
            return image

        # Apply contrast
        if self.contrast > 0:
            image = tvisf.adjust_contrast(image, contrast_factor)

        # Apply saturation
        if self.saturation > 0:
            image = tvisf.adjust_saturation(image, saturation_factor)

        # Apply hue
        if self.hue > 0:
            image = tvisf.adjust_hue(image, hue_factor)

        return image.clamp(0.0, 1.0)


class GaussianBlur(TransformBase):
    """Gaussian blur with random kernel size."""

    def __init__(self, probability=0.3, kernel_range=(3, 7)):
        super().__init__()
        self.probability = probability
        self.kernel_range = kernel_range

    def roll(self):
        do_blur = random.random() < self.probability
        # Kernel size must be odd
        k_min, k_max = self.kernel_range
        kernel_size = random.randrange(k_min, k_max + 1, 2)
        sigma = random.uniform(0.1, 2.0)
        return (do_blur, kernel_size, sigma)

    def transform_image(self, image, do_blur, kernel_size, sigma):
        if not do_blur:
            return image

        if torch.is_tensor(image):
            return tvisf.gaussian_blur(image, kernel_size, [sigma, sigma])

        # NumPy array
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


class RandomErasing(TransformBase):
    """Random erasing (occlusion) augmentation.

    Randomly erases a rectangular region in the image.
    """

    def __init__(self, probability=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
        super().__init__()
        self.probability = probability
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def roll(self):
        do_erase = random.random() < self.probability
        if not do_erase:
            return (False, 0, 0, 0, 0)

        # Generate random erase parameters
        scale = random.uniform(self.scale[0], self.scale[1])
        log_ratio = random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1]))
        aspect_ratio = np.exp(log_ratio)

        return (True, scale, aspect_ratio, random.random(), random.random())

    def transform_image(self, image, do_erase, scale, aspect_ratio, rand_x, rand_y):
        if not do_erase:
            return image

        if torch.is_tensor(image):
            c, h, w = image.shape
        else:
            h, w = image.shape[:2]

        # Calculate erase region size
        area = h * w * scale
        erase_h = int(round(np.sqrt(area * aspect_ratio)))
        erase_w = int(round(np.sqrt(area / aspect_ratio)))

        if erase_h >= h or erase_w >= w:
            return image

        # Calculate erase position
        top = int(rand_y * (h - erase_h))
        left = int(rand_x * (w - erase_w))

        if torch.is_tensor(image):
            image = image.clone()
            image[:, top:top + erase_h, left:left + erase_w] = self.value
        else:
            image = image.copy()
            image[top:top + erase_h, left:left + erase_w] = self.value

        return image


# ============================================================================
# Sonar-specific transforms
# ============================================================================

class SonarSpeckleNoise(TransformBase):
    """Add speckle noise to sonar images."""

    def __init__(self, probability=0.5, noise_level=0.1):
        super().__init__()
        self.probability = probability
        self.noise_level = noise_level

    def roll(self):
        do_noise = random.random() < self.probability
        noise_std = random.uniform(0, self.noise_level) if do_noise else 0
        return (do_noise, noise_std)

    def transform_image(self, image, do_noise, noise_std):
        if not do_noise or noise_std == 0:
            return image

        if torch.is_tensor(image):
            noise = torch.randn_like(image) * noise_std
            return (image + noise * image).clamp(0, 1)

        # NumPy array
        noise = np.random.randn(*image.shape) * noise_std
        noisy = image + noise * image.astype(np.float32)
        return np.clip(noisy, 0, 255).astype(image.dtype)


class SonarToTensor(TransformBase):
    """Convert sonar image (grayscale) to tensor."""

    def transform_image(self, image):
        if torch.is_tensor(image):
            if image.dim() == 2:
                return image.unsqueeze(0).float()
            return image.float()

        if image.ndim == 2:
            image = image[:, :, None]
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        return image.float().div(255)


class SonarNormalize(TransformBase):
    """Normalize sonar image."""

    def __init__(self, mean=0.5, std=0.5):
        super().__init__()
        self.mean = mean
        self.std = std

    def transform_image(self, image):
        return (image - self.mean) / self.std


class SonarRandomErasing(TransformBase):
    """Random erasing for sonar images.

    Simpler version that fills with random noise instead of constant value.
    """

    def __init__(self, probability=0.2, scale=(0.02, 0.15), ratio=(0.5, 2.0)):
        super().__init__()
        self.probability = probability
        self.scale = scale
        self.ratio = ratio

    def roll(self):
        do_erase = random.random() < self.probability
        if not do_erase:
            return (False, 0, 0, 0, 0)

        scale = random.uniform(self.scale[0], self.scale[1])
        log_ratio = random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1]))
        aspect_ratio = np.exp(log_ratio)

        return (True, scale, aspect_ratio, random.random(), random.random())

    def transform_image(self, image, do_erase, scale, aspect_ratio, rand_x, rand_y):
        if not do_erase:
            return image

        if torch.is_tensor(image):
            c, h, w = image.shape
        else:
            h, w = image.shape[:2]

        area = h * w * scale
        erase_h = int(round(np.sqrt(area * aspect_ratio)))
        erase_w = int(round(np.sqrt(area / aspect_ratio)))

        if erase_h >= h or erase_w >= w:
            return image

        top = int(rand_y * (h - erase_h))
        left = int(rand_x * (w - erase_w))

        if torch.is_tensor(image):
            image = image.clone()
            # Fill with random noise for sonar
            noise = torch.rand(c, erase_h, erase_w, device=image.device)
            image[:, top:top + erase_h, left:left + erase_w] = noise
        else:
            image = image.copy()
            noise = np.random.rand(erase_h, erase_w) * 255
            image[top:top + erase_h, left:left + erase_w] = noise.astype(image.dtype)

        return image


# ============================================================================
# Joint multi-modal transforms
# ============================================================================

class JointTransform:
    """Wrapper for synchronized transforms across RGB and Sonar modalities.

    Ensures spatial transforms (horizontal flip) use the same random decision
    for both modalities to maintain correspondence in uncalibrated data.

    Usage:
        joint_transform = JointTransform(
            rgb_transform=Transform(ToTensorAndJitter(), ColorJitter(), ...),
            sonar_transform=Transform(SonarToTensor(), SonarSpeckleNoise(), ...),
            joint_flip=RandomHorizontalFlip(probability=0.5)
        )
        rgb_out, sonar_out = joint_transform(
            rgb_image=rgb, rgb_bbox=rgb_box,
            sonar_image=sonar, sonar_bbox=sonar_box
        )
    """

    def __init__(self, rgb_transform, sonar_transform, joint_flip=None):
        """Initialize joint transform.

        Args:
            rgb_transform: Transform pipeline for RGB images
            sonar_transform: Transform pipeline for Sonar images
            joint_flip: Optional RandomHorizontalFlip for synchronized flipping
        """
        self.rgb_transform = rgb_transform
        self.sonar_transform = sonar_transform
        self.joint_flip = joint_flip

    def __call__(self, rgb_image, rgb_bbox, sonar_image, sonar_bbox,
                 rgb_mask=None, sonar_mask=None, rgb_att=None, sonar_att=None):
        """Apply transforms to RGB and Sonar data.

        Args:
            rgb_image: RGB image (numpy array or tensor)
            rgb_bbox: RGB bounding box
            sonar_image: Sonar image (numpy array or tensor)
            sonar_bbox: Sonar bounding box
            rgb_mask, sonar_mask: Optional masks
            rgb_att, sonar_att: Optional attention maps

        Returns:
            Tuple of (rgb_outputs, sonar_outputs) where each is a dict
        """
        # Apply synchronized horizontal flip if enabled
        if self.joint_flip is not None:
            # Roll once for both modalities
            rgb_out = self.joint_flip(
                image=rgb_image, bbox=rgb_bbox, mask=rgb_mask, att=rgb_att,
                new_roll=True
            )
            # Use same flip decision for sonar (new_roll=False)
            sonar_out = self.joint_flip(
                image=sonar_image, bbox=sonar_bbox, mask=sonar_mask, att=sonar_att,
                new_roll=False
            )

            rgb_image = rgb_out.get('image', rgb_image)
            rgb_bbox = rgb_out.get('bbox', rgb_bbox)
            rgb_mask = rgb_out.get('mask', rgb_mask)
            rgb_att = rgb_out.get('att', rgb_att)

            sonar_image = sonar_out.get('image', sonar_image)
            sonar_bbox = sonar_out.get('bbox', sonar_bbox)
            sonar_mask = sonar_out.get('mask', sonar_mask)
            sonar_att = sonar_out.get('att', sonar_att)

        # Apply modality-specific transforms
        rgb_inputs = {'image': rgb_image, 'bbox': rgb_bbox}
        if rgb_mask is not None:
            rgb_inputs['mask'] = rgb_mask
        if rgb_att is not None:
            rgb_inputs['att'] = rgb_att

        sonar_inputs = {'image': sonar_image, 'bbox': sonar_bbox}
        if sonar_mask is not None:
            sonar_inputs['mask'] = sonar_mask
        if sonar_att is not None:
            sonar_inputs['att'] = sonar_att

        rgb_outputs = self.rgb_transform(**rgb_inputs)
        sonar_outputs = self.sonar_transform(**sonar_inputs)

        # Ensure outputs are dicts
        if not isinstance(rgb_outputs, dict):
            rgb_outputs = {'image': rgb_outputs[0], 'bbox': rgb_outputs[1]}
        if not isinstance(sonar_outputs, dict):
            sonar_outputs = {'image': sonar_outputs[0], 'bbox': sonar_outputs[1]}

        return rgb_outputs, sonar_outputs
