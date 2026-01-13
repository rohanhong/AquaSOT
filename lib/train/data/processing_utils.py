"""
Processing utilities for AquaSOT-3D training.

Based on OSTrack's processing_utils.py.
"""

import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """Extract a square crop centered at target bounding box.

    Args:
        im: Input image (H, W, C) or (H, W) for grayscale
        target_bb: Target bounding box [x, y, w, h]
        search_area_factor: Ratio of crop size to target size
        output_sz: Output size (resized if provided)
        mask: Optional mask to crop along with image

    Returns:
        im_crop: Cropped and optionally resized image
        resize_factor: Resize factor applied
        att_mask: Attention mask (1 for padded regions)
        mask_crop: Cropped mask (if mask provided)
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    # Calculate crop size
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    # Calculate crop coordinates
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    # Calculate padding
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target region
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad with constant value (black)
    if len(im.shape) == 3:
        im_crop_padded = cv2.copyMakeBorder(
            im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT
        )
    else:
        # Grayscale image
        im_crop_padded = cv2.copyMakeBorder(
            im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=0
        )

    # Create attention mask
    H, W = im_crop_padded.shape[:2]
    att_mask = np.ones((H, W))
    end_x = -x2_pad if x2_pad > 0 else None
    end_y = -y2_pad if y2_pad > 0 else None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if mask is not None:
        mask_crop_padded = cv2.copyMakeBorder(
            mask_crop.numpy() if isinstance(mask_crop, torch.Tensor) else mask_crop,
            y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=0
        )

    # Resize if output_sz specified
    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)

        if mask is not None:
            mask_crop_padded = cv2.resize(mask_crop_padded, (output_sz, output_sz))
            return im_crop_padded, resize_factor, att_mask, mask_crop_padded

        return im_crop_padded, resize_factor, att_mask

    if mask is not None:
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded

    return im_crop_padded, 1.0, att_mask.astype(np.bool_)


def transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz, normalize=False):
    """Transform box coordinates from image to crop coordinates.

    Args:
        box_in: Box to transform [x, y, w, h]
        box_extract: Box used for extraction [x, y, w, h]
        resize_factor: Resize factor applied to crop
        crop_sz: Size of crop [w, h]
        normalize: Whether to normalize to [0, 1]

    Returns:
        Transformed box coordinates
    """
    if isinstance(box_in, np.ndarray):
        box_in = torch.from_numpy(box_in)
    if isinstance(box_extract, np.ndarray):
        box_extract = torch.from_numpy(box_extract)
    if isinstance(crop_sz, (int, float)):
        crop_sz = torch.tensor([crop_sz, crop_sz])

    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))

    if normalize:
        return box_out / crop_sz[0]
    return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """Crop frames centered at jittered bounding boxes.

    Args:
        frames: List of frames to crop
        box_extract: List of boxes for extraction (jittered)
        box_gt: List of ground truth boxes
        search_area_factor: Crop size relative to target size
        output_sz: Output crop size
        masks: Optional masks to crop

    Returns:
        frames_crop: List of cropped frames
        box_crop: List of transformed box coordinates (normalized)
        att_mask: List of attention masks
        masks_crop: List of cropped masks (if provided)
    """
    if masks is None:
        crops_resize_factors = [
            sample_target(f, a, search_area_factor, output_sz)
            for f, a in zip(frames, box_extract)
        ]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [
            sample_target(f, a, search_area_factor, output_sz, m)
            for f, a, m in zip(frames, box_extract, masks)
        ]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)

    crop_sz = torch.tensor([output_sz, output_sz])

    # Transform box coordinates to crop coordinates (normalized)
    box_crop = [
        transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
        for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)
    ]

    return frames_crop, box_crop, att_mask, masks_crop
