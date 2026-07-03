"""Mask operation utilities."""

from __future__ import annotations

import cv2
import numpy as np


def refine_mask(
    mask: np.ndarray,
    logits: np.ndarray | None = None,
    threshold: float = 0.5,
    morph_kernel_size: int = 3,
    fill_holes: bool = True,
) -> np.ndarray:
    """Refine a SAM mask to have clean hard edges.

    Args:
        mask: binary mask from SAM (bool or uint8)
        logits: raw logits from SAM (optional, for re-thresholding)
        threshold: logit threshold for binarization (higher = tighter)
        morph_kernel_size: kernel size for morphological cleanup (0 = skip)
        fill_holes: whether to fill internal holes

    Returns:
        Clean binary mask (uint8, 0 or 1)
    """
    # Step 1: Use mask directly (already upscaled to image size).
    # Logits are at SAM's internal resolution (256x256), not usable directly.
    binary = mask.astype(np.uint8)

    # Step 2: Morphological close to seal small gaps
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Step 3: Fill internal holes (flood fill from border)
    if fill_holes:
        h, w = binary.shape
        flood = binary.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 1)
        flood_inv = 1 - flood
        binary = np.maximum(binary, flood_inv)

    # Step 4: Morphological open to remove tiny noise
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def mask_to_bbox(mask: np.ndarray) -> dict | None:
    """Convert binary mask to bounding box {x, y, width, height}."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())
    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
    }


def mask_to_contours(mask: np.ndarray, simplify_epsilon: float = 2.0) -> list[list[tuple[int, int]]]:
    """Convert binary mask to simplified contour points."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for cnt in contours:
        epsilon = simplify_epsilon
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        points = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        if len(points) >= 3:
            result.append(points)
    return result


def mask_area_ratio(mask: np.ndarray) -> float:
    """Calculate what fraction of the image the mask covers."""
    return float(np.sum(mask > 0)) / mask.size


def merge_overlapping_masks(masks: list[np.ndarray], iou_threshold: float = 0.5) -> list[np.ndarray]:
    """Merge masks that overlap significantly."""
    if not masks:
        return []

    keep = []
    used = set()
    for i, m1 in enumerate(masks):
        if i in used:
            continue
        merged = m1.copy()
        for j, m2 in enumerate(masks):
            if j <= i or j in used:
                continue
            intersection = np.logical_and(merged, m2).sum()
            union = np.logical_or(merged, m2).sum()
            if union > 0 and intersection / union > iou_threshold:
                merged = np.logical_or(merged, m2).astype(np.uint8)
                used.add(j)
        keep.append(merged)
        used.add(i)
    return keep


def crop_with_mask(image: np.ndarray, mask: np.ndarray, padding: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to the mask's bounding box with padding."""
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return image, mask

    h, w = image.shape[:2]
    x1 = max(0, bbox["x"] - padding)
    y1 = max(0, bbox["y"] - padding)
    x2 = min(w, bbox["x"] + bbox["width"] + padding)
    y2 = min(h, bbox["y"] + bbox["height"] + padding)

    cropped_img = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    return cropped_img, cropped_mask


def apply_mask_to_image(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image, setting background to black."""
    if len(image_rgb.shape) == 3:
        return image_rgb * mask[:, :, np.newaxis]
    return image_rgb * mask
