"""Edge refinement utilities — Canny edge snapping for PS-like hard edges."""

from __future__ import annotations

import cv2
import numpy as np


def refine_mask_with_edges(
    mask: np.ndarray,
    logits: np.ndarray | None,
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    prob_threshold: float = 0.5,
    canny_low: int = 50,
    canny_high: int = 150,
    snap_radius: int = 5,
    morph_kernel_size: int = 3,
    fill_holes: bool = True,
) -> np.ndarray:
    """Refine SAM mask using Canny edge snapping for PS-like hard edges.

    Pipeline:
        1. SAM logit → threshold → initial binary mask
        2. Canny edge detection on the ROI
        3. Distance transform → snap mask edges to nearest Canny edges
        4. Contour reconstruction + flood fill
        5. Morphological cleanup

    Args:
        mask: binary mask from SAM (bool or uint8)
        logits: raw logits from SAM (for re-thresholding)
        image_rgb: original image in RGB format
        box: (x1, y1, x2, y2) bounding box used for SAM prompt
        prob_threshold: logit threshold for initial binarization
        canny_low: Canny edge detection low threshold
        canny_high: Canny edge detection high threshold
        snap_radius: max pixel distance to snap edges
        morph_kernel_size: kernel size for morphological cleanup
        fill_holes: whether to fill internal holes

    Returns:
        Refined binary mask (uint8, 0 or 1) with hard edges
    """
    x1, y1, x2, y2 = box
    img_h, img_w = image_rgb.shape[:2]

    # Clamp box to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Step 1: Initial binary mask
    # SAM masks are already upscaled to original image size, but logits
    # stay at internal resolution (typically 256x256). Use the mask directly.
    binary = mask.astype(np.uint8)

    # If mask is empty or nearly empty, return as-is
    if binary.sum() < 10:
        return binary

    # Step 2: Canny edge detection within the ROI
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Expand ROI slightly for edge context
    pad = 20
    roi_x1 = max(0, x1 - pad)
    roi_y1 = max(0, y1 - pad)
    roi_x2 = min(img_w, x2 + pad)
    roi_y2 = min(img_h, y2 + pad)

    roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

    # Adaptive Canny thresholds based on image content
    roi_mean = np.mean(roi_gray)
    adaptive_low = max(10, int(roi_mean * 0.5))
    adaptive_high = min(255, int(roi_mean * 1.5))
    actual_low = min(canny_low, adaptive_low)
    actual_high = min(canny_high, adaptive_high)

    edges_roi = cv2.Canny(roi_gray, actual_low, actual_high)

    # Dilate edges slightly to create a "snap zone"
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_roi = cv2.dilate(edges_roi, edge_kernel, iterations=1)

    # Create full-image edges map
    edges_full = np.zeros((img_h, img_w), dtype=np.uint8)
    edges_full[roi_y1:roi_y2, roi_x1:roi_x2] = edges_roi

    # Step 3: Edge snapping — move mask boundary to nearest Canny edge
    snapped = _snap_mask_to_edges(binary, edges_full, snap_radius)

    # Step 4: Morphological cleanup
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        # Close small gaps
        snapped = cv2.morphologyEx(snapped, cv2.MORPH_CLOSE, kernel)
        # Remove tiny noise
        snapped = cv2.morphologyEx(snapped, cv2.MORPH_OPEN, kernel)

    # Step 5: Fill internal holes
    if fill_holes:
        snapped = _fill_holes(snapped)

    return snapped


def _snap_mask_to_edges(
    mask: np.ndarray, edges: np.ndarray, snap_radius: int
) -> np.ndarray:
    """Snap mask boundary pixels to the nearest edge within snap_radius.

    Uses distance transform to find the nearest edge pixel for each
    boundary pixel of the mask.
    """
    # Find mask boundary pixels (contour)
    mask_contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not mask_contours:
        return mask

    # Invert edges: 0 = edge, 255 = non-edge
    # Distance transform gives distance to nearest 0-pixel
    edges_inv = 255 - edges
    dist_map = cv2.distanceTransform(edges_inv, cv2.DIST_L2, 5)

    # Create a "snap zone" — pixels within snap_radius of an edge
    snap_zone = (dist_map <= snap_radius).astype(np.uint8)

    # For each contour point, if it's in the snap zone,
    # find the nearest edge and move it there
    result = mask.copy()
    new_contours = []

    for cnt in mask_contours:
        if len(cnt) < 3:
            new_contours.append(cnt)
            continue

        new_cnt = cnt.copy()
        for i in range(len(cnt)):
            px, py = cnt[i][0]

            # Check if this point is near an edge
            if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                if snap_zone[py, px] == 1 and dist_map[py, px] > 0:
                    # Search in a small window for the nearest edge
                    best_dist = float("inf")
                    best_x, best_y = px, py

                    for dy in range(-snap_radius, snap_radius + 1):
                        for dx in range(-snap_radius, snap_radius + 1):
                            nx, ny = px + dx, py + dy
                            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                                if edges[ny, nx] > 0:
                                    d = dx * dx + dy * dy
                                    if d < best_dist:
                                        best_dist = d
                                        best_x, best_y = nx, ny

                    if best_dist < float("inf"):
                        new_cnt[i][0] = [best_x, best_y]

        new_contours.append(new_cnt)

    # Reconstruct mask from snapped contours
    result = np.zeros_like(mask)
    cv2.drawContours(result, new_contours, -1, 1, cv2.FILLED)

    return result


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill internal holes in a binary mask using flood fill."""
    h, w = binary.shape
    # Flood fill from (0, 0) — anything not connected to border is a hole
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    flood_inv = 1 - flood
    return np.maximum(binary, flood_inv)


def detect_edges_in_region(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """Detect Canny edges within a bounding box region.

    Returns edges as a binary image (255 = edge, 0 = non-edge).
    """
    x1, y1, x2, y2 = box
    img_h, img_w = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    edges = cv2.Canny(roi, canny_low, canny_high)

    result = np.zeros((img_h, img_w), dtype=np.uint8)
    result[y1:y2, x1:x2] = edges
    return result
