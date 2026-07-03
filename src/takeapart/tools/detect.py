"""detect_elements tool — lightweight element detection using Canny + contour (no GPU)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..utils.image_io import load_image_rgb, save_png

logger = logging.getLogger(__name__)

# Color palette for visualization boxes
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
]


def detect_elements(
    image_path: str,
    output_path: str = "",
    min_area_ratio: float = 0.002,
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
) -> dict:
    """Detect elements in an image using Canny edge detection + contour analysis.

    Pure traditional CV — no GPU, no deep learning, fast and lightweight.
    Designed as a fallback for AIs without visual understanding capability.

    Good at finding: discrete objects on clean backgrounds (PPT elements, icons,
    text blocks, photos with clear subject-background separation).

    Args:
        image_path: path to the input image
        output_path: if provided, save annotated visualization to this path
        min_area_ratio: minimum area as fraction of image (filters noise)
        blur_kernel: Gaussian blur kernel size (odd number, 0=disable)
        canny_low: Canny edge detection low threshold
        canny_high: Canny edge detection high threshold

    Returns:
        dict with image_info, elements list, total_count, optional visualization_path
    """
    logger.info("Detecting elements in: %s (Canny+contour)", image_path)

    image_rgb = load_image_rgb(image_path)
    h, w = image_rgb.shape[:2]
    img_area = h * w

    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Gaussian blur to reduce noise
    if blur_kernel > 0:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        blurred = gray

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Dilate edges to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours
    elements = []
    min_area = img_area * min_area_ratio

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / max(bh, 1)

        # Simplified contour for output
        epsilon = 2.0
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        if len(points) < 3:
            continue

        elements.append({
            "bbox": {"x": x, "y": y, "width": bw, "height": bh},
            "area_ratio": round(area / img_area, 4),
            "area_pixels": int(area),
            "contour_points": points[:20],  # Limit for JSON size
            "aspect_ratio": round(aspect_ratio, 2),
        })

    # Sort by area (largest first)
    elements.sort(key=lambda e: e["area_ratio"], reverse=True)

    # Assign IDs
    for i, elem in enumerate(elements):
        elem["id"] = i

    result = {
        "image_info": {"path": image_path, "width": w, "height": h},
        "elements": elements,
        "total_count": len(elements),
    }

    # Visualization
    if output_path:
        vis_path = _save_visualization(image_path, image_rgb, elements, output_path)
        result["visualization_path"] = vis_path

    return result


def _save_visualization(
    image_path: str, image_rgb: np.ndarray, elements: list[dict], output_path: str
) -> str:
    """Draw bounding boxes on the image and save."""
    vis = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for elem in elements:
        bbox = elem["bbox"]
        x, y = bbox["x"], bbox["y"]
        w, h = bbox["width"], bbox["height"]
        color = _COLORS[elem["id"] % len(_COLORS)]

        # Rectangle
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Label
        label = f"#{elem['id']} {elem['area_ratio']:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        label_y = max(y - 4, th + 4)
        cv2.rectangle(vis, (x, label_y - th - 4), (x + tw + 6, label_y + 2), color, -1)
        cv2.putText(vis, label, (x + 3, label_y - 2), font, 0.5, (0, 0, 0), 1)

    # Save
    if not output_path:
        stem = Path(image_path).stem
        output_dir = Path(__file__).parent.parent.parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{stem}_detected.png")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    save_png(vis, output_path)
    return output_path
