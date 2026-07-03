"""detect_elements tool — smart element detection (icons, text, shapes) using CV."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..utils.image_io import load_image_rgb, save_png

logger = logging.getLogger(__name__)

_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
]


def detect_elements(
    image_path: str,
    output_path: str = "",
    min_area_ratio: float = 0.001,
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
) -> dict:
    """Detect all elements in an image: icons, text blocks, shapes, etc.

    Uses two-pass detection:
      Pass 1 (Canny + contour): finds discrete shapes — icons, buttons, images
      Pass 2 (morphological): finds text blocks — paragraphs, lines, labels

    Results are merged, deduplicated, and classified by type.

    Args:
        image_path: path to the input image
        output_path: if provided, save annotated visualization
        min_area_ratio: minimum area as fraction of image
        blur_kernel: Gaussian blur kernel size (odd, 0=disable)
        canny_low / canny_high: Canny thresholds
    """
    logger.info("Detecting elements in: %s", image_path)

    image_rgb = load_image_rgb(image_path)
    h, w = image_rgb.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    if blur_kernel > 0:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        blurred = gray

    # ─── Pass 1: Canny + contour (shapes/icons) ────────────────
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel_close, iterations=1)

    contours_shape, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ─── Pass 2: Morphological text detection ──────────────────
    contours_text = _detect_text_blocks(blurred, w, h)

    # ─── Merge & classify ──────────────────────────────────────
    min_area = img_area * min_area_ratio
    elements = []
    used_boxes = []  # for dedup

    for cnt in contours_shape:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bbox, el_type, aspect = _classify_contour(cnt, w, h)
        if el_type == "text":
            continue  # text will be handled by pass 2
        if _is_duplicate(bbox, used_boxes):
            continue
        used_boxes.append(bbox)
        points = _simplify_contour(cnt)
        elements.append(_make_element(bbox, area, img_area, points, aspect, el_type))

    for cnt in contours_text:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bbox, _, aspect = _classify_contour(cnt, w, h)
        if _is_duplicate(bbox, used_boxes):
            continue
        used_boxes.append(bbox)
        points = _simplify_contour(cnt)
        elements.append(_make_element(bbox, area, img_area, points, aspect, "text"))

    # Sort by area
    elements.sort(key=lambda e: e["area_ratio"], reverse=True)
    for i, elem in enumerate(elements):
        elem["id"] = i

    result = {
        "image_info": {"path": image_path, "width": w, "height": h},
        "elements": elements,
        "total_count": len(elements),
    }

    if output_path:
        vis_path = _save_visualization(image_path, image_rgb, elements, output_path)
        result["visualization_path"] = vis_path

    return result


def _detect_text_blocks(gray: np.ndarray, w: int, h: int) -> list:
    """Detect text blocks using morphological operations.

    Strategy: adaptive threshold → dilate to connect chars into lines
    → dilate more to connect lines into paragraphs → find contours.
    """
    # Adaptive threshold (handles varying backgrounds)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )

    # Connect characters into lines (horizontal dilation)
    kern_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    dilated_h = cv2.dilate(thresh, kern_h, iterations=1)

    # Connect lines into text blocks (vertical dilation, smaller)
    kern_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    dilated = cv2.dilate(dilated_h, kern_v, iterations=1)

    # Close small gaps
    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kern_close)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _classify_contour(
    cnt: np.ndarray, img_w: int, img_h: int
) -> tuple[dict, str, float]:
    """Classify a contour as text/icon/shape based on geometry."""
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect = bw / max(bh, 1)
    bbox = {"x": x, "y": y, "width": bw, "height": bh}

    # Text: wide and short (line) or tall and narrow (column)
    # Typical text line: aspect > 3, height 10-80px relative to 1080p
    rel_height = bh / img_h
    if aspect > 3.0 and 0.01 < rel_height < 0.15:
        return bbox, "text", round(aspect, 2)
    if aspect < 0.33 and 0.01 < (bw / img_w) < 0.15:
        return bbox, "text", round(aspect, 2)

    # Icon: roughly square, small area
    if 0.4 < aspect < 2.5 and rel_height < 0.25:
        return bbox, "icon", round(aspect, 2)

    # Banner: very wide
    if aspect > 4.0 and rel_height < 0.2:
        return bbox, "banner", round(aspect, 2)

    return bbox, "element", round(aspect, 2)


def _is_duplicate(bbox: dict, used: list, iou_thresh: float = 0.5) -> bool:
    """Check if bbox overlaps significantly with any already-used box."""
    for u in used:
        ix = max(0, min(bbox["x"] + bbox["width"], u["x"] + u["width"]) - max(bbox["x"], u["x"]))
        iy = max(0, min(bbox["y"] + bbox["height"], u["y"] + u["height"]) - max(bbox["y"], u["y"]))
        inter = ix * iy
        union = bbox["width"] * bbox["height"] + u["width"] * u["height"] - inter
        if union > 0 and inter / union > iou_thresh:
            return True
    return False


def _simplify_contour(cnt: np.ndarray, max_points: int = 20) -> list:
    """Simplify contour to a manageable number of points."""
    epsilon = 2.0
    approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
    points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
    return points[:max_points]


def _make_element(
    bbox: dict, area: float, img_area: float,
    contour_points: list, aspect: float, el_type: str,
) -> dict:
    return {
        "bbox": bbox,
        "type": el_type,
        "area_ratio": round(area / img_area, 4),
        "area_pixels": int(area),
        "contour_points": contour_points,
        "aspect_ratio": aspect,
    }


def _save_visualization(
    image_path: str, image_rgb: np.ndarray, elements: list[dict], output_path: str
) -> str:
    vis = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for elem in elements:
        bbox = elem["bbox"]
        x, y = bbox["x"], bbox["y"]
        w, h = bbox["width"], bbox["height"]
        color = _COLORS[elem["id"] % len(_COLORS)]
        thickness = 2 if elem["type"] != "text" else 1

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        label = f"#{elem['id']} {elem['type']} {elem['area_ratio']:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
        label_y = max(y - 4, th + 4)
        cv2.rectangle(vis, (x, label_y - th - 4), (x + tw + 6, label_y + 2), color, -1)
        cv2.putText(vis, label, (x + 3, label_y - 2), font, 0.4, (0, 0, 0), 1)

    if not output_path:
        stem = Path(image_path).stem
        output_dir = Path(__file__).parent.parent.parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{stem}_detected.png")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    save_png(vis, output_path)
    return output_path
