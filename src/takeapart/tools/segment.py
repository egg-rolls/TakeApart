"""segment_element tool — precise segmentation with SAM + Canny edge refinement."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..models.sam_model import SAMModel
from ..utils.image_io import load_image_rgb, save_transparent_png, image_to_base64
from ..utils.mask_ops import mask_to_bbox, mask_area_ratio, crop_with_mask
from ..utils.edge_refine import refine_mask_with_edges

logger = logging.getLogger(__name__)


def segment_element(
    image_path: str,
    boxes: list[dict],
    output_mode: str = "png",
    output_dir: str = "",
    padding: int = 10,
    crop: bool = True,
    edge_refine: bool = True,
) -> dict:
    """Precisely segment elements using SAM + Canny edge refinement.

    PS-like selection: draws a box around the target area, SAM generates
    an initial mask, then Canny edge snapping refines the mask boundary
    to produce clean, hard edges.

    Supports batch processing — pass multiple boxes for parallel extraction.

    Args:
        image_path: path to the input image
        boxes: list of bounding boxes, each with:
            - x1, y1, x2, y2: bounding box coordinates (required)
            - label: element label/description (optional)
        output_mode: "mask" | "png" | "both"
            - "mask": return base64-encoded mask only
            - "png": export transparent PNG file only
            - "both": return mask base64 + PNG file path
        output_dir: output directory for PNG files (defaults to output/)
        padding: pixels of padding around extracted elements (PNG mode)
        crop: whether to crop to element bounds (PNG mode)
        edge_refine: whether to apply Canny edge refinement (disable for speed)

    Returns:
        dict with image_info, results list, total_count
    """
    logger.info("Segmenting %d elements from: %s (mode=%s, edge_refine=%s)",
                len(boxes), image_path, output_mode, edge_refine)

    image_rgb = load_image_rgb(image_path)
    h, w = image_rgb.shape[:2]

    # Output directory
    if not output_dir:
        output_dir = str(Path(__file__).parent.parent.parent.parent / "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem

    # Initialize SAM and set image once
    sam = SAMModel()
    sam.set_image(image_rgb)

    results = []

    for idx, box_def in enumerate(boxes):
        x1 = box_def.get("x1", 0)
        y1 = box_def.get("y1", 0)
        x2 = box_def.get("x2", w)
        y2 = box_def.get("y2", h)
        label = box_def.get("label", f"element_{idx}")

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        logger.info("  [%d/%d] Box: (%d,%d)-(%d,%d) label=%s",
                     idx + 1, len(boxes), x1, y1, x2, y2, label)

        try:
            result = _segment_single(
                sam, image_rgb, (x1, y1, x2, y2), label, idx,
                stem, output_dir, output_mode, padding, crop, edge_refine,
            )
            results.append(result)
        except Exception as e:
            logger.error("  Failed: %s", e)
            results.append({
                "index": idx,
                "label": label,
                "input_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "error": str(e),
            })

    return {
        "image_info": {"path": image_path, "width": w, "height": h},
        "results": results,
        "total_count": len(results),
    }


def _segment_single(
    sam: SAMModel,
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    index: int,
    stem: str,
    output_dir: str,
    output_mode: str,
    padding: int,
    crop: bool,
    edge_refine: bool,
) -> dict:
    """Segment a single element from a bounding box."""
    x1, y1, x2, y2 = box
    h, w = image_rgb.shape[:2]

    # SAM prediction
    box_arr = np.array([x1, y1, x2, y2])
    masks, scores, logits = sam.predict_with_box(box_arr, multimask=False)

    best_idx = np.argmax(scores)
    score = float(scores[best_idx])

    # Edge refinement
    if edge_refine:
        mask = refine_mask_with_edges(
            masks[best_idx],
            logits=logits[best_idx],
            image_rgb=image_rgb,
            box=(x1, y1, x2, y2),
            prob_threshold=0.5,
            canny_low=50,
            canny_high=150,
            snap_radius=5,
            morph_kernel_size=3,
            fill_holes=True,
        )
    else:
        # Use SAM mask directly (already upscaled to image size)
        mask = masks[best_idx].astype(np.uint8)

    # Compute metadata
    bbox = mask_to_bbox(mask)
    area_ratio = mask_area_ratio(mask)

    result = {
        "index": index,
        "label": label,
        "input_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "bbox": bbox,
        "area_ratio": round(area_ratio, 4),
        "confidence": round(score, 3),
    }

    # Output based on mode
    if output_mode in ("mask", "both"):
        mask_vis = (mask * 255).astype(np.uint8)
        result["mask_base64"] = image_to_base64(mask_vis, fmt="PNG")

    if output_mode in ("png", "both"):
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        out_path = str(Path(output_dir) / f"{stem}_{index:02d}_{safe_label}.png")

        if crop:
            cropped_img, cropped_mask = crop_with_mask(image_rgb, mask, padding=padding)
            save_transparent_png(cropped_img, cropped_mask, out_path)
            extracted_h, extracted_w = cropped_img.shape[:2]
        else:
            save_transparent_png(image_rgb, mask, out_path)
            extracted_h, extracted_w = h, w

        result["output_path"] = out_path
        result["extracted_size"] = {"width": extracted_w, "height": extracted_h}

    return result
