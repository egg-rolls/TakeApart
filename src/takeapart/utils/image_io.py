"""Image I/O utilities."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array (OpenCV format)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    bgr = load_image(path)
    if len(bgr.shape) == 2:
        return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    if bgr.shape[2] == 4:
        return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_png(image: np.ndarray, path: str) -> str:
    """Save numpy array as PNG. Returns the absolute path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), image)
    return str(p.resolve())


def save_transparent_png(image_rgb: np.ndarray, mask: np.ndarray, path: str) -> str:
    """Save image with alpha channel from mask. Returns the absolute path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure RGBA
    if image_rgb.shape[2] == 3:
        rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
    else:
        rgba = image_rgb.copy()

    # Set alpha from mask
    rgba[:, :, 3] = (mask * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgba)
    pil_img.save(str(p), "PNG")
    return str(p.resolve())


def image_to_base64(image: np.ndarray, fmt: str = "PNG") -> str:
    """Encode numpy image to base64 string."""
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode="L")
    elif image.shape[2] == 4:
        pil_img = Image.fromarray(image, mode="RGBA")
    else:
        pil_img = Image.fromarray(image, mode="RGB")

    from io import BytesIO

    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_image_info(path: str) -> dict:
    """Get basic image information."""
    img = load_image(path)
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    return {
        "path": str(Path(path).resolve()),
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(img.dtype),
    }
