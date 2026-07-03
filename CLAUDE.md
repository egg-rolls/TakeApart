# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TakeApart is an MCP (Model Context Protocol) Server that gives AI models "PS-level eyes and hands" for image element extraction. It provides pixel-precise element detection and segmentation for design drafts, posters, UI screenshots, and PPT images.

## Commands

```bash
# Install project dependencies
pip install -e .

# Install SAM 2 (required for segment tool, from GitHub source)
pip install git+https://github.com/facebookresearch/sam2.git

# Download SAM 2.1 model checkpoint (~300MB) → place in models/
# https://github.com/facebookresearch/sam2/releases

# Run MCP server (stdio transport)
python -m takeapart.server

# Verify installation
python -c "from takeapart.server import create_server; print('OK')"
```

## Architecture

Two MCP tools with different compute profiles:

- **`detect_elements`** — Pure traditional CV (Canny + contour + morphological text detection). No GPU needed. Two-pass: Pass1 finds shapes/icons via edge detection, Pass2 finds text blocks via adaptive threshold + dilation. Returns bbox, type, area, font_size_px for text.

- **`segment_element`** — SAM 2 deep learning + Canny edge refinement. Requires GPU (falls back to CPU). Takes bounding box prompts, returns hard-edge masks or transparent PNGs. Supports batch processing via `boxes` array.

### Key modules

- `server.py` — MCP Server entry point, tool registration, stdio transport. All tool schemas and handlers defined here.
- `tools/detect.py` — Two-pass CV detection (shapes + text). `_detect_text_blocks()` uses morphological ops to connect characters into text blocks.
- `tools/segment.py` — SAM segmentation with Canny edge refinement. `_segment_single()` handles per-box logic.
- `models/sam_model.py` — SAM 2.1 singleton wrapper. Lazy-loads on first use. Uses `initialize_config_dir` for Hydra config resolution (SAM 2.1 config path: `sam2.1/sam2.1_hiera_b+`). Checkpoint search order: `sam2.1_hiera_base_plus.pt` then `sam2_hiera_base_plus.pt`.
- `utils/edge_refine.py` — `refine_mask_with_edges()` snaps SAM mask boundaries to nearest Canny edges using distance transform, producing PS-like hard edges.
- `utils/mask_ops.py` — Mask utilities: bbox extraction, contour simplification, hole filling, overlap merging.
- `utils/image_io.py` — Image I/O: load (BGR/RGB), save PNG, transparent PNG export, base64 encoding.

### Important implementation details

- SAM logits are at internal resolution (256×256), NOT the original image size. Always use the `masks` array (already upscaled) for mask operations, not `logits`.
- SAM config uses Hydra's `compose()` which requires `GlobalHydra.instance().clear()` + `initialize_config_dir()` before each call.
- Model weights path: `models/sam2.1_hiera_base_plus.pt` (relative to project root).
- CPU fallback: when CUDA unavailable, SAM runs on CPU (slower but functional).

## Recommended workflow (3-phase)

1. **CV detect** → precise element positions/sizes/types (no GPU, <1s)
2. **Vision AI** → content/fonts/colors/semantic description (complements CV)
3. **SAM segment** → pixel-precise extraction of selected elements (GPU, ~2s/box)

Conflict resolution: position/size from CV, content/semantics from vision AI.

## Dependencies

Python 3.10+, PyTorch 2.3+, OpenCV 4.9+, MCP SDK 1.0+, Pillow, NumPy. SAM 2 installed from GitHub (not PyPI).

## Output directory

All extracted files go to `output/` (project root). Subdirectories created as needed.
