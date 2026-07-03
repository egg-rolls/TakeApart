"""SAM 2 model wrapper - lazy-loaded singleton."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# SAM 2 / 2.1 model configs (passed to Hydra compose)
# Paths are relative to sam2/configs/ directory
MODEL_CONFIGS = {
    "tiny": "sam2.1/sam2.1_hiera_t",
    "small": "sam2.1/sam2.1_hiera_s",
    "base_plus": "sam2.1/sam2.1_hiera_b+",
    "large": "sam2.1/sam2.1_hiera_l",
}

# Checkpoint filenames to search for (in models/ directory)
MODEL_CHECKPOINTS = [
    "sam2.1_hiera_base_plus.pt",   # SAM 2.1 (preferred)
    "sam2_hiera_base_plus.pt",     # SAM 2.0 fallback
]


class SAMModel:
    """Singleton wrapper for SAM 2 image predictor."""

    _instance: SAMModel | None = None
    _predictor = None
    _model_size: str = "base_plus"

    def __new__(cls, model_size: str = "base_plus") -> SAMModel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model_size = model_size
        return cls._instance

    def _load_model(self) -> None:
        """Load SAM 2 model (called lazily on first use)."""
        if self._predictor is not None:
            return

        logger.info("Loading SAM 2 model (%s)...", self._model_size)

        try:
            import sam2 as sam2_pkg
            from hydra import compose, initialize_config_dir
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "SAM 2 is not installed. Install from GitHub:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                f"Original error: {e}"
            )

        config = MODEL_CONFIGS[self._model_size]
        checkpoint_dir = Path(__file__).parent.parent.parent.parent / "models"

        # Try SAM 2.1 first, then SAM 2.0
        checkpoint = None
        for name in MODEL_CHECKPOINTS:
            candidate = checkpoint_dir / name
            if candidate.exists():
                checkpoint = candidate
                logger.info("Found checkpoint: %s", name)
                break

        if checkpoint is None:
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found in: {checkpoint_dir}\n"
                f"Expected one of: {MODEL_CHECKPOINTS}\n"
                f"Download from: https://github.com/facebookresearch/sam2/releases\n"
                f"Place the .pt file in: {checkpoint_dir}"
            )

        # Initialize Hydra with explicit config directory
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        sam2_configs_dir = str(Path(sam2_pkg.__file__).parent / "configs")
        initialize_config_dir(config_dir=sam2_configs_dir, version_base=None)

        # Load config and build model
        cfg = compose(config_name=config)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

        # Load checkpoint weights (extract 'model' key from checkpoint)
        import torch
        sd = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(sd)

        # Use CUDA if available, otherwise fall back to CPU (slower but works)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("CUDA not available, using CPU (will be slower).")
        model = model.to(device)
        model.eval()

        self._predictor = SAM2ImagePredictor(model)
        logger.info("SAM 2 model loaded successfully.")

    @property
    def predictor(self):
        """Get the SAM2ImagePredictor, loading model if needed."""
        self._load_model()
        return self._predictor

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Set the image for prediction."""
        self.predictor.set_image(image_rgb)

    def predict_with_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        multimask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks from point prompts.

        Args:
            points: (N, 2) array of point coordinates [[x, y], ...]
            labels: (N,) array of point labels (1=foreground, 0=background)
            multimask: whether to return multiple mask candidates

        Returns:
            masks: (K, H, W) boolean array
            scores: (K,) confidence scores
            logits: (K, H, W) mask logits
        """
        return self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask,
        )

    def predict_with_box(
        self,
        box: np.ndarray,
        multimask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks from bounding box prompt.

        Args:
            box: [x1, y1, x2, y2] bounding box
            multimask: whether to return multiple mask candidates

        Returns:
            masks, scores, logits
        """
        return self.predictor.predict(
            box=box,
            multimask_output=multimask,
        )

    def _get_image_size(self) -> tuple[int, int]:
        """Get the original image dimensions (height, width)."""
        # SAM 2 stores original HW as list of tuples: [(h, w)]
        if hasattr(self.predictor, '_orig_hw') and self.predictor._orig_hw:
            return self.predictor._orig_hw[0]
        # Fallback: try to get from the image itself
        if hasattr(self.predictor, '_orig_image') and self.predictor._orig_image is not None:
            img = self.predictor._orig_image
            return img.shape[:2]
        return 1024, 1024

    def predict_auto(self, multimask: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Auto-predict masks using a grid of points over the image."""
        img_h, img_w = self._get_image_size()

        # Use center point as default prompt
        center_x, center_y = img_w // 2, img_h // 2
        points = np.array([[center_x, center_y]])
        labels = np.array([1])

        return self.predict_with_points(points, labels, multimask=multimask)

    def predict_grid(
        self,
        grid_size: int = 8,
        multimask: bool = False,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Predict masks for a grid of points across the image.

        Returns a list of masks and their confidence scores,
        with overlapping masks merged.
        """
        img_h, img_w = self._get_image_size()

        all_masks = []
        all_scores = []

        step_x = img_w // (grid_size + 1)
        step_y = img_h // (grid_size + 1)

        for gy in range(1, grid_size + 1):
            for gx in range(1, grid_size + 1):
                px = gx * step_x
                py = gy * step_y

                points = np.array([[px, py]])
                labels = np.array([1])

                masks, scores, _ = self.predict_with_points(points, labels, multimask=False)

                # Take the best mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(np.uint8)
                score = float(scores[best_idx])

                if np.sum(mask) > 100:  # Filter tiny masks
                    all_masks.append(mask)
                    all_scores.append(score)

        return all_masks, all_scores

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing or model switching)."""
        cls._instance = None
        cls._predictor = None
