"""SAM 3D Body processor wrapping Meta's estimator API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SAM3DProcessor:
    """Wraps the SAM 3D Body estimator for single-image full-body mesh recovery.

    The underlying model (``facebook/sam-3d-body-dinov3``) uses an
    encoder-decoder architecture with a DINOv3 backbone and outputs a
    Momentum Human Rig (MHR) mesh that covers body, feet, and hands.

    Usage example::

        processor = SAM3DProcessor.from_pretrained(
            hf_repo_id="facebook/sam-3d-body-dinov3"
        )
        rgb_image = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
        outputs = processor.process_image(rgb_image)
    """

    def __init__(self, estimator: Any, faces: np.ndarray) -> None:
        self._estimator = estimator
        self.faces = faces

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        hf_repo_id: str = "facebook/sam-3d-body-dinov3",
        checkpoint_path: str | None = None,
        mhr_path: str | None = None,
    ) -> "SAM3DProcessor":
        """Load the SAM 3D Body estimator from Hugging Face or local paths.

        Parameters
        ----------
        hf_repo_id:
            Hugging Face repository ID for the checkpoint.  Ignored when
            *checkpoint_path* and *mhr_path* are provided.
        checkpoint_path:
            Local path to ``model.ckpt``.
        mhr_path:
            Local path to ``mhr_model.pt``.
        """
        try:
            from notebook.utils import setup_sam_3d_body  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The SAM 3D Body package is required.  "
                "Clone https://github.com/facebookresearch/sam-3d-body and "
                "follow INSTALL.md, or install via the project's setup script."
            ) from exc

        if checkpoint_path and mhr_path:
            estimator = setup_sam_3d_body(
                checkpoint_path=checkpoint_path,
                mhr_path=mhr_path,
            )
        else:
            estimator = setup_sam_3d_body(hf_repo_id=hf_repo_id)

        return cls(estimator=estimator, faces=estimator.faces)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_rgb: np.ndarray,
        keypoints: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run SAM 3D Body on a single RGB image.

        Parameters
        ----------
        image_rgb:
            HxWx3 uint8 RGB image.
        keypoints:
            Optional Nx3 array of (x, y, confidence) 2-D keypoints used as
            prompts (aligned with MHR joint ordering).
        mask:
            Optional HxW binary segmentation mask (e.g. from SAM 3) used as
            an additional prompt.

        Returns
        -------
        dict with at least:
            ``vertices``  – Vx3 float32 mesh vertices in camera space.
            ``faces``     – Fx3 int32 triangle faces.
            ``joints``    – Jx3 float32 3-D joint positions.
        """
        kwargs: dict[str, Any] = {}
        if keypoints is not None:
            kwargs["keypoints"] = keypoints
        if mask is not None:
            kwargs["mask"] = mask

        outputs = self._estimator.process_one_image(image_rgb, **kwargs)
        outputs.setdefault("faces", self.faces)
        return outputs

    def process_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Convenience wrapper that converts a BGR frame before inference."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.process_image(frame_rgb)

    def run(self, video_source: str | int) -> None:
        """Process every frame from *video_source* and log structured outputs.

        Parameters
        ----------
        video_source:
            Path to a video file or integer camera index.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source!r}")

        logger.info("Processing video source: %s", video_source)
        frame_idx = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                outputs = self.process_frame(frame_bgr)
                logger.debug(
                    "Frame %d: vertices=%s",
                    frame_idx,
                    outputs.get("vertices", np.array([])).shape,
                )
                frame_idx += 1
        finally:
            cap.release()
        logger.info("Processed %d frames.", frame_idx)
