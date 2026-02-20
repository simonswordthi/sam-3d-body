import logging
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


class SAM3DProcessor:
    """
    Processes video frames using Meta's Segment Anything Model (SAM) to extract
    body segments and derive a 3D point cloud for each frame.

    If the ``segment_anything`` package or model weights are unavailable the
    processor falls back to OpenCV's built-in HOG person detector and generates
    a simulated skeleton point cloud so the application remains functional.
    """

    def __init__(self, log_callback: Optional[Callable[[str], None]] = None) -> None:
        self.log_callback = log_callback
        self.mask_generator = None
        self._sam_available = False
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._initialize()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        logging.info(message)
        if self.log_callback:
            self.log_callback(message)

    def _initialize(self) -> None:
        self._log("Initializing SAM 3D Body processor…")
        try:
            import segment_anything  # noqa: F401 – presence check only
            self._sam_available = True
            self._log(
                "SAM library detected. Call load_sam_weights() with a checkpoint "
                "file (e.g. sam_vit_h_4b8939.pth) to enable full SAM inference."
            )
        except ImportError:
            self._log(
                "segment-anything package not found -- running in simulation mode "
                "(OpenCV HOG detector + synthetic skeleton points)."
            )
        self._log("OpenCV HOG person detector initialised as fallback.")
        self._log("SAM 3D Body processor ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def load_sam_weights(self, checkpoint_path: str, model_type: str = "vit_h") -> bool:
        """Load SAM model weights from *checkpoint_path*."""
        if not self._sam_available:
            self._log("SAM library not available – cannot load weights.")
            return False
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            self._log(f"Loading SAM model ({model_type}) from {checkpoint_path}…")
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
            self._log("SAM model weights loaded successfully.")
            return True
        except Exception as exc:
            self._log(f"Failed to load SAM weights: {exc}")
            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single BGR video frame.

        Returns
        -------
        annotated_frame : np.ndarray
            A copy of *frame* with detections drawn on it.
        point_cloud : np.ndarray, shape (N, 3)
            Detected / simulated 3-D body points in normalised coordinates.
        """
        annotated = frame.copy()
        if self.mask_generator is not None:
            point_cloud, annotated = self._process_with_sam(frame, annotated)
        else:
            point_cloud, annotated = self._process_with_hog(frame, annotated)
        return annotated, point_cloud

    def run(
        self,
        video_path: str,
        frame_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Process every sampled frame of *video_path*.

        Parameters
        ----------
        video_path : str
            Path to the input video file.
        frame_callback : callable(annotated_frame, point_cloud), optional
            Called after each processed frame.
        progress_callback : callable(current_frame, total_frames), optional
            Called after every frame read (processed or skipped).

        Returns
        -------
        list of (annotated_frame, point_cloud) tuples
        """
        self._log(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"ERROR: Cannot open video file: {video_path}")
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._log(f"Video: {total} frames at {fps:.1f} fps")

        # Sample ~2 frames per second of video to keep processing fast
        step = max(1, int(fps // 2))
        results: List[Tuple[np.ndarray, np.ndarray]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                self._log(f"Processing frame {frame_idx}/{total}…")
                annotated, pts = self.process_frame(frame)
                results.append((annotated, pts))
                if frame_callback:
                    frame_callback(annotated, pts)
            if progress_callback:
                progress_callback(frame_idx, max(total, 1))
            frame_idx += 1

        cap.release()
        self._log(f"Done. Sampled {len(results)} frames from {frame_idx} total.")
        return results

    # ── Private processing back-ends ──────────────────────────────────────────

    def _process_with_sam(
        self, frame: np.ndarray, annotated: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use SAM automatic mask generation to build a 3-D point cloud."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(rgb)
        self._log(f"  SAM generated {len(masks)} mask(s).")

        h, w = frame.shape[:2]
        points: List[np.ndarray] = []

        for mask_data in masks[:10]:  # cap at 10 masks per frame
            mask: np.ndarray = mask_data["segmentation"]
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            depth = 1.0 - float(mask_data.get("predicted_iou", 0.5))
            pts = np.column_stack(
                [
                    xs / w * 2 - 1,
                    -(ys / h * 2 - 1),
                    np.full(len(xs), depth, dtype=np.float32),
                ]
            ).astype(np.float32)
            points.append(pts)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            colour = tuple(int(c) for c in np.random.randint(80, 255, 3).tolist())
            cv2.drawContours(annotated, contours, -1, colour, 2)

        if points:
            point_cloud = np.vstack(points).astype(np.float32)
        else:
            point_cloud = np.empty((0, 3), dtype=np.float32)
        return point_cloud, annotated

    def _process_with_hog(
        self, frame: np.ndarray, annotated: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use HOG person detector and generate a simulated skeleton point cloud."""
        h, w = frame.shape[:2]
        rects, _ = self._hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        self._log(f"  HOG detected {len(rects)} person(s).")

        points: List[np.ndarray] = []
        for x, y, bw, bh in rects:
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            points.append(self._simulate_body_points(x, y, bw, bh, w, h))

        if not points:
            point_cloud = self._placeholder_body()
        else:
            point_cloud = np.vstack(points).astype(np.float32)
        return point_cloud, annotated

    # ── Point-cloud helpers ───────────────────────────────────────────────────

    @staticmethod
    def _simulate_body_points(
        x: int, y: int, bw: int, bh: int, fw: int, fh: int
    ) -> np.ndarray:
        """Derive a skeleton + surface point cloud from a person bounding box."""
        cx = (x + bw / 2) / fw * 2 - 1
        cy = -((y + bh / 2) / fh * 2 - 1)
        sx = bw / fw
        sy = bh / fh

        # 14 approximate skeleton landmarks
        landmarks = np.array(
            [
                [cx,           cy + sy * 0.50,  0.0 ],  # head
                [cx,           cy + sy * 0.30,  0.0 ],  # neck
                [cx - sx * 0.3, cy + sy * 0.10, 0.1 ],  # left shoulder
                [cx + sx * 0.3, cy + sy * 0.10, 0.1 ],  # right shoulder
                [cx - sx * 0.4, cy - sy * 0.10, 0.15],  # left elbow
                [cx + sx * 0.4, cy - sy * 0.10, 0.15],  # right elbow
                [cx - sx * 0.4, cy - sy * 0.30, 0.1 ],  # left wrist
                [cx + sx * 0.4, cy - sy * 0.30, 0.1 ],  # right wrist
                [cx - sx * 0.15, cy - sy * 0.10, 0.05], # left hip
                [cx + sx * 0.15, cy - sy * 0.10, 0.05], # right hip
                [cx - sx * 0.2,  cy - sy * 0.35, 0.08], # left knee
                [cx + sx * 0.2,  cy - sy * 0.35, 0.08], # right knee
                [cx - sx * 0.2,  cy - sy * 0.50, 0.05], # left ankle
                [cx + sx * 0.2,  cy - sy * 0.50, 0.05], # right ankle
            ],
            dtype=np.float32,
        )

        # Add scattered surface points around the skeleton
        rng = np.random.default_rng(seed=int(abs(cx * 1000)))
        noise = rng.standard_normal((60, 3)).astype(np.float32) * np.array(
            [sx * 0.12, sy * 0.12, 0.04], dtype=np.float32
        )
        centres = landmarks[rng.integers(0, len(landmarks), 60)]
        return np.vstack([landmarks, centres + noise])

    @staticmethod
    def _placeholder_body() -> np.ndarray:
        """Return a standing stick-figure point cloud for when no person is detected."""
        return np.array(
            [
                [ 0.0,   0.90, 0.0],  # head
                [ 0.0,   0.70, 0.0],  # neck
                [-0.30,  0.50, 0.0],  # left shoulder
                [ 0.30,  0.50, 0.0],  # right shoulder
                [-0.40,  0.20, 0.0],  # left elbow
                [ 0.40,  0.20, 0.0],  # right elbow
                [-0.40, -0.10, 0.0],  # left wrist
                [ 0.40, -0.10, 0.0],  # right wrist
                [-0.15,  0.00, 0.0],  # left hip
                [ 0.15,  0.00, 0.0],  # right hip
                [-0.20, -0.45, 0.0],  # left knee
                [ 0.20, -0.45, 0.0],  # right knee
                [-0.20, -0.90, 0.0],  # left ankle
                [ 0.20, -0.90, 0.0],  # right ankle
            ],
            dtype=np.float32,
        )