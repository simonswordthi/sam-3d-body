"""
viewer_3d.py – Open3D-based 3-D point-cloud viewer for SAM body meshes.

The visualiser runs in a dedicated daemon thread so the PyQt6 main loop is
never blocked.
"""

import threading
from typing import Optional

import numpy as np

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:  # pragma: no cover -- open3d optional
    _O3D_AVAILABLE = False


class Viewer3D:
    """
    Manages an Open3D visualisation window that displays body point clouds.

    Usage
    -----
    viewer = Viewer3D()
    viewer.show(point_cloud)   # opens or refreshes the window
    """

    def __init__(self) -> None:
        self._vis: Optional[object] = None  # o3d.visualization.Visualizer
        self._pcd: Optional[object] = None  # o3d.geometry.PointCloud
        self._lock = threading.Lock()
        self._running = False
        self._pending: Optional[np.ndarray] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def show(self, point_cloud: np.ndarray) -> None:
        """Open (or refresh) the 3-D viewer with *point_cloud*."""
        if not _O3D_AVAILABLE:
            print("[Viewer3D] open3d is not installed – cannot display 3-D scene.")
            return
        if len(point_cloud) == 0:
            return

        with self._lock:
            self._pending = point_cloud.copy()

        if self._running:
            # The background thread picks up _pending on next poll iteration
            return

        thread = threading.Thread(target=self._viewer_loop, daemon=True)
        thread.start()

    # ── Private ───────────────────────────────────────────────────────────────

    def _viewer_loop(self) -> None:
        """Run the Open3D visualiser event loop (called in background thread)."""
        self._running = True
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="SAM 3D Body Viewer", width=800, height=600)

            with self._lock:
                data = self._pending.copy() if self._pending is not None else np.empty((0, 3))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(self._height_colormap(data))
            vis.add_geometry(pcd)

            # Reference coordinate frame
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            vis.add_geometry(frame_mesh)

            render_opt = vis.get_render_option()
            render_opt.background_color = np.array([0.08, 0.08, 0.12])
            render_opt.point_size = 5.0

            self._vis = vis
            self._pcd = pcd

            while vis.poll_events():
                vis.update_renderer()

                with self._lock:
                    pending = self._pending
                    self._pending = None

                if pending is not None:
                    pcd.points = o3d.utility.Vector3dVector(pending.astype(np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(
                        self._height_colormap(pending)
                    )
                    vis.update_geometry(pcd)

            vis.destroy_window()
        finally:
            self._running = False
            self._vis = None
            self._pcd = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _height_colormap(pts: np.ndarray) -> np.ndarray:
        """Colour points from blue (bottom) → red (top) by Y-coordinate."""
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
        y = pts[:, 1]
        y_min, y_max = float(y.min()), float(y.max())
        t = np.zeros(len(pts), dtype=np.float64) if y_max == y_min else (y - y_min) / (y_max - y_min)
        return np.column_stack([t, np.zeros_like(t), 1.0 - t])
