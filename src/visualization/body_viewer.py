"""3D visualisation helpers for SAM 3D Body outputs."""

from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False


class BodyMeshViewer:
    """Visualise a triangulated body mesh using Open3D.

    Parameters
    ----------
    vertices:
        Vx3 float array of mesh vertex positions.
    faces:
        Fx3 int array of triangle face indices.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        if not _OPEN3D_AVAILABLE:
            raise ImportError(
                "open3d is required for BodyMeshViewer.  "
                "Install it with: pip install open3d"
            )
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)

        self._mesh = o3d.geometry.TriangleMesh()
        self._mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        self._mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        self._mesh.compute_vertex_normals()

    def visualize(self) -> None:
        """Open an interactive Open3D window showing the body mesh."""
        o3d.visualization.draw_geometries(
            [self._mesh],
            window_name="SAM 3D Body – Mesh Viewer",
            width=800,
            height=600,
            left=50,
            top=50,
            mesh_show_back_face=True,
        )


class BodyPointCloudViewer:
    """Visualise a body point cloud using Open3D.

    Parameters
    ----------
    point_cloud_data:
        Nx3 float array of 3-D points.
    """

    def __init__(self, point_cloud_data: np.ndarray) -> None:
        if not _OPEN3D_AVAILABLE:
            raise ImportError(
                "open3d is required for BodyPointCloudViewer.  "
                "Install it with: pip install open3d"
            )
        self.point_cloud_data = np.asarray(point_cloud_data, dtype=np.float64)
        self._point_cloud = o3d.geometry.PointCloud()
        self._point_cloud.points = o3d.utility.Vector3dVector(self.point_cloud_data)

    def visualize(self) -> None:
        """Open an interactive Open3D window showing the point cloud."""
        o3d.visualization.draw_geometries(
            [self._point_cloud],
            window_name="SAM 3D Body – Point Cloud Viewer",
            width=800,
            height=600,
            left=50,
            top=50,
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    # Quick smoke-test with random data.
    num_points = 1000
    random_points = np.random.rand(num_points, 3)
    viewer = BodyPointCloudViewer(random_points)
    viewer.visualize()
