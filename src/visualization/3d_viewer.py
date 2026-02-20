import open3d as o3d
import numpy as np

class BodyPointCloudViewer:
    def __init__(self, point_cloud_data):
        self.point_cloud_data = point_cloud_data
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
        
    def visualize(self):
        o3d.visualization.draw_geometries([self.point_cloud],
                                            window_name='3D Body Point Cloud Viewer',
                                            width=800, height=600,
                                            left=50, top=50,
                                            mesh_show_back_face=True)

if __name__ == '__main__':
    # Example usage
    # Generate some random point cloud data for demonstration
    num_points = 1000
    random_points = np.random.rand(num_points, 3)

    viewer = BodyPointCloudViewer(random_points)
    viewer.visualize()