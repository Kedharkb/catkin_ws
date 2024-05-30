import open3d as o3d


def visualize_point_cloud_ply(file_path):
    # Read the point cloud from the .ply file
    pcd = o3d.io.read_point_cloud(file_path)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")

# Example usage:
ply_file_path = "./sensor_data_raw.ply"
visualize_point_cloud_ply(ply_file_path)
