import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# Function to load point cloud data from a binary file
def load_point_cloud_binary(bin_file):
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
        ('ring', np.uint16),
        ('time', np.float32)
    ])

    with open(bin_file, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)

    point_cloud = np.vstack((data['x'], data['y'], data['z'])).T
    return point_cloud

# Function to save image from point cloud data
def save_point_cloud_image(point_cloud, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Order array by Z value (column 3)
    sorted_z = point_cloud[np.argsort(point_cloud[:, 2])[::-1]]
    rows = len(point_cloud)

    pcd.normalize_normals()

    # Define the threshold for z-coordinate to switch to grey color
    grey_threshold = -1.91

    # When Z values are negative, this if else statement switches the min and max
    if sorted_z[0][2] < sorted_z[rows - 1][2]:
        min_z_val = sorted_z[0][2]
        max_z_val = sorted_z[rows - 1][2]
    else:
        max_z_val = sorted_z[0][2]
        min_z_val = sorted_z[rows - 1][2]

    # Assign colors to the point cloud
    cmap_norm = mpl.colors.Normalize(vmin=min_z_val, vmax=max_z_val)
    point_colors = plt.get_cmap('rainbow')(cmap_norm(point_cloud[:, -1]))[:, 0:3]

    # Set points with z-coordinates less than -2.41 to grey
    point_colors[point_cloud[:, 2] < grey_threshold] = [0.5, 0.5, 0.5]  # Grey color


    
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    ctr = vis.get_view_control()

    # Add point cloud geometry
    vis.add_geometry(pcd)

    # Set camera parameters
    ctr.set_lookat([-0.7, 0, 0.3])  # Set the camera look-at point
    ctr.set_up([0, 0, 1])           # Set the up direction of the camera
    ctr.set_front([0, -1, 0])       # Set the front direction of the camera
    ctr.set_zoom(0.005)             # Set the zoom level of the camera

    # Update and render the visualization
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    vis.capture_screen_image(save_path)

    # Close the visualization window
    vis.destroy_window()

# Main function
if __name__ == '__main__':
    # Directory containing point cloud binary files
    bin_dir = "/home/kedhar/workspace/catkin_ws/src/scripts/point_cloud_bins"
    image_dir = '/home/kedhar/workspace/catkin_ws/src/scripts/point_cloud_images'
    bin_files = sorted(os.listdir(bin_dir))

    # Process each binary file in the directory
    for idx, bin_file in enumerate(bin_files):
        if bin_file.endswith(".bin"):
            bin_path = os.path.join(bin_dir, bin_file)

            # Load point cloud data from the binary file
            point_cloud = load_point_cloud_binary(bin_path)

            # Generate image file name based on the current timestamp
            
            image_file = f"point_cloud_image_{idx}.png"
            image_path = os.path.join(image_dir, image_file)

            # Save image from the point cloud data
            save_point_cloud_image(point_cloud, image_path)

            print(f"Image saved: {image_path}")
