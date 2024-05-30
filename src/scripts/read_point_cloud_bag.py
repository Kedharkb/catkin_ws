import os
from datetime import datetime

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image


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
    print(data)
    point_cloud = np.vstack((data['x'], data['y'], data['z'])).T
    np.savetxt('./pointcloud.txt', point_cloud)
    return point_cloud


def load_point_cloud_from_bin(filename):
    # Load the points from the binary file into a NumPy array
    with open(filename, 'rb') as f:
        points = np.fromfile(f, dtype=np.float32)

    # Reshape the points array to the correct shape (number of points, 3)
    points = points.reshape(-1, 3)

    return points


def print_camera_parameters(vis):
    # Get the camera intrinsic parameters
    viewport = vis.get_view_control().convert_to_pinhole_camera_parameters()
    print("Camera intrinsic parameters:")
    print(viewport.intrinsic)

    # Get the camera extrinsic parameters
    extrinsic_matrix = viewport.extrinsic
    print("Extrinsic matrix:")
    print(extrinsic_matrix)

    # Extract look-at point, up direction, and front direction from extrinsic matrix
    # The look-at point is the translation part of the extrinsic matrix
    lookat = extrinsic_matrix[:3, 3]
    
    # The up direction is the second column of the rotation part of the extrinsic matrix
    up = extrinsic_matrix[:3, 1]

    # The front direction is the negative third column of the rotation part of the extrinsic matrix
    front = -extrinsic_matrix[:3, 2]

    print("Look-at point:", lookat)
    print("Up direction:", up)
    print("Front direction:", front)
    
    camera_position = np.array([0, 0, 0, 1])  # Camera position in homogeneous coordinates
    zoom = np.linalg.norm(lookat - np.dot(extrinsic_matrix, camera_position)[:3])  # Extract only the 3D coordinates
    print("Zoom level:", zoom)



# Function to save image from point cloud data
def save_point_cloud_image(point_cloud, image_path, save_path):
    # Define the threshold for z-coordinate to switch to grey color
    # Sort points by z-value

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    print("Minimum z-coordinate value:", min_z,max_z)
    print(point_cloud)

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
    print(point_colors)
    # Set points with z-coordinates less than -2.41 to grey
    point_colors[point_cloud[:, 2] < grey_threshold] = [0.5, 0.5, 0.5]  # Grey color
   

    
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # vis = o3d.visualization.Visualizer()
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(visible=True)
    # ctr = vis.get_view_control()

    # Add point cloud geometry
    vis.add_geometry(pcd)
    # vis.register_animation_callback(print_camera_parameters)
    vis.run() 
    viewport = vis.get_view_control().convert_to_pinhole_camera_parameters()
    print("Camera intrinsic parameters:")
    print(viewport.intrinsic)

    # # Set camera parameters
    # ctr.set_lookat([-0.7, 0, 0.3])  # Set the camera look-at point
    # ctr.set_up([0, 0, 1])           # Set the up direction of the camera
    # ctr.set_front([0, -1, 0])       # Set the front direction of the camera
    # ctr.set_zoom(0.005)             # Set the zoom level of the camera

    while vis.poll_events():
            vis.update_renderer()



def render_point_cloud(point_cloud):
    image_width = 1920
    image_height = 1080
    K = np.array([[843.3386731699313, 0.0, 960.5],
                [0.0, 843.3386731699313, 540.5],
                [0.0, 0.0, 1.0]])

    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    point_cloud = point_cloud.astype(np.float64)

    projected_points, _ = cv2.projectPoints(point_cloud, np.eye(3), np.zeros((3, 1)), K, D)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)  
    print(projected_points)
    color = (0, 255, 0)  # Green color
    for i in range(len(projected_points)):
        x, y = projected_points[i, 0]
        # Check if coordinates are within image bounds
        if 0 <= x < 1920 and 0 <= y < 1080:
            # Convert coordinates to integers
            x_int = int(round(x))
            y_int = int(round(y))
            cv2.circle(image, (x_int, y_int), radius=1, color=color, thickness=-1)
    cv2.imshow("Rendered Point Cloud", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

# Main function
if __name__ == '__main__':

    bin_path = './transformed_point_cloud.bin'

    # Load point cloud data from the binary file
    point_cloud = load_point_cloud_from_bin(bin_path)
    save_point_cloud_image(point_cloud,'','')
    # resp = save_point_cloud_image(point_cloud,'./camera_images/image_20240405_171335_499705_178.jpg','')
    # print(resp)
