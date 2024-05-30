import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import transformations as tf


def transform_to_camera_coordinates(point_cloud, vehicle_position_world, vehicle_quaternion, camera_to_vehicle_translation, camera_to_vehicle_quaternion):
    # Convert quaternion to rotation matrix for camera to vehicle transformation
    camera_to_vehicle_pose = tf.quaternion_matrix(camera_to_vehicle_quaternion)
    camera_to_vehicle_pose[:3, 3] = camera_to_vehicle_translation

    # Get the inverse of the camera to vehicle pose to transform from camera to vehicle
    camera_to_vehicle_pose_inv = np.linalg.inv(camera_to_vehicle_pose)

    # Convert quaternion to rotation matrix for vehicle pose
    vehicle_pose_world = tf.quaternion_matrix(vehicle_quaternion)
    vehicle_pose_world[:3, 3] = vehicle_position_world

    # Invert rotation matrix for vehicle pose
    vehicle_pose_world_inv = np.linalg.inv(vehicle_pose_world)

    # Apply transformation from vehicle to camera
    point_cloud_vehicle = np.dot(point_cloud, vehicle_pose_world_inv[:3, :3].T) + vehicle_pose_world_inv[:3, 3]
    point_cloud_camera = np.dot(point_cloud_vehicle - camera_to_vehicle_translation, camera_to_vehicle_pose_inv[:3, :3].T)

    return point_cloud_camera




camera_info = {
    'K': [843.3386731699313, 0.0, 960.5, 0.0, 843.3386731699313, 540.5, 0.0, 0.0, 1.0],
    'D' : [0.0, 0.0, 0.0, 0.0, 0.0],
    'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    'P' : [843.3386731699313, 0.0, 960.5, -0.0, 0.0, 843.3386731699313, 540.5, 0.0, 0.0, 0.0, 1.0, 0.0]
}



# Load point cloud data
point_cloud_world = np.load('point_cloud_data.npy')
cones_point_cloud = np.load('closest_cones.npy')

# Vehicle pose in world coordinates
vehicle_position_world = np.array([-99.49257890565369, -201.3691917210684, -0.016319196356962773])
vehicle_quaternion = np.array([-0.00029460861964082993, -0.0006380377440078058, 0.596975459123266, 0.8022591896127861])

# Camera to vehicle transformation
camera_to_vehicle_translation = np.array([1.750, 0.000, 1.591])
camera_to_vehicle_quaternion = np.array([-0.500, 0.500, -0.500, 0.500])

camera_rotation_matrix = tf.quaternion_matrix(camera_to_vehicle_quaternion)[:3, :3]


camera_transform_matrix = np.eye(4)
camera_transform_matrix[:3, :3] = camera_rotation_matrix
camera_transform_matrix[:3, 3] = camera_to_vehicle_translation

# Convert vehicle pose to transformation matrix
vehicle_transform_matrix = tf.quaternion_matrix(vehicle_quaternion)
vehicle_transform_matrix[:3, 3] = vehicle_position_world

# Calculate camera pose in the world coordinate system
camera_pose_world = np.dot(vehicle_transform_matrix, camera_transform_matrix)


camera_position_world = camera_pose_world[:3, 3]


combined_point_cloud = np.concatenate((point_cloud_world, cones_point_cloud))
combined_point_cloud_o3d = o3d.geometry.PointCloud()
combined_point_cloud_o3d.points = o3d.utility.Vector3dVector(combined_point_cloud)

# Set camera intrinsic parameters
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(1920, 1080, camera_info['K'][0], camera_info['K'][4], camera_info['K'][2], camera_info['K'][5])


# Extract camera position and orientation
camera_position = np.array(camera_position_world)
rotation_matrix = tf.quaternion_matrix(camera_to_vehicle_quaternion)

camera_front = np.dot(camera_pose_world[:3, :3], np.array([0, -1, 0]))
camera_up = np.dot(camera_pose_world[:3, :3], np.array([0, 0, 1]))

print("Camera Position:", camera_position)
print("Camera Front Vector:", camera_front)
print("Camera Up Vector:", camera_up)

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1080)

# Add the point cloud to the visualization
vis.add_geometry(combined_point_cloud_o3d)

camera_position_offset = np.array([1, 0.0, 0.0])  # Adjust this value as needed
camera_position_shifted = camera_position + camera_position_offset
print("Camera Position shifted :", camera_position_shifted)

# Set the viewpoint to match the camera pose
view_control = vis.get_view_control()
view_control.set_lookat(camera_position_shifted)
view_control.set_up(camera_up)
view_control.set_front(camera_front)
view_control.set_zoom(0.001) 

vis.run()
vis.destroy_window()
