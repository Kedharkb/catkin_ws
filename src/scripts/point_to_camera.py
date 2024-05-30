import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion
from tf.transformations import quaternion_matrix

point_cloud_data = np.load('point_cloud_data.npy')
# Vehicle pose in world coordinates
vehicle_pose = Pose()
vehicle_pose.position = Point(-100.74209768,-199.69339932,-0.016319196356962773)
vehicle_pose.orientation = Quaternion(-0.00029460861964082993, -0.0006380377440078058, 0.596975459123266, 0.8022591896127861)

# Vehicle base to camera transform
translation = np.array([1.750, 0.000, 1.591])
quaternion = np.array([-0.500, 0.500, -0.500, 0.500])



# Convert quaternion to transformation matrix
T_vehicle_to_camera = quaternion_matrix(quaternion)
T_vehicle_to_camera[:3, 3] = translation

camera_info = {
    'K': [843.3386731699313, 0.0, 960.5, 0.0, 843.3386731699313, 540.5, 0.0, 0.0, 1.0],
    'D' : [0.0, 0.0, 0.0, 0.0, 0.0],
    'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    'P' : [843.3386731699313, 0.0, 960.5, -0.0, 0.0, 843.3386731699313, 540.5, 0.0, 0.0, 0.0, 1.0, 0.0]
}


# Create an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

# Set camera intrinsic parameters
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(1920, 1080, camera_info['K'][0], camera_info['K'][4], camera_info['K'][2], camera_info['K'][5])

# Extract camera position and orientation
camera_position = np.array([vehicle_pose.position.x, vehicle_pose.position.y, vehicle_pose.position.z])
rotation_matrix = quaternion_matrix([vehicle_pose.orientation.x, vehicle_pose.orientation.y, vehicle_pose.orientation.z, vehicle_pose.orientation.w])

camera_front = np.dot(rotation_matrix[:3, :3], np.array([-1, 0, 0]))
camera_up = np.dot(rotation_matrix[:3, :3], np.array([0, 0, 1]))

print("Camera Position:", camera_position)
print("Camera Front Vector:", camera_front)
print("Camera Up Vector:", camera_up)

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1080)

# Add the point cloud to the visualization
vis.add_geometry(pcd)

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
