import xml.etree.ElementTree as ET

import cv2
import matplotlib as mpl
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


def get_traffic_cone_positions(world_file_path):
    tree = ET.parse(world_file_path)
    root = tree.getroot()

    # List to store positions of traffic cones
    cones_positions = []
    cone_models = {model.get('name'):model  for model in root.findall('.//model') if "Traffic_Cone" in model.get('name')}
    # Iterate through the elements to find cones or other objects
    world = root.find('.//state')
    for model in world.findall('.//model'):
        model_name = model.get('name')
        if "Traffic_Cone" in model_name:  # Adjust this condition based on how cones are defined in your world file
            model_details = cone_models[model_name]
            uri = model_details.findall('.//uri')[0].text
            pose = model.find('pose').text
            x, y, z, yaw, pitch, roll = map(float, pose.split())  # Convert pose values to float
            cones_positions.append((x, y, z, yaw, pitch, roll,True if 'large'in uri else False ))

    return np.array(cones_positions)


def filter_traffic_cones(traffic_cones, cones_point_cloud, max_distance=0.2):
    filtered_cones = []
    
    for cone in traffic_cones:
        distances = np.linalg.norm(cones_point_cloud - cone[0:3], axis=1)
        if np.any(distances <= max_distance):
            filtered_cones.append(cone)
    
    return np.array(filtered_cones)

def get_bounding_box_corners(dimensions, model_position):
    # Calculate half of the scaled dimensions
    half_dimensions = tuple(dim / 2 for dim in dimensions)

    # Calculate the minimum and maximum coordinates of the bounding box
    min_coords = (model_position[0] - half_dimensions[0], model_position[1] - half_dimensions[1], model_position[2] - half_dimensions[2])
    max_coords = (model_position[0] + half_dimensions[0], model_position[1] + half_dimensions[1], model_position[2] + half_dimensions[2])

    # Compute the coordinates of the eight corners
    corners = [
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]]
    ]
    
    return corners


def get_pixel_coordinates(point_cloud):
    camera_info = {
        'K': [843.3386731699313, 0.0, 960.5, 0.0, 843.3386731699313, 540.5, 0.0, 0.0, 1.0],
        'D' : [0.0, 0.0, 0.0, 0.0, 0.0],
        'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'P' : [843.3386731699313, 0.0, 960.5, -0.0, 0.0, 843.3386731699313, 540.5, 0.0, 0.0, 0.0, 1.0, 0.0]
    }
    # Camera intrinsic parameters
    K = np.array(camera_info['K']).reshape(3, 3)
    D = np.array(camera_info['D'])
    P = np.array(camera_info['P']).reshape(3, 4)
    if np.any(D):
        undistorted_points, _ = cv2.undistortPoints(point_cloud[:, np.newaxis, :], K, D, P=P)
        pixel_coordinates = cv2.projectPoints(undistorted_points, np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0][:, 0]
    else:
        pixel_coordinates = cv2.projectPoints(point_cloud, np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0][:, 0]
    
    return pixel_coordinates
    





# Load point cloud data
point_cloud_world = np.load('point_cloud_data.npy')
cones_point_cloud = np.load('closest_cones.npy')
world_file_path = "../vehicle_sim/worlds/gazebo_world_description/worlds/mcity_new.world"
traffic_cones = get_traffic_cone_positions(world_file_path)
filtered_cones = filter_traffic_cones(traffic_cones,cones_point_cloud)
# Vehicle pose in world coordinates
vehicle_position_world = np.array([-99.49257890565369, -201.3691917210684, -0.016319196356962773])
vehicle_quaternion = np.array([-0.00029460861964082993, -0.0006380377440078058, 0.596975459123266, 0.8022591896127861])

# Camera to vehicle transformation
camera_to_vehicle_translation = np.array([1.750, 0.000, 1.591])
camera_to_vehicle_quaternion = np.array([-0.500, 0.500, -0.500, 0.500])

point_cloud_camera = transform_to_camera_coordinates(point_cloud_world,vehicle_position_world,vehicle_quaternion,camera_to_vehicle_translation,camera_to_vehicle_quaternion)

pixel_coordinates = get_pixel_coordinates(point_cloud_camera)
# Round pixel coordinates to integers
pixel_coordinates_x = np.round(pixel_coordinates[:, 0]).astype(int)
pixel_coordinates_y = np.round(pixel_coordinates[:, 1]).astype(int)


sorted_z = point_cloud_camera[np.argsort(point_cloud_camera[:, 1])[::-1]]
rows = len(point_cloud_camera)
if sorted_z[0][1] < sorted_z[rows - 1][1]:
        min_z_val = sorted_z[0][2]
        max_z_val = sorted_z[rows - 1][1]
else:
    max_z_val = sorted_z[0][1]
    min_z_val = sorted_z[rows - 1][1]

cmap_norm = mpl.colors.Normalize(vmin=min_z_val, vmax=max_z_val)
point_colors = plt.get_cmap('gist_rainbow')(cmap_norm(point_cloud_camera[:, -2]))[:, 0:3]
low_z_pixels = point_cloud_camera[:, 1] >= 1.57
point_colors[low_z_pixels] = [0.5, 0.5, 0.5]

# Create a blank image canvas
width = 1920  # Width of the image
height = 1080  # Height of the image
image = np.ones((height, width, 3), dtype=np.uint8)   

pixel_coordinates_x_clipped = np.clip(pixel_coordinates_x, 0, width - 1)

pixel_coordinates_y_clipped = np.clip(pixel_coordinates_y, 0, height - 1)

# Assign colors to pixels based on coordinates and calculated colors
for x, y, color in zip(pixel_coordinates_x_clipped, pixel_coordinates_y_clipped, point_colors):
    image[y, x] = (color * 255).astype(np.uint8)  


gazebo_scale = (0.254, 0.254, 0.254)    
for cone in filtered_cones:
    dimensions_dae = (3.1943, 3.1943, 5.306) if cone[6] else (2.2328, 2.29317, 3.20000)
    dimensions_gazebo = tuple(dim * scale for dim, scale in zip(dimensions_dae, gazebo_scale))
    bounding_boxes_cones = get_bounding_box_corners(dimensions_gazebo,cone)
    bounding_boxes_camera = transform_to_camera_coordinates(bounding_boxes_cones,vehicle_position_world,vehicle_quaternion,camera_to_vehicle_translation,camera_to_vehicle_quaternion)
    corners = get_pixel_coordinates(bounding_boxes_camera)
    # Define the corners of the bounding box
    top_left = (int(min(corners[:, 0])), int(min(corners[:, 1])))
    bottom_right = (int(max(corners[:, 0])), int(max(corners[:, 1])))

    # Draw the 2D bounding box on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)


# Convert the image to Open3D's format
o3d_image = o3d.geometry.Image(image)

o3d.io.write_image("./pixel_image.png", o3d_image)

# Display the image using Open3D
o3d.visualization.draw_geometries([o3d_image], window_name="Pixel Image")
