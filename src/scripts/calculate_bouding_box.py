import math
import os
import xml.etree.ElementTree as ET

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rospy
import transformations as tf
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from tf.transformations import quaternion_matrix


def pose_constructor(loader, node):
    # Load the data from the YAML node
    state_data = node.value

    # Extract Point data
    
    point_data = state_data[0][1].value[0].value[0][1].value
    point = Point(x=point_data[0].value, y=point_data[1].value, z=point_data[2].value)

    # Extract Quaternion data

    quaternion_data = state_data[0][1].value[1].value[0][1].value
    quaternion = Quaternion(x=quaternion_data[0].value, y=quaternion_data[1].value, z=quaternion_data[2].value, w=quaternion_data[3].value)

    # Create Pose object
    pose = Pose(position=point, orientation=quaternion)
    return pose

yaml.SafeLoader.add_constructor(
    'tag:yaml.org,2002:python/object/new:geometry_msgs.msg._Pose.Pose', pose_constructor
)


def get_traffic_cone_to_positions_mapping(world_file_path):
    tree = ET.parse(world_file_path)
    root = tree.getroot()

    # Dictionary to store positions of traffic cones
    cones_positions = {}
    # Iterate through the elements to find cones or other objects
    world = root.find('.//state')
    for model in world.findall('.//model'):
        model_name = model.get('name')
        if "Traffic_Cone" in model_name:  # Adjust this condition based on how cones are defined in your world file
            pose = model.find('pose').text
            x, y, z, _, _, _ = map(float, pose.split())  # Convert pose values to float
            cones_positions[model_name] = (x, y, z)

    return cones_positions

def read_vehicle_position_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        return data

def transform_point_to_world(sensor_points, vehicle_pose, sensor_to_camera_tf, camera_to_vehicle_tf):
    # Convert sensor points to homogeneous coordinates
    sensor_points_homogeneous = np.column_stack((sensor_points, np.ones(sensor_points.shape[0])))
    
    # Create transformation matrix from sensor to camera
    rotation_matrix_camera = quaternion_matrix([sensor_to_camera_tf.rotation.x,
                                                sensor_to_camera_tf.rotation.y,
                                                sensor_to_camera_tf.rotation.z,
                                                sensor_to_camera_tf.rotation.w])
    translation_vector_camera = [sensor_to_camera_tf.translation.x,
                                 sensor_to_camera_tf.translation.y,
                                 sensor_to_camera_tf.translation.z]
    sensor_to_camera_matrix = np.identity(4)
    sensor_to_camera_matrix[:3, :3] = rotation_matrix_camera[:3, :3]
    sensor_to_camera_matrix[:3, 3] = translation_vector_camera

    # Create transformation matrix from camera to vehicle
    rotation_matrix_vehicle = quaternion_matrix([camera_to_vehicle_tf.rotation.x,
                                                 camera_to_vehicle_tf.rotation.y,
                                                 camera_to_vehicle_tf.rotation.z,
                                                 camera_to_vehicle_tf.rotation.w])
    translation_vector_vehicle = [camera_to_vehicle_tf.translation.x,
                                  camera_to_vehicle_tf.translation.y,
                                  camera_to_vehicle_tf.translation.z]
    camera_to_vehicle_matrix = np.identity(4)
    camera_to_vehicle_matrix[:3, :3] = rotation_matrix_vehicle[:3, :3]
    camera_to_vehicle_matrix[:3, 3] = translation_vector_vehicle

    # Convert sensor points to camera frame
    sensor_points_in_camera_frame = np.dot(sensor_points_homogeneous, sensor_to_camera_matrix.T)[:, :3]

    # Convert camera frame to vehicle frame
    sensor_points_in_vehicle_frame = np.dot(np.column_stack((sensor_points_in_camera_frame, np.ones(sensor_points.shape[0]))), camera_to_vehicle_matrix.T)[:, :3]

    # Convert vehicle frame to world frame
    # Assuming the vehicle_pose is given as a transformation from the world frame to the vehicle frame
    rotation_matrix_world = quaternion_matrix([vehicle_pose.orientation.x,
                                               vehicle_pose.orientation.y,
                                               vehicle_pose.orientation.z,
                                               vehicle_pose.orientation.w])
    translation_vector_world = [vehicle_pose.position.x,
                                vehicle_pose.position.y,
                                vehicle_pose.position.z]
    world_to_vehicle_matrix = np.identity(4)
    world_to_vehicle_matrix[:3, :3] = rotation_matrix_world[:3, :3]
    world_to_vehicle_matrix[:3, 3] = translation_vector_world

    # Convert sensor points to world frame
    world_points = np.dot(np.column_stack((sensor_points_in_vehicle_frame, np.ones(sensor_points.shape[0]))), world_to_vehicle_matrix.T)[:, :3]

    return world_points


def transform_world_to_camera(world_points, sensor_to_camera_tf):
    # Convert world points to homogeneous coordinates
    world_points_homogeneous = np.column_stack((world_points, np.ones(world_points.shape[0])))

    # Create transformation matrix from sensor to camera
    rotation_matrix = quaternion_matrix([sensor_to_camera_tf.rotation.x,
                                          sensor_to_camera_tf.rotation.y,
                                          sensor_to_camera_tf.rotation.z,
                                          sensor_to_camera_tf.rotation.w])
    translation_vector = [sensor_to_camera_tf.translation.x,
                          sensor_to_camera_tf.translation.y,
                          sensor_to_camera_tf.translation.z]
    sensor_to_camera_matrix = np.identity(4)
    sensor_to_camera_matrix[:3, :3] = rotation_matrix[:3, :3]
    sensor_to_camera_matrix[:3, 3] = translation_vector

    # Convert world points to camera frame
    world_points_in_camera_frame = np.dot(world_points_homogeneous, np.linalg.inv(sensor_to_camera_matrix).T)[:, :3]

    return world_points_in_camera_frame



def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def find_closest_cone(point, cones):
    min_distance = float('inf')
    closest_cone = None
    for cone, position in cones.items():
        dist = distance(point, position)
        if dist < min_distance:
            min_distance = dist
            closest_cone = cone
    return closest_cone, min_distance

def find_closest_cones_to_point_cloud(point_cloud, cones_positions, max_distance=0.2):
    closest_cones_to_points = {}
    
    for cone, cone_position in cones_positions.items():
        # Calculate distances between all points and the cone
        distances = np.linalg.norm(point_cloud - np.array(cone_position), axis=1)
        # Find the indices of points within max_distance
        within_distance_indices = np.where(distances < max_distance)[0]
        if len(within_distance_indices) > 0:
            # Find the index of the closest point within max_distance
            min_distance_index = within_distance_indices[np.argmin(distances[within_distance_indices])]
            min_distance = distances[min_distance_index]
            closest_point = point_cloud[min_distance_index]
            
            # Store the closest point and its distance
            closest_cones_to_points[cone] = {'closest_point': closest_point, 'min_distance': min_distance}
    
    return closest_cones_to_points

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
    

def get_rgb_values(image, pixel_coordinates):
    height, width, _ = image.shape
    rgb_values = []

    for (u, v) in pixel_coordinates:
        if 0 <= int(v) < height and 0 <= int(u) < width:
            rgb_values.append(image[int(v), int(u)])
        else:
            rgb_values.append([0, 0, 0])  # Default color for out-of-bound points

    return np.array(rgb_values)
   

def form_rgb_d_image(image, point_cloud, pixel_coordinates):
    depth_values = point_cloud[:, 2]  # Assuming Z is the depth

    rgb_d_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
    rgb_d_image[:, :, :3] = image[:, :, :3]
    for i, (u, v) in enumerate(pixel_coordinates):
        if 0 <= int(v) < rgb_d_image.shape[0] and 0 <= int(u) < rgb_d_image.shape[1]:
            # Use original RGB value from the image for all pixels
            rgb_d_image[int(v), int(u), 3] = depth_values[i]  # Use depth value

    return rgb_d_image

def save_point_cloud_to_bin(point_cloud, filename):
    # Convert the point cloud to a NumPy array
    points = np.asarray(point_cloud.points)

    # Save the points to a binary file
    with open(filename, 'wb') as f:
        points.tofile(f)


def plot_rgb_d_image(rgb_d_image):
    rgb_image = rgb_d_image[:, :, :3].astype(np.uint8)  # Convert RGB to uint8
    depth_image = rgb_d_image[:, :, 3]

    # Alpha channel visualization
    alpha = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    combined_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2BGRA)

    combined_image[:, :, 3] = alpha

    
    cv2.imshow("RGB Image", rgb_image)
    cv2.imshow("Alpha Channel", alpha)
    cv2.imshow("Combined RGB-D Image", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./rgbd.png', combined_image)


# def plot_rgb_d_image(rgb_d_image):
#     rgb_image = rgb_d_image[:, :, :3] / 255.0  # Normalize RGB values
#     depth_image = rgb_d_image[:, :, 3]

#     # Normalize depth image for visualization
#     depth_min = np.nanmin(depth_image)
#     depth_max = np.nanmax(depth_image)
#     depth_image_normalized = (depth_image - depth_min) / (depth_max - depth_min)

#     # Apply colormap to depth image
#     depth_colormap = plt.cm.viridis(depth_image_normalized)
#     depth_colormap_rgb = depth_colormap[:, :, :3]

#     # Overlay depth colormap on RGB image with transparency
#     alpha = 0.5  # Transparency factor
#     combined_image = (1 - alpha) * rgb_image + alpha * depth_colormap_rgb

#     plt.figure(figsize=(10, 10))
#     plt.imshow(combined_image)
#     plt.title('Combined RGB-D Image')
#     plt.axis('off')
#     plt.show()

if __name__ == "__main__":

    # Directory containing point cloud binary files
    bin_dir = "/home/kedhar/workspace/catkin_ws/src/scripts/point_cloud_bins"
    image_dir = '/home/kedhar/workspace/catkin_ws/src/scripts/point_cloud_images'
    pose_messages_dir = '/home/kedhar/workspace/catkin_ws/src/scripts/pose_messages'
    bin_files = sorted(os.listdir(bin_dir))
    pose_files = sorted(os.listdir(pose_messages_dir))
    world_file_path = "../vehicle_sim/worlds/gazebo_world_description/worlds/mcity_new.world"
    cone_position_mapping = get_traffic_cone_to_positions_mapping(world_file_path)
    traffic_cones = get_traffic_cone_positions(world_file_path)
    
    
    sensor_translation = (1.250, 0.000, 1.841)


    sensor_to_camera_tf = TransformStamped()

    # Set the header timestamp
    sensor_to_camera_tf.header.stamp = rospy.Time(0)

    # Set translation values
    sensor_to_camera_tf.transform.translation.x = 0.000 
    sensor_to_camera_tf.transform.translation.y = -0.341
    sensor_to_camera_tf.transform.translation.z = -0.500

    # Set rotation values (quaternion)
    sensor_to_camera_tf.transform.rotation.x = 0.707
    sensor_to_camera_tf.transform.rotation.y = 0.000
    sensor_to_camera_tf.transform.rotation.z = -0.000
    sensor_to_camera_tf.transform.rotation.w = 0.707

    camera_to_vehicle_tf = TransformStamped()
    camera_to_vehicle_tf.header.stamp = rospy.Time(0)
    camera_to_vehicle_tf.transform.translation.x = 1.750
    camera_to_vehicle_tf.transform.translation.y = 0.000
    camera_to_vehicle_tf.transform.translation.z = 1.591
    camera_to_vehicle_tf.transform.rotation.x = -0.500
    camera_to_vehicle_tf.transform.rotation.y = 0.500
    camera_to_vehicle_tf.transform.rotation.z = -0.500
    camera_to_vehicle_tf.transform.rotation.w = 0.500

    camera_to_vehicle_translation = np.array([1.750, 0.000, 1.591])
    camera_to_vehicle_quaternion = np.array([-0.500, 0.500, -0.500, 0.500])


    image_width = 1920  # Width of the camera image in pixels
    image_height = 1080  # Height of the camera image in pixels

    # Process each binary file in the directory
    for idx, (bin_file, pose_file) in enumerate(zip(bin_files, pose_files)):
        bin_path = os.path.join(bin_dir, bin_file)
        pose_file_path = os.path.join(pose_messages_dir, pose_file)

        # Load point cloud data from the binary file
        point_cloud = load_point_cloud_binary(bin_path)
        vehicle_pose = read_vehicle_position_from_yaml(pose_file_path)

        image_file = f"point_cloud_image_{idx}.png"
        image_path = os.path.join(image_dir, image_file)

        point_cloud_world = transform_point_to_world(point_cloud, vehicle_pose,sensor_to_camera_tf.transform, camera_to_vehicle_tf.transform)

        closest_cones_to_points = find_closest_cones_to_point_cloud(point_cloud_world, cone_position_mapping)
        cloeset_cones_points_list = [data['closest_point'] for data in closest_cones_to_points.values()]
        cloeset_cones_points_list = np.array(cloeset_cones_points_list)


        filtered_cones = filter_traffic_cones(traffic_cones,cloeset_cones_points_list)
        vehicle_position_world =  [float(vehicle_pose.position.x),
                                float(vehicle_pose.position.y),
                                float(vehicle_pose.position.z)]        
        vehicle_quaternion = [vehicle_pose.orientation.x,
                                               vehicle_pose.orientation.y,
                                               vehicle_pose.orientation.z,
                                               vehicle_pose.orientation.w]
        
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
        # image = np.ones((height, width, 3), dtype=np.uint8)
        image = cv2.imread('/home/kedhar/workspace/catkin_ws/src/scripts/camera_images/image_20240517_143947_395069_1.jpg')
        rgb_d_image = form_rgb_d_image(image, point_cloud, pixel_coordinates)

        plot_rgb_d_image(rgb_d_image)
        exit()

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

        o3d.io.write_image(f"./labelled_point_cloud_images/labelled_point_cloud_image_{idx}_tmp.png", o3d_image)

        # # Display the image using Open3D
        # o3d.visualization.draw_geometries([o3d_image], window_name="Pixel Image")

        break

                