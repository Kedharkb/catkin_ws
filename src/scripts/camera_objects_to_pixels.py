import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import transformations as tf
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from scipy.spatial.transform import Rotation as R

# def get_bounding_box_corners(dimensions, model_position):
#     # Extract position and orientation
#     position = model_position[0:3]
#     yaw_pitch_roll = model_position[3:6]

#     # Convert yaw, pitch, and roll to rotation matrix
#     rotation_matrix = R.from_euler('xyz', yaw_pitch_roll, degrees=True).as_matrix()
#     print('rotation_matrix',rotation_matrix)

#     # Define half dimensions (assuming uniform for simplicity)
#     half_dimensions = tuple(dim / 2 for dim in dimensions)

#     # Define corner offsets relative to center
#     corner_offsets = [
#         (-half_dimensions[0], -half_dimensions[1], -half_dimensions[2]),
#         (half_dimensions[0], -half_dimensions[1], -half_dimensions[2]),
#         (half_dimensions[0], half_dimensions[1], -half_dimensions[2]),
#         (-half_dimensions[0], half_dimensions[1], -half_dimensions[2]),
#         (-half_dimensions[0], -half_dimensions[1], half_dimensions[2]),
#         (half_dimensions[0], -half_dimensions[1], half_dimensions[2]),
#         (half_dimensions[0], half_dimensions[1], half_dimensions[2]),
#         (-half_dimensions[0], half_dimensions[1], half_dimensions[2])
#     ]

#     # Apply rotation to each corner (using corner offsets)
#     rotated_corners = []
#     for corner_offset in corner_offsets:
#         corner_point = np.array(corner_offset)
#         rotated_corner = np.dot(rotation_matrix, corner_point)
#         rotated_corners.append(rotated_corner + position)  # Add position for final corner location

#     return rotated_corners



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

def read_vehicle_position_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        return data
    
def get_bounding_box_corners(dimensions, model_position):
    # Calculate half of the scaled dimensions
    half_dimensions = tuple(dim / 2 for dim in dimensions)

    # Calculate the minimum and maximum coordinates of the bounding box
    min_coords = (model_position[0] - half_dimensions[0], model_position[1] - half_dimensions[1], model_position[2] - half_dimensions[2])
    max_coords = (model_position[0] + half_dimensions[0], model_position[1] + half_dimensions[1], model_position[2] + half_dimensions[2])

    # print("Bounding box minimum coordinates:", min_coords)
    # print("Bounding box maximum coordinates:", max_coords)


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

def transform_to_camera_coordinates(point_cloud, vehicle_position_world, vehicle_quaternion, camera_to_vehicle_translation, camera_to_vehicle_quaternion):
        # Convert quaternion to rotation matrix for vehicle pose
    vehicle_pose_world = tf.quaternion_matrix(vehicle_quaternion)
    vehicle_pose_world[:3, 3] = vehicle_position_world

    # Invert rotation matrix for vehicle pose
    vehicle_pose_world_inv = np.linalg.inv(vehicle_pose_world)
    
    # Convert quaternion to rotation matrix for camera to vehicle transformation
    camera_to_vehicle_pose = tf.quaternion_matrix(camera_to_vehicle_quaternion)
    camera_to_vehicle_pose[:3, 3] = camera_to_vehicle_translation

    # Get the inverse of the camera to vehicle pose to transform from camera to vehicle
    camera_to_vehicle_pose_inv = np.linalg.inv(camera_to_vehicle_pose)


    # Convert points to homogeneous coordinates
    point_cloud_homogeneous = np.hstack((point_cloud, np.ones((len(point_cloud), 1))))

    # Apply transformation from world to camera
    point_cloud_vehicle = np.dot(vehicle_pose_world_inv, point_cloud_homogeneous.T).T
    point_cloud_camera = np.dot(camera_to_vehicle_pose_inv, point_cloud_vehicle.T).T


    return point_cloud_camera[:, :3]

def get_bounding_box_center(bounding_boxes_camera):
  """
  Calculates the center of the bounding box in camera coordinates.
  """
  center = np.mean(bounding_boxes_camera, axis=0)
  return center



if __name__ == "__main__":
    camera_image_dir = '/home/kedhar/workspace/catkin_ws/src/scripts/camera_images'
    pose_messages_dir = '/home/kedhar/workspace/catkin_ws/src/scripts/pose_messages'
    pose_files = sorted(os.listdir(pose_messages_dir))
    camera_image_files = sorted(os.listdir(camera_image_dir))
    labelled_camera_images = '/home/kedhar/workspace/catkin_ws/src/scripts/labelled_camera_images'
    world_file_path = "../vehicle_sim/worlds/gazebo_world_description/worlds/mcity_new.world"
    cone_positions = get_traffic_cone_positions(world_file_path)
    
    camera_info = {
    'K': [843.3386731699313, 0.0, 960.5, 0.0, 843.3386731699313, 540.5, 0.0, 0.0, 1.0],
    'D' : [0.0, 0.0, 0.0, 0.0, 0.0],
    'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    'P' : [843.3386731699313, 0.0, 960.5, -0.0, 0.0, 843.3386731699313, 540.5, 0.0, 0.0, 0.0, 1.0, 0.0]
}
    

    camera_to_vehicle_translation = np.array([1.750, 0.000, 1.591])
    camera_to_vehicle_quaternion = np.array([-0.500, 0.500, -0.500, 0.500])
    
    for idx, (camera_file, pose_file) in enumerate(zip(camera_image_files, pose_files)):
        camera_image_path= os.path.join(camera_image_dir, camera_file)
        pose_file_path = os.path.join(pose_messages_dir, pose_file)
        vehicle_pose = read_vehicle_position_from_yaml(pose_file_path)
        vehicle_position_world =  [float(vehicle_pose.position.x),
                                float(vehicle_pose.position.y),
                                float(vehicle_pose.position.z)]
        print(vehicle_position_world)
        
        vehicle_quaternion = [vehicle_pose.orientation.x,
                                               vehicle_pose.orientation.y,
                                               vehicle_pose.orientation.z,
                                               vehicle_pose.orientation.w]

        image = cv2.imread(camera_image_path)
        image_height, image_width = image.shape[:2]
        color = (0, 255, 0)  # Green color
        thickness = 2


        # Define the 30-meter offset radius
        offset_distance = 100.0  # in meters
        rotation_matrix = R.from_quat(vehicle_quaternion).as_matrix()

        # Define the direction vector of the vehicle
        vehicle_direction = rotation_matrix @ np.array([1, 0, 0])

        # Filter cone positions based on the 30-meter offset radius from the vehicle
        cone_positions_filtered = []
        for cone_position in cone_positions:
            # Calculate the vector from the vehicle to the cone
            print(cone_position[0:3])
            cone_vector = cone_position[0:3] - vehicle_position_world
            
            # Calculate the dot product between the cone vector and the vehicle direction
            dot_product = np.dot(vehicle_direction, cone_vector)
            
            # Check if the cone falls within the specified distance range and is in front of the vehicle
            if 0 < dot_product <= offset_distance:
                cone_positions_filtered.append(cone_position)

        # point_cloud_camera = transform_to_camera_coordinates(cone_positions_filtered,vehicle_position_world,vehicle_quaternion,camera_to_vehicle_translation,camera_to_vehicle_quaternion)
        gazebo_scale = (0.254, 0.254, 0.254)    
        for cone in cone_positions_filtered:
            dimensions_dae = (3.1943, 3.1943, 5.306) if cone[6] else (2.2328, 2.29317, 3.20000)
            dimensions_gazebo = tuple(dim * scale for dim, scale in zip(dimensions_dae, gazebo_scale))
            bounding_boxes_cones = get_bounding_box_corners(dimensions_gazebo,cone)
            bounding_boxes_camera = transform_to_camera_coordinates(bounding_boxes_cones,vehicle_position_world,vehicle_quaternion,camera_to_vehicle_translation,camera_to_vehicle_quaternion)
            object_center_camera = get_bounding_box_center(bounding_boxes_camera)
            object_mid_point = transform_to_camera_coordinates(np.array([cone[0:3]]), vehicle_position_world, vehicle_quaternion, camera_to_vehicle_translation, camera_to_vehicle_quaternion)


            # Camera intrinsic parameters
            K = np.array(camera_info['K']).reshape(3, 3)
            D = np.array(camera_info['D'])
            P = np.array(camera_info['P']).reshape(3, 4)


            if np.any(D):
                undistorted_points, _ = cv2.undistortPoints(bounding_boxes_camera[:, np.newaxis, :], K, D, P=P)
                pixel_coordinates = cv2.projectPoints(bounding_boxes_camera, np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0][:, 0]
            else:
                pixel_coordinates = cv2.projectPoints(bounding_boxes_camera, np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0][:, 0]
            
            if np.any(D):
                undistorted_points, _ = cv2.undistortPoints(object_mid_point[:, np.newaxis, :], K, D, P=P)
                object_center_coordinates = cv2.projectPoints(object_mid_point, np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0][:, 0]
            else:
                object_center_coordinates = cv2.projectPoints(object_mid_point, np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0][:, 0]

            # Draw the bounding box
            corners = pixel_coordinates
            # Define the corners of the bounding box
            top_left = (int(min(corners[:, 0])), int(min(corners[:, 1])))
            bottom_right = (int(max(corners[:, 0])), int(max(corners[:, 1])))

            # Draw the 2D bounding box on the image
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            center = (int(object_center_coordinates[:,0]), int(object_center_coordinates[:,1]))
            radius = 2  # Adjust the radius as needed
            color = (0, 0, 255)  # Red color
            thickness = 1  # Filled circle
            cv2.circle(image, center, radius, color, thickness)

        cv2.imwrite(f'{labelled_camera_images}/labelled_image_{idx}.png', image)