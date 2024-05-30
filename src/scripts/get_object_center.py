import xml.etree.ElementTree as ET

import cv2
import numpy as np
import transformations as tf
from scipy.spatial.transform import Rotation as R


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


if __name__ == "__main__":
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



    vehicle_position_world = [-97.35860187856731,111.74854469067287,-0.011539982672574611]
    vehicle_quaternion = [-0.0015490925626798835, 0.000530947968306934, 0.8641576618859504,0.5032184950995886
]

    image_path = "./camera_images/image_20240425_185544_710868_499.jpg"
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    image_size = (image_width, image_height)
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
        cone_vector = cone_position[0:3] - vehicle_position_world
        
        # Calculate the dot product between the cone vector and the vehicle direction
        dot_product = np.dot(vehicle_direction, cone_vector)
        
        # Check if the cone falls within the specified distance range and is in front of the vehicle
        if 0 < dot_product <= offset_distance:
            cone_positions_filtered.append(cone_position)
    
    for cone in cone_positions_filtered:
        object_mid_point = transform_to_camera_coordinates(np.array([cone[0:3]]), vehicle_position_world, vehicle_quaternion, camera_to_vehicle_translation, camera_to_vehicle_quaternion)


        # Camera intrinsic parameters
        K = np.array(camera_info['K']).reshape(3, 3)
        D = np.array(camera_info['D'])
        P = np.array(camera_info['P']).reshape(3, 4)
    
        
        if np.any(D):
            undistorted_points, _ = cv2.undistortPoints(object_mid_point[:, np.newaxis, :], K, D, P=P)
            object_center_coordinates = cv2.projectPoints(object_mid_point, np.zeros((3, 1)), np.zeros((3, 1)), K, D,)[0][:, 0]
        else:
            object_center_coordinates = cv2.projectPoints(object_mid_point, np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0][:, 0]

        center = (int(object_center_coordinates[:,0]), int(object_center_coordinates[:,1]))
        radius = 2  # Adjust the radius as needed
        color = (0, 0, 255)  # Red color
        thickness = 1  # Filled circle
        cv2.circle(image, center, radius, color, thickness)

# Display the image with the bounding box
cv2.imshow("Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()