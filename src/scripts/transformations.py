import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from tf.transformations import quaternion_matrix


def transform_point_to_world(sensor_point, vehicle_pose, sensor_to_vehicle_tf):
    # Convert sensor point to homogeneous coordinates
    sensor_point_homogeneous = [sensor_point.x, sensor_point.y, sensor_point.z, 1.0]

    # Create transformation matrix from sensor to vehicle
    rotation_matrix = quaternion_matrix([sensor_to_vehicle_tf.rotation.x,
                                          sensor_to_vehicle_tf.rotation.y,
                                          sensor_to_vehicle_tf.rotation.z,
                                          sensor_to_vehicle_tf.rotation.w])
    translation_vector = [sensor_to_vehicle_tf.translation.x,
                          sensor_to_vehicle_tf.translation.y,
                          sensor_to_vehicle_tf.translation.z]
    sensor_to_vehicle_matrix = rotation_matrix
    sensor_to_vehicle_matrix[:3, 3] = translation_vector

    # Convert sensor point to vehicle frame
    sensor_point_in_vehicle_frame = sensor_to_vehicle_matrix.dot(sensor_point_homogeneous)[:3]

    # Convert vehicle pose to transformation matrix
    vehicle_rotation_matrix = quaternion_matrix([vehicle_pose.orientation.x,
                                                  vehicle_pose.orientation.y,
                                                  vehicle_pose.orientation.z,
                                                  vehicle_pose.orientation.w])
    vehicle_translation_vector = [vehicle_pose.position.x,
                                  vehicle_pose.position.y,
                                  vehicle_pose.position.z]
    vehicle_matrix = vehicle_rotation_matrix
    vehicle_matrix[:3, 3] = vehicle_translation_vector

    # Convert sensor point to world frame
    world_point = vehicle_matrix.dot([sensor_point_in_vehicle_frame[0],
                                       sensor_point_in_vehicle_frame[1],
                                       sensor_point_in_vehicle_frame[2], 1.0])[:3]

    return Point(*world_point)

if __name__ == "__main__":
    rospy.init_node('point_cloud_transformer')

    # Define the sensor point
    sensor_point = Point(-9.6, 14, -1.4)

    # Define the vehicle pose
    vehicle_pose = Pose(Point(-99.486880484034, -201.35020115539373, -0.016276652234012312),
                        Quaternion(-0.00039773933916192514, -0.0006993177139353208, 0.5969063631525637, 0.8023105049705712))

    # Define the sensor-to-vehicle transform
    sensor_to_vehicle_tf = TransformStamped()
    sensor_to_vehicle_tf.header.stamp = rospy.Time(0)
    sensor_to_vehicle_tf.transform.translation.x = 1.250
    sensor_to_vehicle_tf.transform.translation.y = 0.000
    sensor_to_vehicle_tf.transform.translation.z = 1.932
    sensor_to_vehicle_tf.transform.rotation.x = 0.000
    sensor_to_vehicle_tf.transform.rotation.y = 0.000
    sensor_to_vehicle_tf.transform.rotation.z = -0.707
    sensor_to_vehicle_tf.transform.rotation.w = 0.707

    # Transform the sensor point to world coordinates
    world_point = transform_point_to_world(sensor_point, vehicle_pose, sensor_to_vehicle_tf.transform)

    rospy.loginfo("Sensor Point in World Coordinates: {}".format(world_point))
