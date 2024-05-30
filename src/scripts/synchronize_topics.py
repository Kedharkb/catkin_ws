import threading
from datetime import datetime

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2


class DataSynchronizer:
    def __init__(self):
        self.bridge = CvBridge()
        self.file_counter = 1
        self.lock = threading.Lock()

    def callback(self, pc_msg, img_msg, pose_msg):
        # Extract point cloud data
        point_cloud_data = pc_msg.data
        # Extract image data
        image_data = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        # image_data= img_msg.data
        pose_data = pose_msg.pose
        
        # Check time difference between messages
        pose_time = pose_msg.header.stamp
        img_time = img_msg.header.stamp
        pc_time = pc_msg.header.stamp

        max_time_diff = rospy.Duration.from_sec(0.01)  # Maximum time difference allowed (0.01 seconds)
        if abs(pose_time - img_time) > max_time_diff or abs(pose_time - pc_time) > max_time_diff:
            rospy.logwarn("Time difference between messages exceeds threshold.")
            return

        # Generate a timestamp for the current time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Construct the file names with the timestamp and file counter
        pc_file_name = f"./point_cloud_bins/point_cloud_{timestamp}_{self.file_counter}.bin"
        img_file_name = f"./camera_images/image_{timestamp}_{self.file_counter}.jpg"
        pose_file_name = f'./pose_messages/pose_{timestamp}_{self.file_counter}.yaml'
        
        # Increment the file counter for the next file
        with self.lock:
            self.file_counter += 1

        # Save raw point cloud data to file
        with open(pc_file_name, 'wb') as f:
            f.write(point_cloud_data)
        
        with open(pose_file_name, 'w') as f:
            yaml.dump(pose_data, f)  # Serialize pose data to YAML file
        rospy.loginfo("Data received and saved to files: Point Cloud - %s, Image - %s, Pose - %s", pc_file_name, img_file_name, pose_file_name)

        # Save image data to file
       
        with open(img_file_name, 'w') as f:
            # yaml.dump(image_data, f)  # Serialize pose data to YAML file
             cv2.imwrite(img_file_name, image_data)
        rospy.loginfo("Data received and saved to files: Point Cloud - %s, Image - %s", pc_file_name, img_file_name)



if __name__ == "__main__":
    rospy.init_node('data_synchronizer', anonymous=True)

    synchronizer = DataSynchronizer()

    pc_sub = Subscriber('/points_raw', PointCloud2)
    img_sub = Subscriber('/image_raw', Image)
    vehicle_pose_sub = Subscriber('/vehicle_info/pose', PoseStamped)


    sync = ApproximateTimeSynchronizer([pc_sub, img_sub, vehicle_pose_sub], queue_size=100, slop=0.01)
    sync.registerCallback(synchronizer.callback)

    rospy.spin()
