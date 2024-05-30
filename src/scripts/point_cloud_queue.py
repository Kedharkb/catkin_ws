import queue
import threading
from datetime import datetime

import rospy
from sensor_msgs.msg import PointCloud2


class PointCloudQueue:
    def __init__(self):
        self.q = queue.Queue()

    def add_message(self, msg):
        self.q.put(msg)

    def get_message(self):
        return self.q.get()

def save_point_cloud(queue):
    file_counter = 1  # Initialize file counter
    while not rospy.is_shutdown():
        msg = queue.get_message()
        if msg is None:  # Exit condition
            break
        # Extract point cloud data
        point_cloud_data = msg.data

        # Generate a timestamp for the current time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Construct the file name with the timestamp and file counter
        file_name = f"./point_cloud_bins/point_cloud_{timestamp}_{file_counter}.bin"

        # Increment the file counter for the next file
        file_counter += 1

        # Save raw point cloud data to file
        with open(file_name, 'wb') as f:
            f.write(point_cloud_data)

        rospy.loginfo("Point cloud received and saved to file: %s", file_name)

def point_cloud_callback(msg, queue):
    queue.add_message(msg)

if __name__ == "__main__":
    rospy.init_node('lidar_data_recorder', anonymous=True)

    point_cloud_queue = PointCloudQueue()

    save_thread = threading.Thread(target=save_point_cloud, args=(point_cloud_queue,))
    save_thread.start()

    rospy.Subscriber('/points_raw', PointCloud2, lambda msg: point_cloud_callback(msg, point_cloud_queue),queue_size=500)

    rospy.spin()

    point_cloud_queue.add_message(None)
    save_thread.join()
