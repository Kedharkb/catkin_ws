from datetime import datetime

import rospy
from sensor_msgs.msg import PointCloud2

# Global variable for the base file name
base_file_name = "point_cloud_raw"
count = 0
# Callback function to process incoming PointCloud2 messages and save data to file
def point_cloud_callback(msg):
    global count
    count+=1
    print(count)
    # # Extract point cloud data
    # point_cloud_data = msg.data

    # # Generate a timestamp for the current time
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # # Construct the file name with the timestamp
    # file_name = f"./point_cloud_bins/{base_file_name}_{timestamp}.bin"

    # # Save raw point cloud data to file
    # with open(file_name, 'wb') as f:
    #     f.write(point_cloud_data)

    # rospy.loginfo("Point cloud received and saved to file: %s", file_name)

if __name__ == '__main__':
    rospy.init_node('point_cloud_saver', anonymous=True)

    # Subscribe to the '/points_raw' topic
    rospy.Subscriber('/points_raw', PointCloud2, point_cloud_callback)

    rospy.loginfo("Waiting for a point cloud to be received...")
    rospy.spin()  # Keeps the node running until it is shut down
