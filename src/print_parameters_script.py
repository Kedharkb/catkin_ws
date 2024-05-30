#!/usr/bin/env python

import rospy


def print_parameters():
    # Initialize the ROS node
    rospy.init_node('print_parameters_node', anonymous=True)

    # Print the parameters
    rospy.loginfo("Printing parameters:")
    rospy.loginfo("Origin: %s", rospy.get_param('/robot_description'))
    # Add similar log statements for other parameters

if __name__ == '__main__':
    try:
        print_parameters()
    except rospy.ROSInterruptException:
        pass