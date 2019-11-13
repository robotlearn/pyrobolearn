#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Coman subscriber.
"""

import rospy
# from std_msgs import msg as stdMsg
from sensor_msgs import msg as senMsg
from geometry_msgs import msg as geoMsg

import sys
import numpy as np

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# ROS - Gazebo
class ComanSubscriber(object):
    """
    ROS node that subscribes to topics to get the state of the coman.

    Here is the list of topics the node can subscribe to:
    - joint states
    - IMUs
    - Force/torque sensors
    - cameras
    """

    def __init__(self, robot_id=None):

        if robot_id is None:
            rospy.init_node('ComanSubscriber', anonymous=True)
        else:
            rospy.init_node('ComanSubscriber' + str(robot_id))

        # Joint States
        self.sub_joints = rospy.Subscriber(
            "/coman/joint_states", senMsg.JointState, self.joints_callback)
        self.joints = None

        # IMUs
        # self.sub_imu1 = rospy.Subscriber(
        #     "/coman/sensor/IMU", senMsg.Imu, self.imu1_callback)
        # self.imu1 = None

        # self.sub_imu2 = rospy.Subscriber(
        #     "/coman/sensor/imu2", senMsg.Imu, self.imu2_callback)
        # self.imu2 = None

        # Force-Torque sensors
        self.sub_ft_LForearm = rospy.Subscriber(
            "/coman/ft_sensor/LForearm", geoMsg.WrenchStamped, self.ft_LForearm_callback)
        self.ft_LForearm = None

        self.sub_ft_RForearm = rospy.Subscriber(
            "/coman/ft_sensor/RForearm", geoMsg.WrenchStamped, self.ft_RForearm_callback)
        self.ft_RForearm = None

        self.sub_ft_LAnkle = rospy.Subscriber(
            "/coman/ft_sensor/LAnkle", geoMsg.WrenchStamped, self.ft_LAnkle_callback)
        self.ft_LAnkle = None

        self.sub_ft_RAnkle = rospy.Subscriber(
            "/coman/ft_sensor/RAnkle", geoMsg.WrenchStamped, self.ft_RAnkle_callback)
        self.ft_RAnkle = None

        # Cameras
        self.sub_camera_rgb = rospy.Subscriber(
            "/camera/rgb/image_raw", senMsg.Image, self.camera_rgb_callback)
        self.camera_rgb = None

        self.sub_camera_depth = rospy.Subscriber(
            "/camera/depth/image_raw", senMsg.Image, self.camera_depth_callback)
        self.camera_depth = None


    def joints_callback(self, msg):
        pass

    def imu_callback(self, msg):
        pass

    def ft_callback(self, msg):
        pass

    def camera_callback(self, msg):
        pass


# if __name__ == '__main__':

#     node = ComanListener()

#     rospy.loginfo("Running coman listener...")

#     rospy.spin()
