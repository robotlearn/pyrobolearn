# -*- coding: utf-8 -*-
import rospy
#from std_msgs import msg as stdMsg
from sensor_msgs import msg as senMsg
from geometry_msgs import msg as geoMsg

import sys
import numpy as np

__author__ = "Brian Delhaisse"
__email__ = "Brian.Delhaisse@iit.it"
__credits__ = ["Brian Delhaisse"]
__version__ = "1.0.0"
__date__ = "10/01/2017"


# ROS - Gazebo
class WalkmanListener:
    """
    ROS node that subscribes to topics to get the state of the walkman.

    Here is the list of topics the node can subscribe to:
    - joint states
    - IMUs
    - Force/torque sensors
    - cameras
    """

    def __init__(self):

        rospy.init_node('WalkmanListener')

        ### Joint States
        self.sub_joints = rospy.Subscriber(
            "/bigman/joint_states", senMsg.JointState, self.joints_callback)
        self.joints = None

        ### IMUs
        self.sub_imu1 = rospy.Subscriber(
            "/bigman/sensor/imu1", senMsg.Imu, self.imu1_callback)
        self.imu1 = None

        self.sub_imu2 = rospy.Subscriber(
            "/bigman/sensor/imu2", senMsg.Imu, self.imu2_callback)
        self.imu2 = None

        self.sub_head_imu = rospy.Subscriber(
            "/bigman/sensor/head_imu", senMsg.Imu, self.head_imu_callback)
        self.head_imu = None

        ### Force-Torque sensors
        self.sub_ft_LWrist = rospy.Subscriber(
            "/bigman/ft_sensor/LWrj2", geoMsg.WrenchStamped, self.ft_LWrist_callback)
        self.ft_LWrist = None

        self.sub_ft_RWrist = rospy.Subscriber(
            "/bigman/ft_sensor/RWrj2", geoMsg.WrenchStamped, self.ft_RWrist_callback)
        self.ft_RWrist = None

        self.sub_ft_LAnkle = rospy.Subscriber(
            "/bigman/ft_sensor/LAnkle", geoMsg.WrenchStamped, self.ft_LAnkle_callback)
        self.ft_LAnkle = None

        self.sub_ft_RAnkle = rospy.Subscriber(
            "/bigman/ft_sensor/RAnkle", geoMsg.WrenchStamped, self.ft_RAnkle_callback)
        self.ft_RAnkle = None

        ### Cameras
        self.sub_Lcamera = rospy.Subscriber(
            "/multisense/camera/left/image_raw", senMsg.Image, self.camera_left_callback)
        self.Lcamera = None

        self.sub_Rcamera = rospy.Subscriber(
            "/multisense/camera/right/image_raw", senMsg.Image, self.camera_right_callback)
        self.Rcamera = None


    def joints_callback(self, msg):
        pass

    def imu_callback(self, msg):
        pass

    def ft_callback(self, msg):
        pass

    def camera_callback(self, msg):
        pass


# if __name__ == '__main__':

#     node = WalkmanListener()

#     rospy.loginfo("Running walkman listener...")

#     rospy.spin()