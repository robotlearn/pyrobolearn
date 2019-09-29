# -*- coding: utf-8 -*-
import rospy
from std_msgs import msg as stdMsg

import sys
import numpy as np

__author__ = "Brian Delhaisse"
__email__ = "Brian.Delhaisse@iit.it"
__credits__ = ["Brian Delhaisse"]
__version__ = "1.0.0"
__date__ = "06/02/2016"


# ROS - Gazebo
class ComanPublisher:
    """
    ROS node that publishes joint positions/torques to the coman controllers.

    Examples
    --------

    Send zero position value to the controllers:

    >>> import numpy as np
    >>> node = ComanPublisher()
    >>> r = rospy.Rate(10)  # 10hz

    >>> rospy.loginfo("Running coman publisher...")
    >>> while not rospy.is_shutdown():
    ... node.send(np.zeros(29))
    ... r.sleep()

    """

    def __init__(self, init_ros=True, controller_type='position', joints=['WaistLat',
        'WaistSag', 'WaistYaw', 'LShLat', 'LShSag','LShYaw', 'RShLat',
        'RShSag', 'RShYaw', 'LElbj', 'RElbj','LForearmPlate',
        'RForearmPlate', 'LWrj1', 'LWrj2', 'RWrj1','RWrj2', 
        'LHipLat', 'LHipSag', 'LHipYaw', 'RHipLat', 'RHipSag','RHipYaw',
        'LKneeSag', 'RKneeSag', 'LAnkLat', 'LAnkSag', 'RAnkLat','RAnkSag']):

        if init_ros:
            rospy.init_node('ComanPublisher')

        joints = set(joints)
        self.publishers = {}

        if 'WaistLat' in joints:
            self.publishers['WaistLat'] = rospy.Publisher(
                "/coman/WaistLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'WaistSag' in joints:
            self.publishers['WaistSag'] = rospy.Publisher(
                "/coman/WaistSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'WaistYaw' in joints:
            self.publishers['WaistYaw'] = rospy.Publisher(
                "/coman/WaistYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LShLat' in joints:
            self.publishers['LShLat'] = rospy.Publisher(
                "/coman/LShLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LShSag' in joints:
            self.publishers['LShSag'] = rospy.Publisher(
                "/coman/LShSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LShYaw' in joints:
            self.publishers['LShYaw'] = rospy.Publisher(
                "/coman/LShYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'RShLat' in joints:
            self.publishers['RShLat'] = rospy.Publisher(
                "/coman/RShLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RShSag' in joints:
            self.publishers['RShSag'] = rospy.Publisher(
                "/coman/RShSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RShYaw' in joints:
            self.publishers['RShYaw'] = rospy.Publisher(
                "/coman/RShYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LElbj' in joints:
            self.publishers['LElbj'] = rospy.Publisher(
                "/coman/LElbj_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RElbj' in joints:
            self.publishers['RElbj'] = rospy.Publisher(
                "/coman/RElbj_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LForearmPlate' in joints:
            self.publishers['LForearmPlate'] = rospy.Publisher(
                "/coman/LForearmPlate_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RForearmPlate' in joints:
            self.publishers['RForearmPlate'] = rospy.Publisher(
                "/coman/RForearmPlate_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LWrj1' in joints:
            self.publishers['LWrj1'] = rospy.Publisher(
                "/coman/LWrj1_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LWrj2' in joints:
            self.publishers['LWrj2'] = rospy.Publisher(
                "/coman/LWrj2_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RWrj1' in joints:
            self.publishers['RWrj1'] = rospy.Publisher(
                "/coman/RWrj1_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RWrj2' in joints:
            self.publishers['RWrj2'] = rospy.Publisher(
                "/coman/RWrj2_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LHipLat' in joints:
            self.publishers['LHipLat'] = rospy.Publisher(
                "/coman/LHipLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LHipSag' in joints:
            self.publishers['LHipSag'] = rospy.Publisher(
                "/coman/LHipSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LHipYaw' in joints:
            self.publishers['LHipYaw'] = rospy.Publisher(
                "/coman/LHipYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RHipLat' in joints:
            self.publishers['RHipLat'] = rospy.Publisher(
                "/coman/RHipLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RHipSag' in joints:
            self.publishers['RHipSag'] = rospy.Publisher(
                "/coman/RHipSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RHipYaw' in joints:
            self.publishers['RHipYaw'] = rospy.Publisher(
                "/coman/RHipYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LKneeSag' in joints:
            self.publishers['LKneeSag'] = rospy.Publisher(
                "/coman/LKneeSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RKneeSag' in joints:
            self.publishers['RKneeSag'] = rospy.Publisher(
                "/coman/RKneeSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LAnkLat' in joints:
            self.publishers['LAnkLat'] = rospy.Publisher(
                "/coman/LAnkLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LAnkSag' in joints:
            self.publishers['LAnkSag'] = rospy.Publisher(
                "/coman/LAnkSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RAnkLat' in joints:
            self.publishers['RAnkLat'] = rospy.Publisher(
                "/coman/RAnkLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RAnkSag' in joints:
            self.publishers['RAnkSag'] = rospy.Publisher(
                "/coman/RAnkSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)


    def send(self, msgs={}):
        for k,v in msgs.items():
            self.publishers[k].publish(stdMsg.Float64(v))

    def send_to_all(self, msg):
        for pub in self.publishers.values():
            pub.publish(stdMsg.Float64(msg))



# if __name__ == '__main__':

#     node = ComanPublisher()

#     if len(sys.argv) > 1:
#         r = rospy.Rate(10)  # 10hz
#     else:
#         r = rospy.Rate(sys.argv[1])

#     rospy.loginfo("Running coman publisher...")

#     while not rospy.is_shutdown():
#         # Send zeros to coman
#         node.send_to_all(np.zeros(29))
#         r.sleep()