import rospy
from std_msgs import msg as stdMsg

import sys
import numpy as np

__author__ = "Brian Delhaisse"
__email__ = "Brian.Delhaisse@iit.it"
__credits__ = ["Brian Delhaisse"]
__version__ = "1.0.0"
__date__ = "01/08/2016"


# ROS - Gazebo
class WalkmanPublisher:
    """
    ROS node that publishes joint positions/torques to the walkman controllers.

    Examples
    --------

    Send zero position value to the controllers:

    >>> import numpy as np
    >>> node = WalkmanPublisher()
    >>> r = rospy.Rate(10)  # 10hz

    >>> rospy.loginfo("Running walkman publisher...")
    >>> while not rospy.is_shutdown():
    ... node.send(np.zeros(31))
    ... r.sleep()

    """

    def __init__(self, controller_type='position', joints=['NeckPitchj',
        'NeckYawj', 'WaistLat', 'WaistSag', 'WaistYaw', 'LShLat', 'LShSag',
        'LShYaw', 'RShLat', 'RShSag', 'RShYaw', 'LElbj', 'RElbj',
        'LForearmPlate', 'RForearmPlate', 'LWrj1', 'LWrj2', 'RWrj1',
        'RWrj2', 'LHipLat', 'LHipSag', 'LHipYaw', 'RHipLat', 'RHipSag',
        'RHipYaw', 'LKneeSag', 'RKneeSag', 'LAnkLat', 'LAnkSag', 'RAnkLat',
        'RAnkSag']):

        rospy.init_node('WalkmanPublisher')

        joints = set(joints)
        self.publishers = {}

        if 'NeckPitchj' in joints:
            self.publishers['NeckPitchj'] = rospy.Publisher(
                "/bigman/NeckPitchj_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'NeckYawj' in joints:
            self.publishers['NeckYawj'] = rospy.Publisher(
                "/bigman/NeckYawj_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'WaistLat' in joints:
            self.publishers['WaistLat'] = rospy.Publisher(
                "/bigman/WaistLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'WaistSag' in joints:
            self.publishers['WaistSag'] = rospy.Publisher(
                "/bigman/WaistSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'WaistYaw' in joints:
            self.publishers['WaistYaw'] = rospy.Publisher(
                "/bigman/WaistYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LShLat' in joints:
            self.publishers['LShLat'] = rospy.Publisher(
                "/bigman/LShLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LShSag' in joints:
            self.publishers['LShSag'] = rospy.Publisher(
                "/bigman/LShSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LShYaw' in joints:
            self.publishers['LShYaw'] = rospy.Publisher(
                "/bigman/LShYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'RShLat' in joints:
            self.publishers['RShLat'] = rospy.Publisher(
                "/bigman/RShLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RShSag' in joints:
            self.publishers['RShSag'] = rospy.Publisher(
                "/bigman/RShSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RShYaw' in joints:
            self.publishers['RShYaw'] = rospy.Publisher(
                "/bigman/RShYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LElbj' in joints:
            self.publishers['LElbj'] = rospy.Publisher(
                "/bigman/LElbj_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RElbj' in joints:
            self.publishers['RElbj'] = rospy.Publisher(
                "/bigman/RElbj_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LForearmPlate' in joints:
            self.publishers['LForearmPlate'] = rospy.Publisher(
                "/bigman/LForearmPlate_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RForearmPlate' in joints:
            self.publishers['RForearmPlate'] = rospy.Publisher(
                "/bigman/RForearmPlate_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LWrj1' in joints:
            self.publishers['LWrj1'] = rospy.Publisher(
                "/bigman/LWrj1_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LWrj2' in joints:
            self.publishers['LWrj2'] = rospy.Publisher(
                "/bigman/LWrj2_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RWrj1' in joints:
            self.publishers['RWrj1'] = rospy.Publisher(
                "/bigman/RWrj1_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RWrj2' in joints:
            self.publishers['RWrj2'] = rospy.Publisher(
                "/bigman/RWrj2_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LHipLat' in joints:
            self.publishers['LHipLat'] = rospy.Publisher(
                "/bigman/LHipLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LHipSag' in joints:
            self.publishers['LHipSag'] = rospy.Publisher(
                "/bigman/LHipSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LHipYaw' in joints:
            self.publishers['LHipYaw'] = rospy.Publisher(
                "/bigman/LHipYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RHipLat' in joints:
            self.publishers['RHipLat'] = rospy.Publisher(
                "/bigman/RHipLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RHipSag' in joints:
            self.publishers['RHipSag'] = rospy.Publisher(
                "/bigman/RHipSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RHipYaw' in joints:
            self.publishers['RHipYaw'] = rospy.Publisher(
                "/bigman/RHipYaw_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LKneeSag' in joints:
            self.publishers['LKneeSag'] = rospy.Publisher(
                "/bigman/LKneeSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RKneeSag' in joints:
            self.publishers['RKneeSag'] = rospy.Publisher(
                "/bigman/RKneeSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)

        if 'LAnkLat' in joints:
            self.publishers['LAnkLat'] = rospy.Publisher(
                "/bigman/LAnkLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'LAnkSag' in joints:
            self.publishers['LAnkSag'] = rospy.Publisher(
                "/bigman/LAnkSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RAnkLat' in joints:
            self.publishers['RAnkLat'] = rospy.Publisher(
                "/bigman/RAnkLat_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)
        if 'RAnkSag' in joints:
            self.publishers['RAnkSag'] = rospy.Publisher(
                "/bigman/RAnkSag_"+controller_type+"_controller/command",
                stdMsg.Float64, queue_size=10)


    def send(self, msgs={}):
        for k,v in msgs.items():
            self.publishers[k].publish(stdMsg.Float64(v))

    def send_to_all(self, msg):
        for pub in self.publishers.values():
            pub.publish(stdMsg.Float64(msg))



# if __name__ == '__main__':

#     node = WalkmanPublisher()

#     if len(sys.argv) > 1:
#         r = rospy.Rate(10)  # 10hz
#     else:
#         r = rospy.Rate(sys.argv[1])

#     rospy.loginfo("Running walkman publisher...")

#     while not rospy.is_shutdown():
#         # Send zeros to walkman
#         node.send_to_all(np.zeros(31))
#         r.sleep()