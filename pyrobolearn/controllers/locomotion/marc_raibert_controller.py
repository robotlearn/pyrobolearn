#!/usr/bin/env python
"""Provide the Marc Raibert's controller for locomotion.
"""

import rospy
import numpy as np
from custom_srv.srv import *
from util import sigmoid


__author__ = ["Songyan Xin", "Brian Delhaisse"]
# S.X. wrote the main initial code
# B.D. integrated it in the PRL framework, cleaned it, added the documentation, and made it more modular and flexible
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def MarcRaibertFootPlacement(com_vel_des, com_vel, K, T):
    shift = com_vel * T / 2.0 + K * (com_vel - com_vel_des)
    return shift


class MarcRaibertController(object):
    r"""Marc Raibert Controller

    """

    def __init__(self, des_com_vel, Kp=[0.3, 0.3, 0.0], Kd=[0.1, 0.1, 0.0]):
        self.des_com_vel = np.array(des_com_vel)
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)
        self.update_des_com_vel_server = rospy.Service('update_des_com_vel', PassVector, self.handle_update_des_com_vel)
        self.update_K_server = rospy.Service('update_K', PassVector, self.handle_update_K)

    def __call__(self, cur_com_vel, T, leg_length, step_count):
        # if T == 0.0:
        #     feedback = self.Kp * (cur_com_vel - self.des_com_vel) - self.Kd * cur_com_vel
        #     shift = feedback
        #     print "shift: ", shift
        # else:
        #     neutral_point = cur_com_vel * T / 2.0
        #     neutral_point[2] = 0.0
        #     feedback = self.Kp * (cur_com_vel - self.des_com_vel) - self.Kd*cur_com_vel
        #     shift = (neutral_point + feedback)*sigmoid(self.count, shift_x= 0.0, scale_x = 0.5,shift_y=-1.0, scale_y=2.0)
        #     print "shift: ", shift, " = ", neutral_point, "+ ", feedback

        neutral_point = cur_com_vel * T / 2.0
        neutral_point[2] = 0.0
        feedback = self.Kp * (cur_com_vel - self.des_com_vel) - self.Kd * cur_com_vel
        shift = (neutral_point + feedback) * sigmoid(step_count, shift_x=0.0, scale_x=0.5, shift_y=-1.0, scale_y=2.0)
        shift[2] = - np.sqrt(leg_length ** 2 - shift[0] ** 2 - shift[1] ** 2)
        # print "step_count: ", step_count, "shift: ", shift, " = ", neutral_point, "+ ", feedback

        return shift

    def update_des_com_vel(self, des_com_vel):
        self.des_com_vel = des_com_vel

    def update_K(self, K):
        self.K = K

    def handle_update_des_com_vel(self, request):
        print "request: ", request
        self.des_com_vel = np.array([request.vector.x, request.vector.y, request.vector.z])
        return PassVectorResponse(1, "MarcRaibertController: des_com_vel updated!")

    def handle_update_K(self, request):
        print "request: ", request
        self.K = np.array([request.vector.x, request.vector.y, request.vector.z])
        return PassVectorResponse(1, "MarcRaibertController: K updated!")
