# -*- coding: utf-8 -*-
import numpy as np


class HighLevelCommand(object):
    def __init__(self, com_pos=[0, 0, 0], com_quat=[0, 0, 0, 1],
                 lsole_pos=[0, 0, 0], lsole_quat=[0, 0, 0, 1],
                 rsole_pos=[0, 0, 0], rsole_quat=[0, 0, 0, 1],
                 com_vel_linear=np.zeros(3),
                 com_vel_angular=np.zeros(3),
                 com_acc_linear=np.zeros(3),
                 com_acc_angular=np.zeros(3),
                 lsole_vel_linear=np.zeros(3),
                 lsole_vel_angular=np.zeros(3),
                 lsole_acc_linear=np.zeros(3),
                 lsole_acc_angular=np.zeros(3),
                 rsole_vel_linear=np.zeros(3),
                 rsole_vel_angular=np.zeros(3),
                 rsole_acc_linear=np.zeros(3),
                 rsole_acc_angular=np.zeros(3),
                 angular_momentum=np.zeros(3),
                 contact_state=""):

        self.com_pos = com_pos
        self.com_quat = com_quat
        self.com_pose = np.concatenate((com_pos, com_quat))

        self.com_vel_linear = com_vel_linear
        self.com_vel_angular = com_vel_angular
        self.com_spatial_velocity = np.concatenate((com_vel_angular, com_vel_linear))

        self.com_acc_linear = com_acc_linear
        self.com_acc_angular = com_acc_angular
        self.com_spatial_acceleration = np.concatenate((com_acc_angular, com_acc_linear))

        self.lsole_pos = lsole_pos
        self.lsole_quat = lsole_quat
        self.lsole_pose = np.concatenate((lsole_pos, lsole_quat))

        self.lsole_vel_linear = lsole_vel_linear
        self.lsole_vel_angular = lsole_vel_angular
        self.lsole_spatial_velocity = np.concatenate((lsole_vel_angular, lsole_vel_linear))

        self.lsole_acc_linear = lsole_acc_linear
        self.lsole_acc_angular = lsole_acc_angular
        self.lsole_spatial_acceleration = np.concatenate((lsole_acc_angular,lsole_acc_linear))

        self.rsole_pos = rsole_pos
        self.rsole_quat = rsole_quat
        self.rsole_pose = np.concatenate((rsole_pos, rsole_quat))

        self.rsole_vel_linear = rsole_vel_linear
        self.rsole_vel_angular = rsole_vel_angular
        self.rsole_spatial_velocity = np.concatenate((rsole_vel_angular, rsole_vel_linear))

        self.rsole_acc_linear = rsole_acc_linear
        self.rsole_acc_angular = rsole_acc_angular
        self.rsole_spatial_acceleration = np.concatenate((rsole_acc_angular, rsole_acc_linear))

        self.angular_momentum = angular_momentum

        self.contact_state = contact_state

    def show(self):
        print("-" * 30)
        print("[HighLevelCommand]")
        print("contact_state: ", self.contact_state)
        print("com_pos:", self.com_pos)
        print("lsole_pos:", self.lsole_pos)
        print("rsole_pos:", self.rsole_pos)
        print("-" * 30)
