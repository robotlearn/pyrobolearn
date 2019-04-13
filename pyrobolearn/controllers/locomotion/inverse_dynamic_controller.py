#!/usr/bin/env python
"""Provide the inverse dynamic controller for locomotion.

The inverse dynamic controller is a low-level controller that uses quadratic programming to solve several dynamic
tasks and constraints.

References:
    [1] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Songyan Xin, 2018
"""

import numpy as np
from scipy import linalg
from qpsolvers import solve_qp

import rbdl

from utils.task import Task
from utils.robot_param import RobotParam
from utils.geometry import quaternionPD, posePD

from pyrobolearn.controllers.controller import Controller


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


class InverseDynamicController(Controller):
    r"""Inverse Dynamic Controller

    The inverse dynamic controller is a low-level controller that uses quadratic programming to solve several dynamic
    tasks and constraints.

    References:
        [1] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Songyan Xin, 2018
    """

    def __init__(self, urdf_path=''):
        super(InverseDynamicController, self).__init__()

        # # params
        # self.robot_name = rospy.get_param("/robot_name")
        # self.joint_controller_name = rospy.get_param("/joint_controller_name")
        # self.actuated_joint_names = rospy.get_param("/" + self.robot_name + "/" + self.joint_controller_name + "/joints")

        # model load from rosparam
        # self.urdf_string = rospy.get_param("/robot_description")
        # self.rbdl_model = rbdl.URDFReadFromString(self.urdf_string, verboase=False, floating_base=True)

        # load model from file
        # urdf_path = '/home/xin/codes/agile_robot/envs/robots/urdfs/cogimon_urdf/cogimon.urdf'

        self.rbdl_model = rbdl.loadModel(urdf_path, verboase=False, floating_base=True)
        self.N = self.rbdl_model.dof_count

        # robot params
        self.robot_param = RobotParam(urdf_path)

    def __call__(self, robot_state, high_level_cmd):
        low_level_cmd = None
        return low_level_cmd

    def CalcJointTorqueNewFormulation(self, robot_state, high_level_cmd):
        cmd = high_level_cmd
        # cmd.show()

        if cmd.contact_state == "noSupport":
            num_of_contacts = 0
        elif cmd.contact_state == "leftSupport":
            num_of_contacts = 1
            J_contact = robot_state.lsole.J
            Jdqd_contact = robot_state.lsole.Jdqd
        elif cmd.contact_state == "rightSupport":
            num_of_contacts = 1
            J_contact = robot_state.rsole.J
            Jdqd_contact = robot_state.rsole.Jdqd
        elif cmd.contact_state == "doubleSupport":
            num_of_contacts = 2
            J_contact = np.vstack((robot_state.lsole.J, robot_state.rsole.J))
            Jdqd_contact = np.hstack((robot_state.lsole.Jdqd, robot_state.rsole.Jdqd))

        # least square task: minimize qdd and GRFs
        ls_A = np.identity(self.N + num_of_contacts * 6)
        ls_b = np.zeros(self.N + num_of_contacts * 6)
        ls_task = Task(ls_A, ls_b)

        # minimize joint torque task: tau
        if cmd.contact_state == "noSupport":
            min_torque_A = robot_state.inertia_matrix
            min_torque_b = - robot_state.nonlinear_effects
        else:
            min_torque_A = np.hstack((robot_state.inertia_matrix, -J_contact.T))
            min_torque_b = - robot_state.nonlinear_effects
        min_torque_task = Task(min_torque_A, min_torque_b)

        # minimize qdd:
        min_qdd_A = np.hstack((np.identity(self.N), np.zeros((self.N, num_of_contacts * 6))))
        min_qdd_b = np.zeros(self.N)
        min_qdd_task = Task(min_qdd_A, min_qdd_b)

        # minmize GRF
        if cmd.contact_state == "noSupport":
            min_GRF_task = None
        else:
            min_GRF_A = np.hstack((np.zeros((num_of_contacts * 6,self.N)), np.identity(num_of_contacts * 6)))
            min_GRF_b = np.zeros(num_of_contacts * 6)
            min_GRF_task = Task(min_GRF_A, min_GRF_b)


        # foot tracking task
        if cmd.contact_state == "noSupport":
            lsole_acc = posePD(pose_des=cmd.lsole_pose, pose_cur=robot_state.lsole.pose,
                               spatial_velocity_des=cmd.lsole_spatial_velocity,
                               spatial_velocity_cur=robot_state.lsole.spatial_velocity,
                               kp_linear=1000, kd_linear=2.0 * np.sqrt(100),
                               kp_angular=1000, kd_angular=2.0 * np.sqrt(10))

            rsole_acc = posePD(pose_des=cmd.rsole_pose, pose_cur=robot_state.rsole.pose,
                               spatial_velocity_des=cmd.rsole_spatial_velocity,
                               spatial_velocity_cur=robot_state.rsole.spatial_velocity,
                               kp_linear=1000, kd_linear=2.0 * np.sqrt(100),
                               kp_angular=1000, kd_angular=2.0 * np.sqrt(10))

            feet_track_A = np.vstack((robot_state.lsole.J, robot_state.rsole.J))
            feet_track_b = np.hstack((lsole_acc, rsole_acc)) - np.hstack((robot_state.lsole.Jdqd, robot_state.rsole.Jdqd))
            feet_track_task = Task(feet_track_A, feet_track_b)


        elif cmd.contact_state == "leftSupport":
            rsole_acc = posePD(pose_des=cmd.rsole_pose, pose_cur=robot_state.rsole.pose,
                               spatial_velocity_des=cmd.rsole_spatial_velocity,
                               spatial_velocity_cur=robot_state.rsole.spatial_velocity,
                               kp_linear=500, kd_linear=2.0 * np.sqrt(500),
                               kp_angular=1000, kd_angular=2.0 * np.sqrt(500))
            rfoot_track_A = np.hstack((robot_state.rsole.J, np.zeros((6, num_of_contacts * 6))))
            rfoot_track_b = rsole_acc - robot_state.rsole.Jdqd
            rfoot_track_task = Task(rfoot_track_A, rfoot_track_b)


        elif cmd.contact_state == "rightSupport":
            lsole_acc = posePD(pose_des=cmd.lsole_pose, pose_cur=robot_state.lsole.pose,
                               spatial_velocity_des=cmd.lsole_spatial_velocity,
                               spatial_velocity_cur=robot_state.lsole.spatial_velocity,
                               kp_linear=500, kd_linear=2.0 * np.sqrt(500),
                               kp_angular=1000, kd_angular=2.0 * np.sqrt(500))
            lfoot_track_A = np.hstack((robot_state.lsole.J, np.zeros((6, num_of_contacts * 6))))
            lfoot_track_b = lsole_acc - robot_state.lsole.Jdqd
            lfoot_track_task = Task(lfoot_track_A, lfoot_track_b)



        elif cmd.contact_state == "doubleSupport":
            lsole_acc = np.zeros(6)
            rsole_acc = np.zeros(6)
            lsole_acc = - 2.0 * np.sqrt(10) * robot_state.lsole.spatial_velocity
            rsole_acc = - 2.0 * np.sqrt(10) * robot_state.rsole.spatial_velocity
            # lsole_acc = posePD(pose_des=cmd.lsole_pose, pose_cur=robot_state.lsole.pose,
            #                    spatial_velocity_des=cmd.lsole_spatial_velocity,
            #                    spatial_velocity_cur=robot_state.lsole.spatial_velocity,
            #                    kp_linear=0, kd_linear=2.0 * np.sqrt(10),
            #                    kp_angular=0, kd_angular=2.0 * np.sqrt(10))
            # rsole_acc = posePD(pose_des=cmd.rsole_pose, pose_cur=robot_state.rsole.pose,
            #                    spatial_velocity_des=cmd.rsole_spatial_velocity,
            #                    spatial_velocity_cur=robot_state.rsole.spatial_velocity,
            #                    kp_linear=0, kd_linear=2.0 * np.sqrt(10),
            #                    kp_angular=0, kd_angular=2.0 * np.sqrt(10))

            feet_damp_A = np.hstack((np.vstack((robot_state.lsole.J, robot_state.rsole.J)), np.zeros((12, num_of_contacts * 6))))
            feet_damp_b = np.hstack((lsole_acc, rsole_acc)) - np.hstack((robot_state.lsole.Jdqd, robot_state.rsole.Jdqd))
            feet_damp_task = Task(feet_damp_A, feet_damp_b)


        # foot_A = np.hstack((np.vstack((robot_state.lsole.J, robot_state.rsole.J)), np.zeros((12, num_of_contacts * 6))))
        # foot_b = np.hstack((lsole_acc, rsole_acc)) - np.hstack((robot_state.lsole.Jdqd, robot_state.rsole.Jdqd))
        # foot_task = Task(foot_A, foot_b)

        # centroidal dynamic task
        kp_linear = 300
        kp_angular = 10
        des_linear_momentum = robot_state.mass * (
                kp_linear * (cmd.com_pose[:3] - robot_state.com) + 2 * np.sqrt(kp_linear) * (
                cmd.com_spatial_velocity[-3:] - robot_state.com_velocity) + cmd.com_spatial_acceleration[-3:])
        des_angular_momentum = kp_angular * cmd.angular_momentum - 2 * np.sqrt(
            kp_angular) * robot_state.angular_momentum

        # angular momentum task
        angular_momentum_A = np.hstack((robot_state.CMM[:3, :], np.zeros((3, num_of_contacts * 6))))
        angular_momentum_b = des_angular_momentum - robot_state.CMM_bias_force[:3]
        angular_momentum_task = Task(angular_momentum_A, angular_momentum_b)

        # linear momentum task
        linear_momentum_A = np.hstack((robot_state.CMM[-3:, :], np.zeros((3, num_of_contacts * 6))))
        linear_momentum_b = des_linear_momentum - robot_state.CMM_bias_force[-3:]
        linear_momentum_task = Task(linear_momentum_A, linear_momentum_b)

        # pelvis orientation task
        pelvis_orientation_A = np.hstack((robot_state.waist.J[:3, :], np.zeros((3, num_of_contacts * 6))))
        pelvis_orientation_b = quaternionPD(quat_des=cmd.com_pose[-4:], quat_cur=robot_state.waist.quaternion,
                                            omega_des=cmd.com_spatial_velocity[:3],
                                            omega_cur=robot_state.waist.spatial_velocity[:3],
                                            kp=1000, kd=2.0 * np.sqrt(500))
        pelvis_orientation_task = Task(pelvis_orientation_A, pelvis_orientation_b)

        # set task weight
        # ls_task.set_weight(50)
        # pelvis_orientation_task.set_weight(100)
        # linear_momentum_task.set_weight(300)
        # angular_momentum_task.set_weight(100)


        # choose task combination
        if cmd.contact_state == "noSupport":
            ls_task.set_weight(10)
            min_qdd_task.set_weight(10)
            min_torque_task.set_weight(10)
            feet_track_task.set_weight(100)
            pelvis_orientation_task.set_weight(10)
            tasks = [min_qdd_task, min_torque_task, feet_track_task, pelvis_orientation_task]

        elif cmd.contact_state == "doubleSupport":
            ls_task.set_weight(50) # = min_qdd + min_GRF
            min_torque_task.set_weight(50)
            min_qdd_task.set_weight(50)
            min_GRF_task.set_weight(50)
            pelvis_orientation_task.set_weight(100)
            linear_momentum_task.set_weight(300)
            feet_damp_task.set_weight(100)
            angular_momentum_task.set_weight(100)
            tasks = [ls_task, feet_damp_task, pelvis_orientation_task, linear_momentum_task, angular_momentum_task]

        elif cmd.contact_state == "leftSupport":
            ls_task.set_weight(1)
            rfoot_track_task.set_weight(300)
            pelvis_orientation_task.set_weight(10)
            linear_momentum_task.set_weight(300)
            angular_momentum_task.set_weight(10)
            tasks = [ls_task, rfoot_track_task, pelvis_orientation_task, linear_momentum_task, angular_momentum_task]

        elif cmd.contact_state == "rightSupport":
            ls_task.set_weight(1)
            lfoot_track_task.set_weight(300)
            pelvis_orientation_task.set_weight(10)
            linear_momentum_task.set_weight(300)
            angular_momentum_task.set_weight(10)
            tasks = [ls_task, lfoot_track_task, pelvis_orientation_task, linear_momentum_task, angular_momentum_task]

        # combine all task A and b matrices
        A = np.empty((0, self.N + num_of_contacts * 6))
        b = np.empty(0)
        for task in tasks:
            A = np.vstack((A, task.w * task.A))
            b = np.hstack((b, task.w * task.b))

        ############################
        #  Inequality constraints  #
        ############################

        if cmd.contact_state == "noSupport":
            inequalityConsMatrix = np.zeros((9, self.N + num_of_contacts * 6))
            inequalityConsVector = np.zeros(9)

        elif cmd.contact_state == "leftSupport":
            lGRFCons = self.robot_param.SpatialForceCons.dot(
                linalg.block_diag(robot_state.lsole.rot.T, robot_state.lsole.rot.T))
            inequalityConsMatrix = np.hstack((np.zeros((9, self.N)), lGRFCons))
            inequalityConsVector = np.zeros(9)

        elif cmd.contact_state == "rightSupport":
            rGRFCons = self.robot_param.SpatialForceCons.dot(
                linalg.block_diag(robot_state.rsole.rot.T, robot_state.rsole.rot.T))
            inequalityConsMatrix = np.hstack((np.zeros((9, self.N)), rGRFCons))
            inequalityConsVector = np.zeros(9)

        elif cmd.contact_state == "doubleSupport":
            lGRFCons = self.robot_param.SpatialForceCons.dot(
                linalg.block_diag(robot_state.lsole.rot.T, robot_state.lsole.rot.T))
            rGRFCons = self.robot_param.SpatialForceCons.dot(
                linalg.block_diag(robot_state.rsole.rot.T, robot_state.rsole.rot.T))
            GRFCons = np.vstack((np.hstack((lGRFCons, np.zeros((9, 6)))),
                                 np.hstack((np.zeros((9, 6)), rGRFCons))))
            inequalityConsMatrix = np.hstack((np.zeros((18, self.N)), GRFCons))
            inequalityConsVector = np.zeros(18)

        ########################
        # Equality constraints #
        ########################
        # dynamic constraints

        if cmd.contact_state == "noSupport":
            dynamicConsMatrix = robot_state.inertia_matrix[:6, :]  # 6*N
            dynamicConsVector = - robot_state.nonlinear_effects[:6]  # 6*1
        else:
            dynamicConsMatrix = np.hstack((robot_state.inertia_matrix[:6, :], -J_contact.T[:6, :]))  # 6*(N+12)
            dynamicConsVector = - robot_state.nonlinear_effects[:6]  # 6*1

        # # foot no movement constraints
        # feetFixConsMatrix = np.hstack((J_contact, np.zeros((12, num_of_contacts * 6))))
        # feetFixConsVector = - Jdqd_contact
        #
        #
        # equalityConsMatrix = np.vstack((dynamicConsMatrix, feetFixConsMatrix))
        # equalityConsVector = np.hstack((dynamicConsVector, feetFixConsVector))

        equalityConsMatrix = dynamicConsMatrix
        equalityConsVector = dynamicConsVector

        ##########################
        #  Solve the QP problem  #
        ##########################
        X = solve_qp(P=A.T.dot(A), q=-A.T.dot(b),
                     G=inequalityConsMatrix, h=inequalityConsVector,
                     A=equalityConsMatrix, b=equalityConsVector,
                     solver='quadprog')

        # print("X.shape:", X.shape)
        qdd = X[:self.N]
        GRF = X[self.N:]

        ###################
        # Inverse Dynamic #
        ###################
        if cmd.contact_state == "noSupport":
            tau_N = robot_state.inertia_matrix.dot(qdd) + robot_state.nonlinear_effects  # N
            tau_n = tau_N[6:]
        else:
            tau_N = robot_state.inertia_matrix.dot(qdd) + robot_state.nonlinear_effects - J_contact.T.dot(GRF)  # N
            tau_n = tau_N[6:]

        return tau_n

