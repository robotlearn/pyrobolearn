#!/usr/bin/env python
"""Provide the inverse kinematic controller for locomotion.

The inverse kinematic controller is a low-level controller that uses quadratic programming to solve several kinematic
tasks and constraints.

References:
    [1] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Songyan Xin, 2018
"""

import os
import tf
import rbdl
import cvxopt
import numpy as np
from cvxopt import matrix, solvers

from utils.utils import *
from utils.geometry import PositionPD, QuaternionPD, PosePD

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


class InverseKinematicController(Controller):
    r"""Inverse Kinematics Controller

    The inverse kinematic controller is a low-level controller that uses quadratic programming to solve several
    kinematic tasks and constraints.

    References:
        [1] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Songyan Xin, 2018
    """

    def __init__(self):
        super(InverseKinematicController, self).__init__()

    def jacobian_stack(self, robot, cmd):

        Xd_lsole = PosePD(pose_des=cmd.lsole_pose, pose_cur=robot.state.lsole.pose, kp=100.0, kd=0.0)
        Xd_rsole = PosePD(pose_des=cmd.rsole_pose, pose_cur=robot.state.rsole.pose, kp=100.0, kd=0.0)
        Xd_com = robot.state.mass * PositionPD(pos_des=cmd.com_pose.position, pos_cur=robot.state.com_pos, vel_cur=robot.state.com_vel, kp=300.0, kd=1.0)
        Xd_waist = QuaternionPD(quat_des=cmd.com_pose.quaternion, quat_cur=robot.state.waist.quaternion, kp=100.0, kd=10.0)

        if cmd.contact_state is ContactState.doubleSupport:
            # stack all task jacobians to get J
            J = np.vstack((robot.state.lsole.J, robot.state.rsole.J, robot.state.CMM[-3:, :], robot.state.waist.J[:3, :]))
            Xd = np.hstack((Xd_lsole, Xd_rsole, Xd_com, Xd_waist))
            qd_cmd = np.linalg.pinv(J).dot(Xd)

        elif cmd.contact_state is ContactState.leftSupport:
            # task of first priority
            J_1 = np.vstack((robot.state.lsole.J, robot.state.CMM[-3:, :], robot.state.waist.J[:3, :]))
            Xd_1 = np.hstack((Xd_lsole, Xd_com, Xd_waist))
            # task of second priority
            J_2 = robot.state.rsole.J
            Xd_2 = Xd_rsole
            qd_cmd = np.linalg.pinv(J_1).dot(Xd_1) + nullspace(J_1).dot(np.linalg.pinv(J_2)).dot(Xd_2)

        elif cmd.contact_state is ContactState.rightSupport:
            # task of first priority
            J_1 = np.vstack((robot.state.rsole.J, robot.state.CMM[-3:, :], robot.state.waist.J[:3, :]))
            Xd_1 = np.hstack((Xd_rsole, Xd_com, Xd_waist))
            # task of second priority
            J_2 = robot.state.lsole.J
            Xd_2 = Xd_lsole
            qd_cmd = np.linalg.pinv(J_1).dot(Xd_1) + nullspace(J_1).dot(np.linalg.pinv(J_2)).dot(Xd_2)

        elif cmd.contact_state is ContactState.noSupport:
            J = np.vstack((robot.state.CMM[-3:, :], robot.state.waist.J[:3, :]))
            Xd = np.hstack((Xd_com, Xd_waist))
            qd_cmd = np.linalg.pinv(J).dot(Xd)

        return qd_cmd[6:]

    def nullspace_projection(self, robot, cmd):

        Xd_com = robot.state.mass * PositionPD(pos_des=cmd.com_pose.position, pos_cur=robot.state.com_pos, vel_des=cmd.com_vel, vel_cur=robot.state.com_vel, kp=300.0, kd=1.0)
        Xd_waist = QuaternionPD(quat_des=cmd.com_pose.quaternion, quat_cur=robot.state.waist.quaternion, kp=100.0, kd=1.0)

        # task of first priority
        J_1 = np.vstack((robot.state.lsole.J, robot.state.rsole.J, robot.state.CMM[-3:, :]))
        Xd_1 = np.hstack((np.zeros(6), np.zeros(6), Xd_com))

        # task of second priority
        J_2 = robot.state.waist.J[:3, :]
        Xd_2 = Xd_waist

        qd_cmd = np.linalg.pinv(J_1).dot(Xd_1) + nullspace(J_1).dot(np.linalg.pinv(J_2)).dot(Xd_2)

        return qd_cmd[6:]

    def qp(self, robot, cmd):

        # define all tasks

        # 1. least square task
        A_ls = np.identity(robot.N)
        b_ls = np.zeros(robot.N)

        # 2. com task
        kp = 100
        A_com = robot.state.CMM[-3:, :]
        b_com = robot.state.mass * (kp*(cmd.com_pos-robot.state.com_pos) - 2*np.sqrt(kp)*robot.state.com_vel)

        # 3. base orientation task
        A_base = robot.state.waist.J[:3, :]
        b_base = RotationPD(cmd.waist_rot, robot.state.waist.rot, np.zeros(3), robot.state.waist.angular_vel, np.zeros(3), kp=500)

        # 4. lsole task
        A_lsole = robot.state.lsole.J
        b_lsole = SE3PD(cmd.lsole_pos, cmd.lsole_rot,
                              robot.state.lsole.pos, robot.state.lsole.rot, np.hstack((np.zeros(3), cmd.lsole_vel)),
                              robot.state.lsole.vel,
                              np.zeros(6),
                              200, 200)

        # 5. rsole task
        A_rsole = robot.state.rsole.J
        b_rsole = SE3PD(cmd.rsole_pos, cmd.rsole_rot,
                              robot.state.rsole.pos, robot.state.rsole.rot, np.hstack((np.zeros(3), cmd.rsole_vel)),
                              robot.state.rsole.vel,
                              np.zeros(6),
                              200, 200)

        w_ls = 1.0
        w_com = 10.0
        w_base = 10.0
        w_lsole = 0.0
        w_rsole = 0.0

        # combine all tasks
        A = np.vstack((w_ls*A_ls, w_com*A_com, w_base*A_base, w_lsole*A_lsole, w_rsole*A_rsole))
        b = np.hstack((w_ls*b_ls, w_com*b_com, w_base*b_base, w_lsole*b_lsole, w_rsole*b_rsole))

        qd_N = np.linalg.pinv(A).dot(b)
        qd_n = qd_N[6:]

        return qd_n

