#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide feedback laws used in control.

References:
    [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010, chapter 2 and 3
    [2] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Xin, 2018
"""

import numpy as np

from pyrobolearn.utils.transformation import quaternion_error, vector_from_skew_matrix


__author__ = ["Songyan Xin (code)", "Brian Delhaisse (documentation)"]
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Songyan Xin", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def position_pd(pos_des, pos_cur, vel_des=np.zeros(3), vel_cur=np.zeros(3), acc_des=np.zeros(3), kp=100, kd=0.0):
    r"""
    Return the commanded spatial linear acceleration using the position PD feedback law given by:

    .. math:: \ddot{p}_c = \ddot{p}_d + K_D (\dot{p}_d - \dot{p}) + K_P (p_d - p)

    where :math:`\ddot{p}_c` is the commanded acceleration, :math:`\ddot{p}_d` is the desired acceleration, :math:`K_D`
    is the derivative gain, :math:`\dot{p}_d` is the desired velocity, :math:`\dot{p}` is the current velocity,
    :math:`K_P` is the proportional gain, :math:`p_d` is the desired position, and :math:`p` is the current position.

    Args:
        pos_des (np.array[3]): desired position
        pos_cur (np.array[3]): current position
        vel_des (np.array[3]): desired velocity
        vel_cur (np.array[3]): current velocity
        acc_des (np.array[3]): desired acceleration
        kp (float, np.array[3,3]): proportional gain
        kd (float, np.array[3,3]): derivative gain

    Returns:
        np.array[3]: the commanded spatial linear acceleration computed by the position PD feedback law
    """
    return kp * (pos_des - pos_cur) + kd * (vel_des - vel_cur) + acc_des


def rotation_pd(rot_des, rot_cur, omega_des=np.zeros(3), omega_cur=np.zeros(3), omega_dot_des=np.zeros(3), kp=200,
                kd=0.0):
    r"""
    Return the commanded spatial angular acceleration using the orientation PD feedback law given by:

    .. math:: \dot{\omega}_c = \dot{\omega}_d + K_D (\omega_d - \omega) + K_P e_o

    where :math:`\dot{\omega}_c` is the commanded angular acceleration, \dot{\omega}_d is the desired angular
    acceleration, :math:`K_D` is the derivative gain, :math:`\omega` is the angular velocity, :math:`K_P` is the
    proportional gain, and :math:`e_o` is the orientation error (which depends on the orientation representation we
    are using).

    If rotation matrices are given, the orientation error is given by:

    .. math:: e_o = vex(R_d R^\top - I)

    where :math:`vex` convert a skew-symmetric matrix to a vector (it is the inverse of the `skew(.)` function).

    Args:
        rot_des (np.array[3, 3]): desired rotation matrix
        rot_cur (np.array[3, 3]): current rotation matrix
        omega_des (np.array[3]): desired angular velocity
        omega_cur (np.array[3]): current angular velocity
        omega_dot_des (np.array[3]): desired angular acceleration
        kp (float, np.array[3,3]): proportional gain
        kd (float, np.array[3,3]): derivative gain

    Returns:
        np.array[3]: the commanded spatial angular acceleration computed by the orientation PD feedback law
    """
    rot_error = vector_from_skew_matrix(rot_des.dot(rot_cur.T) - np.identity(3))
    return kp * rot_error + kd * (omega_des - omega_cur) + omega_dot_des


def quaternion_pd(quat_des, quat_cur, omega_des=np.zeros(3), omega_cur=np.zeros(3), omega_dot_des=np.zeros(3), kp=100,
                 kd=0.0):
    r"""
    Return the commanded spatial angular acceleration using the orientation PD feedback law given by:

    .. math:: \dot{\omega}_c = \dot{\omega}_d + K_D (\omega_d - \omega) + K_P e_o

    where :math:`\dot{\omega}_c` is the commanded angular acceleration, \dot{\omega}_d is the desired angular
    acceleration, :math:`K_D` is the derivative gain, :math:`\omega` is the angular velocity, :math:`K_P` is the
    proportional gain, and :math:`e_o` is the orientation error (which depends on the orientation representation we
    are using).

    If quaternions are given, the orientation error is given by:

    .. math:: e_o = s v_d - s_d v - v_d \cross v

    where a quaternion is represented as an ordered pair :math:`[s, v]` with :math:`s \in \mathbb{R}` (the scalar part)
    and :math:`v \in \mathbb{R}^3` (the vector part).

    Args:
        quat_des (np.array[4]): desired quaternion [x,y,z,w]
        quat_cur (np.array[4]): current quaternion [x,y,z,w]
        omega_des (np.array[3]): desired angular velocity
        omega_cur (np.array[3]): current angular velocity
        omega_dot_des (np.array[3]): desired angular acceleration
        kp (float, np.array[3,3]): proportional gain
        kd (float, np.array[3,3]): derivative gain

    Returns:
        np.array[3]: the commanded spatial angular acceleration computed by the orientation PD feedback law
    """
    return kp * quaternion_error(quat_des=quat_des, quat_cur=quat_cur) + kd * (omega_des - omega_cur) + omega_dot_des


def pose_pd(pose_des, pose_cur, spatial_velocity_des=np.zeros(6), spatial_velocity_cur=np.zeros(6),
            spatial_acceleration_des=np.zeros(6), kp_linear=100, kd_linear=10, kp_angular=100, kd_angular=10):
    r"""
    Return the commanded spatial acceleration given by :math:`\dot{v}_c = [\dot{\omega}_c^\top, \ddot{p}_c^\top]^\top`,
    computed using the position/orientation PD feedback laws:

    .. math::

        \dot{\omega}_c &= \dot{\omega}_d + K_{D,\omega} (\omega_d - \omega) + K_{P,\omega} e_o \\
        \ddot{p}_c = \ddot{p}_d + K_{D,p} (\dot{p}_d - \dot{p}) + K_{P,p} (p_d - p),

    with :math:`[p, q] \in \mathbb{R}^7` is the pose with :math:`p \in \mathbb{R}^3` being the position, and :math:`q`
    being the quaternion, while :math:`v = [\omega, \dot{p}] \in \mathbb{R}^6` is the spatial velocity with
    :math:`\omega \in \mathbb{R}^3` being the angular velocity and :math:`\dot{p} \in \mathbb{R}^3` being the linear
    velocity.

    Args:
        pose_des (np.array[7]): desired pose :math:`[p_d, q_d]` which is the concatenation of the desired position and
            orientation  (where the later is represented as a quaternion [x,y,z,w])
        pose_cur (np.array[7]): current pose :math:`[p, q]` which is the concatenation of the current position and
            orientation (where the latter is represented as a quaternion [x,y,z,w])
        spatial_velocity_des (np.array[6]): desired spatial velocity :math:`v = [\omega, \dot{p}]`
        spatial_velocity_cur (np.array[6]): current spatial velocity :math:`v = [\omega, \dot{p}]`
        spatial_acceleration_des (np.array[6]): desired spatial acceleration :math:`\dot{v} = [\dot{\omega}, \ddot{p}]`
        kp_linear (float, np.array[3,3]): linear proportional gain
        kd_linear (float, np.array[3,3]): linear derivative gain
        kp_angular (float, np.array[3,3]): angular proportional gain
        kd_angular (float, np.array[3,3]): angular derivative gain

    Returns:
        np.array[6]: commanded spatial acceleration computed by the PD feedback law (that is the error computed by the
            PD feedback laws)
    """
    error_linear = position_pd(pos_cur=pose_cur[:3], pos_des=pose_des[:3], vel_cur=spatial_velocity_cur[-3:],
                               vel_des=spatial_velocity_des[-3:], acc_des=spatial_acceleration_des[-3:],
                               kp=kp_linear, kd=kd_linear)
    error_angular = quaternion_pd(quat_cur=pose_cur[-4:], quat_des=pose_des[-4:], omega_cur=spatial_velocity_cur[:3],
                                  omega_des=spatial_velocity_des[:3], omega_dot_des=spatial_acceleration_des[:3],
                                  kp=kp_angular, kd=kd_angular)
    error = np.concatenate((error_angular, error_linear))
    return error
