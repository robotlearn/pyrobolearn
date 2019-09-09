#!/usr/bin/env python
"""Provide utils code to transform orientation expressed in different forms

This includes rotation matrices, euler angles (RPY), axis-angle, and quaternions.

References:
    - [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010, chapter 2 and 3
    - [2] "Understanding Quaternions", https://www.3dgep.com/understanding-quaternions
    - [3] "Orientation in Cartesian Space Dynamic Movement Primitives", Ude et al., 2014
"""

import numpy as np
from scipy.linalg import block_diag
import quaternion
# from pyquaternion import Quaternion  # TODO: check API at http://kieranwynn.github.io/pyquaternion
import sympy
from collections import Iterable

from pyrobolearn.utils.converter import QuaternionNumpyConverter

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def min_angle_difference(q1, q2):
    r"""
    Return the minimum angle difference between two (set of) angles (expressed in radians). The differences are
    always between [-pi, pi].

    Args:
        q1 (float, np.array[float[N]]): first angle(s)
        q2 (float, np.array[float[N]]): second angle(s)

    Returns:
        float, np.array[float[N]]: minimum angle difference(s)
    """
    diff = np.maximum(q1, q2) - np.minimum(q1, q2)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff


def get_adjoint_from_rotation(rotation_matrix):
    r"""
    Get the adjoint matrix from a rotation matrix.

    Args:
        rotation_matrix (np.array[float[3,3]]): rotation matrix.

    Returns:
        np.array[float[6,6]]: adjoint matrix.
    """
    return block_diag(rotation_matrix, rotation_matrix)


def get_homogeneous_transform(position, orientation):
    r"""
    Return the Homogeneous transform matrix given the position vector and the orientation.

    .. math::

        H = [[R, p],
             [zeros(3),1]]

    where :math:`R` is the 3x3 rotation matrix, :math:`p` is the 3x1 position vector.

    Args:
        position (np.array[float[3]]): position vector
        orientation (np.array[float[4]], np.array[float[3,3]], np.array[float[3]]): orientation (expressed as a
            quaternion [x,y,z,w], 3x3 rotation matrix, or roll-pitch-yaw angles).

    Returns:
        np.array[float[4,4]]: homogeneous matrix
    """
    if isinstance(orientation, quaternion.quaternion):
        R = quaternion.as_rotation_matrix(orientation)
    else:
        orientation = np.asarray(orientation)
        if orientation.shape == (3,):  # RPY Euler angles
            R = get_matrix_from_rpy(orientation)
        elif orientation.shape == (4,):  # quaternion in the form [x,y,z,w]
            R = get_matrix_from_quaternion(orientation)
        elif orientation.shape == (3, 3):  # Rotation matrix
            R = orientation
        else:
            raise ValueError("Expecting a quaternion, RPY Euler angles, or rotation matrix")

    H = np.vstack((np.hstack((R, position.reshape(-1, 1))), np.array([[0, 0, 0, 1]])))
    return H


def get_inverse_homogeneous(matrix):
    r"""
    Return the inverse of the homogeneous matrix.

    If the homogeneous matrix is expressed as:

    .. math::

        H = [[R, p],
             [zeros(3), 1]],

    where :math:`R` is the 3x3 rotation matrix, :math:`p` is the 3x1 position vector. Then, the inverse homogeneous
    matrix is given by:

    .. math::

        H^{-1} = [[R^\top, -R^\top p],
                  [zeros(3), 1]].

    Args:
        matrix (np.array[float[4,4]]): homogeneous matrix to inverse.

    Returns:
        np.array[float[4,4]]: inverse homogeneous matrix.
    """
    R = matrix[:3, :3].T
    p = -R.dot(matrix[:3, 3].reshape(-1, 1))
    return np.vstack((np.hstack((R, p)),
                      np.array([[0, 0, 0, 1]])))


def homogeneous_to_pose(matrix):
    r"""
    Return a pose (7D vector: position + quaternion) from a homogeneous matrix.

    Args:
        matrix (np.array[float[4,4]]): homogeneous matrix

    Returns:
        np.array[float[7]]: pose (position + quaternion [x,y,z,w])
    """
    position = matrix[:3, -1]
    quaternion = get_quaternion_from_matrix(matrix[:3, :3])
    return np.concatenate((position, quaternion))


def pose_to_homogeneous(pose):
    r"""
    Return a homogeneous matrix from a pose (7D vector: concatenation of position and quaternion).

    Args:
        pose (np.array[float[7]]): concatenation of position and orientation (expressed as a quaternion [x,y,z,w])

    Returns:
        np.array[float[4,4]]: homogeneous matrix
    """
    pose = np.array(pose).flatten()
    position, orientation = pose[:3], pose[-4:]
    return get_homogeneous_transform(position=position, orientation=orientation)


def get_quaternion(orientation, convert_to_quat=False, convention='xyzw'):
    r"""
    Return a quaternion from an arbitrary orientation (expressed as a quaternion, rotation matrix, roll-pitch-yaw
    angles, or an axis-angle).

    Args:
        orientation (np.array[float[4]], np.array[float[3,3]], np.array[float[3]], tuple of float and
            np.array[float[3]]): orientation (expressed as a quaternion [x,y,z,w], 3x3 rotation matrix,
            roll-pitch-yaw angles, or axis-angle).
        convert_to_quat (bool): If True, it will return an instance of `quaternion.quaternion`. Otherwise, it will
            return a numpy array.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[4]], quaternion.quaternion: quaternion.
    """
    if isinstance(orientation, quaternion.quaternion):  # quaternion
        return orientation
    if isinstance(orientation, tuple) and len(orientation) == 2:  # axis-angle
        angle, axis = orientation
        if isinstance(axis, (float, int)) and isinstance(angle, np.ndarray):
            angle, axis = axis, angle
        return get_quaternion_from_axis_angle(axis, angle, convert_to_quat, convention)
    if isinstance(orientation, (np.ndarray, list)):
        orientation = np.asarray(orientation)
        if orientation.shape == (3,):  # RPY Euler angles
            return get_quaternion_from_rpy(orientation, convert_to_quat, convention)
        if orientation.shape == (4,):  # quaternion
            if convention == 'wxyz':
                x, y, z, w = orientation
                return np.array([w, x, y, z])
            return orientation
        if orientation.shape == (3, 3):  # rotation matrix
            return get_quaternion_from_matrix(orientation, convert_to_quat, convention)
        raise ValueError("Expecting the shape of the orientation to be (3,), (3,3), or (4,), instead got: "
                         "{}".format(orientation.shape))
    raise TypeError("Expecting the given orientation to be a np.ndarray, quaternion, tuple or list, instead got: "
                    "{}".format(type(orientation)))


def get_rotation_matrix(orientation):
    r"""
    Return a rotation matrix from an arbitrary orientation (expressed as a quaternion, rotation matrix, roll-pitch-yaw
    angles, or an axis-angle).

    Args:
        orientation (np.array[float[4]], np.array[float[3,3]], np.array[float[3]], tuple of float and
            np.array[float[3]]): orientation (expressed as a quaternion [x,y,z,w], 3x3 rotation matrix, roll-pitch-yaw
            angles, or axis-angle).

    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    if isinstance(orientation, quaternion.quaternion):  # quaternion
        return get_matrix_from_quaternion(orientation)
    if isinstance(orientation, tuple) and len(orientation) == 2:
        angle, axis = orientation
        if isinstance(axis, (float, int)) and isinstance(angle, np.ndarray):
            angle, axis = axis, angle
        return get_matrix_from_axis_angle(axis, angle)
    if isinstance(orientation, (np.ndarray, list)):
        orientation = np.asarray(orientation)
        if orientation.shape == (3,):  # RPY Euler angles
            return get_matrix_from_rpy(orientation)
        if orientation.shape == (4,):  # quaternion
            return get_matrix_from_quaternion(orientation)
        if orientation.shape == (3, 3):  # rotation matrix
            return orientation
        raise ValueError("Expecting the shape of the orientation to be (3,), (3,3), or (4,), instead got: "
                         "{}".format(orientation.shape))
    raise TypeError("Expecting the given orientation to be a np.ndarray, quaternion, tuple or list, instead got: "
                    "{}".format(type(orientation)))


def get_matrix_from_axis_angle(axis, angle):
    """Return the rotation matrix from the specified axis and angle.

    Args:
        axis (np.array[float[3]], list[float[3]]): 3d axis vector.
        angle (float): angle.

    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    x, y, z = axis
    a = angle
    c, s = np.cos(a), np.sin(a)
    c1 = 1 - c
    R = np.array([[x ** 2 * c1 + c, x * y * c1 - z * s, x * z * c1 + y * s],
                  [x * y * c1 + z * s, y ** 2 * c1 + c, y * z * c1 - x * s],
                  [x * z * c1 - y * s, y * z * c1 + x * s, z ** 2 * c1 + c]])
    return R


def get_symbolic_matrix_from_axis_angle(axis, angle):
    """Return the symbolic rotation matrix from the specified axis and angle.

    Args:
        axis (np.array[float[3]], list[float[3]], list[sympy.Symbol[3]]): 3d axis vector.
        angle (float, sympy.Symbol): angle.

    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    x, y, z = axis
    a = angle
    c, s = sympy.cos(a), sympy.sin(a)
    c1 = 1 - c
    R = np.array([[x**2 * c1 + c, x * y * c1 - z * s, x * z * c1 + y * s],
                  [x * y * c1 + z * s, y**2 * c1 + c, y * z * c1 - x * s],
                  [x * z * c1 - y * s, y * z * c1 + x * s, z**2 * c1 + c]])
    return R


def get_axis_angle_from_matrix(R):
    """Return the associated axis and angle from the specified rotation matrix.

    Args:
        R (np.array[float[3,3]]): 3-by-3 rotation matrix.

    Returns:
        float: angle.
        np.array[float[3]]: 3d axis vector.
    """
    angle = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.)
    axis = 1. / (2. * np.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle, axis


def get_symbolic_axis_angle_from_matrix(R):
    """Return the symbolic axis and angle from the specified rotation matrix.

    Args:
        R (np.array[sympy.Symbol[3,3]]): 3-by-3 rotation matrix.

    Returns:
        sympy.Symbol: angle.
        np.array[sympy.Symbol[3]]: 3d axis vector.
    """
    angle = sympy.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.)
    axis = 1. / (2. * sympy.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle, axis


def get_quaternion_from_axis_angle(axis, angle, convert_to_quat=False, convention='xyzw'):
    """Get the quaternion associated from the axis/angle representation.

    Args:
        axis (np.array[float[3]]): 3d axis vector.
        angle (float): angle.
        convert_to_quat (bool): If True, it will return an instance of `quaternion.quaternion`.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[4]], quaternion.quaternion: quaternion.
    """
    w = np.cos(angle / 2.)
    x, y, z = np.sin(angle / 2.) * axis
    if convert_to_quat:
        return quaternion.quaternion(w, x, y, z)
    else:
        if convention == 'xyzw':
            return np.array([x, y, z, w])
        elif convention == 'wxyz':
            return np.array([w, x, y, z])
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")


def get_symbolic_quaternion_from_axis_angle(axis, angle, convention='xyzw'):
    """Get the symbolic quaternion associated from the axis/angle representation.

    Args:
        axis (np.array[float[3]], np.array[sympy.Symbol[3]]): 3d axis vector.
        angle (float, sympy.Symbol): angle.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[4]]: symbolic quaternion.
    """
    w = sympy.cos(angle / 2.)
    x, y, z = sympy.sin(angle / 2.) * axis
    if convention == 'xyzw':
        return np.array([x, y, z, w])
    elif convention == 'wxyz':
        return np.array([w, x, y, z])
    else:
        raise NotImplementedError("Asking for a convention that has not been implemented")


def get_rpy_from_matrix(R):
    """Get the Roll-Pitch-Yaw angle values from the given rotation matrix.

    Args:
        R (np.array[float[3,3]]): 3-by-3 rotation matrix.

    Returns:
        np.array[float[3]]: roll-pitch-yaw angle values.
    """
    # r = np.arctan2(R[1, 0], R[0, 0])
    # p = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    # y = np.arctan2(R[2, 1], R[2, 2])

    r = np.arctan2(R[2, 1], R[2, 2])
    p = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    y = np.arctan2(R[1, 0], R[0, 0])

    return np.array([r, p, y])


def get_symbolic_rpy_from_matrix(R):
    """Get the symbolic Roll-Pitch-Yaw angles from the given rotation matrix.

    Args:
        R (np.array[float[3,3]], np.array[sympy.Symbol[3,3]]): symbolic 3-by-3 rotation matrix.

    Returns:
        np.array[sympy.Symbol[3]]: symbolic roll-pitch-yaw angles.
    """
    # r = sympy.atan2(R[1, 0], R[0, 0])
    # p = sympy.atan2(-R[2, 0], sympy.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    # y = sympy.atan2(R[2, 1], R[2, 2])

    r = sympy.atan2(R[2, 1], R[2, 2])
    p = sympy.atan2(-R[2, 0], sympy.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    y = sympy.atan2(R[1, 0], R[0, 0])

    return np.array([r, p, y])


def get_matrix_from_rpy(rpy):
    """Get rotation matrix from the given Roll-Pitch-Yaw angles.

    Args:
        rpy (np.array[float[3]]): roll-pitch-yaw angles

    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    cr, cp, cy = [np.cos(i) for i in rpy]
    sr, sp, sy = [np.sin(i) for i in rpy]
    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp, cp*sr, cp*cr]])
    return R


def get_symbolic_matrix_from_rpy(rpy):
    """
    Get the symbolic rotation matrix from the given Roll-Pitch-Yaw angles.

    Args:
        rpy (np.array[float[3]], np.array[sympy.Symbol[3]]): roll-pitch-yaw angles.

    Returns:
        np.array[sympy.Symbol[3,3]]: symbolic rotation matrix
    """
    cr, cp, cy = [sympy.cos(i) for i in rpy]
    sr, sp, sy = [sympy.sin(i) for i in rpy]
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    return R


def get_quaternion_from_matrix(R, convert_to_quat=False, convention='xyzw'):
    """
    Get the quaternion from the given rotation matrix.

    Args:
        R (np.array[float[3,3]]): rotation matrix.
        convert_to_quat (bool): If True, it will return an instance of `quaternion.quaternion`.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[4]], quaternion.quaternion: quaternion
    """
    w = 1./2 * np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)
    x, y, z = 1./2 * np.array([np.sign(R[2, 1] - R[1, 2]) * np.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1),
                               np.sign(R[0, 2] - R[2, 0]) * np.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1),
                               np.sign(R[1, 0] - R[0, 1]) * np.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)])
    if convert_to_quat:
        return quaternion.quaternion(w, x, y, z)
    else:
        if convention == 'xyzw':
            return np.array([x, y, z, w])
        elif convention == 'wxyz':
            return np.array([w, x, y, z])
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")


def get_symbolic_quaternion_from_matrix(R, convention='xyzw'):
    """
    Get the symbolic quaternion from the given rotation matrix.

    Args:
        R (np.array[sympy.Symbol[3,3]], np.array[float[3,3]]): (symbolic) rotation matrix
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[sympy.Symbol[4]]: symbolic quaternion.
    """
    w = 1. / 2 * sympy.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)
    x, y, z = 1. / 2 * np.array([sympy.sign(R[2, 1] - R[1, 2]) * sympy.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1),
                                 sympy.sign(R[0, 2] - R[2, 0]) * sympy.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1),
                                 sympy.sign(R[1, 0] - R[0, 1]) * sympy.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)])
    if convention == 'xyzw':
        return np.array([x, y, z, w])
    elif convention == 'wxyz':
        return np.array([w, x, y, z])
    else:
        raise NotImplementedError("Asking for a convention that has not been implemented")


def get_matrix_from_quaternion(q, convention='xyzw'):
    """
    Get rotation matrix from the given quaternion.

    Args:
        q (np.array[float[4]], quaternion.quaternion): quaternion
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    if isinstance(q, quaternion.quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w
    elif isinstance(q, Iterable):
        if convention == 'xyzw':
            x, y, z, w = q
        elif convention == 'wxyz':
            w, x, y, z = q
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")
    else:
        raise TypeError
    R = np.array([[2 * (w**2 + x**2) - 1, 2 * (x*y - w*z), 2 * (x*z + w*y)],
                  [2 * (x*y + w*z), 2 * (w**2 + y**2) - 1, 2*(y*z - w*x)],
                  [2 * (x*z - w*y), 2 * (y*z + w*x), 2 * (w**2 + z**2) - 1]])
    return R


def get_symbolic_matrix_from_quaternion(q, convention='xyzw'):
    """
    Get symbolic rotation matrix from the given quaternion.

    Args:
        q (np.array[sympy.Symbol[4]], np.array[float[4]]): (symbolic) quaternion.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[sympy.Symbol[3,3]]: symbolic rotation matrix.
    """
    return get_matrix_from_quaternion(q, convention=convention)


def get_rpy_from_quaternion(q, convention='xyzw'):
    """
    Get the Roll-Pitch-Yaw angle(s) from the given quaternion(s).

    Args:
        q (np.array[float[4]], np.array[float[N,4]], (list of) quaternion.quaternion): quaternion(s)
        convention: convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[3]], np.array[float[N,3]]: roll-pitch-yaw angles.
    """
    multiple_quaternions = True
    if isinstance(q, (np.ndarray, list, tuple)):
        q = np.asarray(q)
    if isinstance(q, quaternion.quaternion) or (isinstance(q, np.ndarray) and len(q.shape) == 1):
        multiple_quaternions = False
        q = np.array([q])  # (1,4)

    if isinstance(q[0], quaternion.quaternion):
        q = quaternion.as_float_array(q)
        x, y, z, w = q[:, 1], q[:, 2], q[:, 3], q[:, 0]  # (N,)
    elif isinstance(q, Iterable):
        if convention == 'xyzw':
            x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]  # (N,)
        elif convention == 'wxyz':
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]  # (N,)
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")
    else:
        raise TypeError

    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x**2 + y**2))   # (N,)
    pitch = np.arcsin(2 * (w*y - z*x))                          # (N,)
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2))    # (N,)
    rpy = np.vstack((roll, pitch, yaw)).T  # (N,3)

    if not multiple_quaternions:
        return rpy[0]
    return np.array(rpy)


def get_symbolic_rpy_from_quaternion(q, convention='xyzw'):
    """
    Get the symbolic Roll-Pitch-Yaw angle from the given quaternion.

    Args:
        q (np.array[float[4]], np.array[sympy.Symbol[4]]): quaternion
        convention: convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[sympy.Symbol[3]]: symbolic roll-pitch-yaw angles.
    """
    if isinstance(q, quaternion.quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w
    elif isinstance(q, Iterable):
        if convention == 'xyzw':
            x, y, z, w = q
        elif convention == 'wxyz':
            w, x, y, z = q
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")
    else:
        raise TypeError

    roll = sympy.atan2(2*(w*x + y*z), 1 - 2 * (x**2 + y**2))
    pitch = sympy.asin(2 * (w*y - z*x))
    yaw = sympy.atan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2))

    return np.array([roll, pitch, yaw])


def get_quaternion_from_rpy(rpy, convert_to_quat=False, convention='xyzw'):
    """
    Get quaternion from Roll-Pitch-Yaw angle.

    Args:
        rpy (np.array[float[3]], np.array[float[N,3]]): roll-pitch-yaw angles
        convert_to_quat (bool): If True, it will return an instance of `quaternion.quaternion`.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[4]], quaternion.quaternion: quaternion
    """
    rpy = np.asarray(rpy)
    multiple_rpy = True
    if len(rpy.shape) < 2:
        multiple_rpy = False
        rpy = np.array([rpy])  # (1,3)

    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = np.cos(r/2.), np.sin(r/2.)
    cp, sp = np.cos(p/2.), np.sin(p/2.)
    cy, sy = np.cos(y/2.), np.sin(y/2.)

    w = cr * cp * cy + sr * sp * sy  # (N,)
    x = sr * cp * cy - cr * sp * sy  # (N,)
    y = cr * sp * cy + sr * cp * sy  # (N,)
    z = cr * cp * sy - sr * sp * cy  # (N,)

    if convert_to_quat:
        q = quaternion.from_float_array(np.vstack((w, x, y, z)).T)
    else:
        if convention == 'xyzw':
            q = np.vstack([x, y, z, w]).T
        elif convention == 'wxyz':
            q = np.vstack([w, x, y, z]).T
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")

    if not multiple_rpy:
        return q[0]
    return q


def get_symbolic_quaternion_from_rpy(rpy, convention='xyzw'):
    """
    Get symbolic quaternion from Roll-Pitch-Yaw angle.

    Args:
        rpy (np.array[float[3]], np.array[sympy.Symbol[3]]): (symbolic) roll-pitch-yaw angles
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[sympy.Symbol[4]]: symbolic quaternion
    """
    r, p, y = rpy
    cr, sr = sympy.cos(r/2.), sympy.sin(r/2.)
    cp, sp = sympy.cos(p/2.), sympy.sin(p/2.)
    cy, sy = sympy.cos(y/2.), sympy.sin(y/2.)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    if convention == 'xyzw':
        return np.array([x, y, z, w])
    elif convention == 'wxyz':
        return np.array([w, x, y, z])
    else:
        raise NotImplementedError("Asking for a convention that has not been implemented")


def skew_matrix(vector):
    r"""
    Return the skew-symmetric matrix of the given vector, which allows to represents the cross product between the
    given vector and another vector, as the multiplication of the returned skew-symmetric matrix with the other
    vector.

    The skew-symmetric matrix from a 3D vector :math:`v=[x,y,z]` is given by:

    .. math::

        S(v) = \left[ \begin{array}{ccc} 0 & -z & y \\ z & 0 & -x \\ -y & x & 0 \\ \end{array} \right]

    It can be shown [2] that: :math:`\dot{R}(t) = \omega(t) \times R(t) = S(\omega(t)) R(t)`, where :math:`R(t)` is
    a rotation matrix that varies as time :math:`t` goes, :math:`\omega(t)` is the angular velocity vector of frame
    :math:`R(t) with respect to the reference frame at time :math:`t`, and :math:`S(.)` is the skew operation that
    returns the skew-symmetric matrix from the given vector.

    Args:
        vector (np.array[float[3]]): 3D vector

    Returns:
        np.array[float[3,3]]: skew-symmetric matrix

    References:
        - [1] Wikipedia: https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product
        - [2] "Robotics: Modelling, Planning and Control" (sec 3.1.1), by Siciliano et al., 2010
    """
    x, y, z = np.array(vector).flatten()
    return np.array([[0., -z, y],
                     [z, 0., -x],
                     [-y, x, 0.]])


# alias (for people who use http://www.petercorke.com/RTB/r9/html/skew.html)
skew = skew_matrix


def vector_from_skew_matrix(matrix):
    r"""
    Convert skew-symmetric matrix to vector; this is the inverse of the function `skew_matrix`.

    Warnings: this function does not check if the given matrix is skew-symmetric.

    Args:
        matrix (np.array[float[3,3]]): skew-symmetric matrix

    Returns:
        np.array[float[3]]: vector which produced the skew-symmetric matrix

    References:
        - [1] Wikipedia: https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product
        - [2] "Robotics: Modelling, Planning and Control" (sec 3.1.1), by Siciliano et al., 2010
    """
    return 0.5 * np.array([matrix[2, 1] - matrix[1, 2],
                           matrix[0, 2] - matrix[2, 0],
                           matrix[1, 0] - matrix[0, 1]])


# alias (for people who use http://www.petercorke.com/RTB/r9/html/vex.html)
vex = vector_from_skew_matrix


def rotation_matrix_x(angle):
    """
    Return the rotation matrix around the x-axis by the given angle.

    Args:
        angle (float): angle in radians

    Returns:
        np.array[float[3,3]]: rotation matrix around the x-axis
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.],
                     [0., c, -s],
                     [0., s, c]])


def rotation_matrix_y(angle):
    """
    Return the rotation matrix around the y-axis by the given angle.

    Args:
        angle (float): angle in radians

    Returns:
        np.array[float[3,3]]: rotation matrix around the y-axis
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s],
                     [0., 1., 0.],
                     [-s, 0, c]])


def rotation_matrix_z(angle):
    """
    Return the rotation matrix around the z-axis by the given angle.

    Args:
        angle (float): angle in radians

    Returns:
        np.array[float[3,3]]: rotation matrix around the z-axis
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.],
                     [s, c, 0.],
                     [0., 0., 1.]])


def get_spatial_transformation_matrix(rotation, position):
    r"""
    Get spatial transformation matrix that transforms

    .. math:: ^1X_2^T =

    Args:
        rotation (np.array[float[3,3]]): rotation matrix
        position (np.array[float[3]]): position of body

    Returns:
        np.array[float[6,6]]: spatial transformation matrix
    """
    pass


###############
# Quaternions #
###############

# reference: https://www.3dgep.com/understanding-quaternions/

quat_converter = QuaternionNumpyConverter(convention=1)


def get_rotated_point_from_quaternion(q, p, convention='xyzw'):
    """
    Return the rotated point due to the provided quaternion.

    .. math:: P_{\text{rotated}} = q * P * q^{-1} = q * P * q'

    where :math:`q` is the unit quaternion (to represents a proper rotation and thus the inverse :math:`q^{-1}` is
    equal to the conjugate :math:`q' = \bar{q}`), and :math:`P` is the quaternion where its vector part is equal to
    the 3D position of the point :math:`p` and its scalar part is 0.

    Args:
        q (np.array[float[4]], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)
        p (np.array[float[3]]): 3d point in space.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[3]]: rotated 3 point.
    """
    # TODO: fix this!
    p_quat = np.array([p[0], p[1], p[2], 0])
    q_inv = get_quaternion_inverse(q, convention=convention)
    p_rot = get_quaternion_product(q, get_quaternion_product(p_quat, q_inv, convention=convention),
                                   convention=convention)
    return p_rot[:3]


def get_quaternion_conjugate(q, convention='xyzw'):
    r"""Return the conjugate of the given quaternion; i.e. if the quaternion is given by q = [x,y,z,w] where [x,y,z]
    is the vector part and w is the scalar part, the conjugate is q'=[-x,-y,-z,w].

    If the quaternion is a unit quaternion, the conjugate is equal to the inverse of that quaternion.

    Args:
        q (np.array[float[4]], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[4]], quaternion.quaternion: quaternion inverse.
    """
    if isinstance(q, quaternion.quaternion):
        return q.inverse()
    elif isinstance(q, Iterable):
        if convention == 'xyzw':
            x, y, z, w = q
            return np.array([-x, -y, -z, w])
        elif convention == 'wxyz':
            w, x, y, z = q
            return np.array([w, -x, -y, -z])
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")
    else:
        raise TypeError


def get_quaternion_norm(q):
    r"""
    Return the norm of a quaternion: :math:`|q| = \sqrt(x^2 + y^2 + z^2 + w^2)`

    Args:
        q (np.array[float[4]], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)

    Returns:
        float: norm of a quaternion

    References:
        - [1] https://www.3dgep.com/understanding-quaternions/#Quaternions
    """
    return np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)


def normalize_quaternion(q):
    r"""
    Return the normalized quaternion; the quaternion divided by its norm.

    Args:
        q (np.array[float[4]], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)

    Returns:
        np.array[float[4]], quaternion.quaternion: normalized quaternion.
    """
    return q / get_quaternion_norm(q)


def get_quaternion_inverse(q, convention='xyzw'):
    """Return the inverse of the given quaternion.

    Note: the inverse of a quaternion is the conjugate of the quaternion divided by the square norm of that quaternion.
    Thus for a unit quaternion, the inverse of a quaternion is equal to its conjugate.

    This is given by:

    .. math:: q^{-1} = \frac{\bar{q}}{||q||}

    where :math:`\bar{q}` is the conjugate of the quaternion :math:`q`.

    Args:
        q (np.array[float[4]], quaternion.quaternion): quaternion.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[4]], quaternion.quaternion: quaternion inverse.
    """
    if isinstance(q, quaternion.quaternion):
        return q.inverse()
    elif isinstance(q, (np.ndarray, tuple, list)):
        if convention == 'xyzw':
            x, y, z, w = q
            return np.array([-x, -y, -z, w]) / np.linalg.norm(q)
        elif convention == 'wxyz':
            w, x, y, z = q
            return np.array([w, -x, -y, -z]) / np.linalg.norm(q)
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")
    else:
        raise TypeError


def get_quaternion_product(q1, q2, convention='xyzw'):
    """Return the quaternion product between two quaternions.

    The quaternion corresponding to the product :math:`R_1 R_2` where :math:`R_i` are rotation matrices is given by
    :math:`q_1 * q_2`.

    Args:
        q1 (np.array[float[4]], quaternion.quaternion): first quaternion
        q2 (np.array[float[4]], quaternion.quaternion): second quaternion
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[float[4]], quaternion.quaternion: resulting quaternion.
    """
    if type(q1) != type(q2):
        raise TypeError("Expecting q1 and q2 to be of the same type")
    if isinstance(q1, quaternion.quaternion):
        return q1*q2
    elif isinstance(q1, Iterable):

        def product(q1, q2, convention):
            if convention == 'xyzw':
                x1, y1, z1, w1 = q1
                x2, y2, z2, w2 = q2
                v1, v2 = np.array([x1, y1, z1]), np.array([x2, y2, z2])
                v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
                w = w1 * w2 - v1.dot(v2)
                return np.array([v[0], v[1], v[2], w])
            elif convention == 'wxyz':
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                v1, v2 = np.array([x1, y1, z1]), np.array([x2, y2, z2])
                v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
                w = w1 * w2 - v1.dot(v2)
                return np.array([w, v[0], v[1], v[2]])

        if isinstance(q1, np.ndarray):
            if len(q1.shape) == 1 and len(q2.shape) == 1:
                return product(q1, q2, convention)
            elif len(q1.shape) == 2 and len(q2.shape) == 1:
                return np.array([product(q, q2, convention) for q in q1])
            elif len(q1.shape) == 1 and len(q2.shape) == 2:
                return np.array([product(q1, q, convention) for q in q2])
            else:
                return np.array([product(q1_, q2_, convention) for q1_, q2_ in zip(q1, q2)])
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")
    else:
        raise TypeError


def quaternion_error(quat_des, quat_cur):
    r"""
    Compute the orientation (vector) error between the current and desired quaternion; that is, it is the difference
    between :math:`q_curr` and :math:`q_des`, which is given by: :math:`\Delta q = q_{curr}^{-1} q_{des}`.
    Only the vector part is returned which can be used in PD control.

    Args:
        quat_des (np.array[float[4]]): desired quaternion [x,y,z,w]
        quat_cur (np.array[float[4]]): current quaternion [x,y,z,w]

    Returns:
        np.array[float[3]]: vector error between the current and desired quaternion
    """
    diff = quat_cur[-1] * quat_des[:3] - quat_des[-1] * quat_cur[:3] - skew_matrix(quat_des[:3]).dot(quat_cur[:3])
    return diff


def logarithm_map(q):
    r"""
    Apply the logarithm map to a quaternion; :math:`log : S^3 \rightarrow R^3`, where :math:`\mathbb{S}^3` is a unit
    sphere in :math:`\mathbb{R}^4`.

    The mapping is given by:

    .. math::

        \log(q) = \log(s + \pmb{v}) = \left\{ \begin{array}{ll}
                \arccos(s) \frac{pmb{v}}{||\pmb{v}||}, & \pmb{v} \neq \pmb{0} \\
                [0, 0, 0]^\top, & \text{otherwise}
            \end{array} \right.

    where a quaternion :math:`q` is represented as :math:`s + \pmb{v}` with :math:`s \in \mathbb{R}` being the scalar
    part and :math:`\pmb{v} \in \mathbb{R}^3` the vector part.

    Args:
        q (np.array[float[4]]): quaternion

    Returns:
        np.array[float[3]]: resulting 3d vector
    """
    q = quat_converter.convert_to(q)
    s, v = q.w, np.array([q.x, q.y, q.z])

    zero = np.zeros(3)
    if np.allclose(v, zero):
        return zero
    return np.arccos(s) * v / np.linalg.norm(v)


def exponential_map(r):
    r"""
    Apply the exponential map to a 3d vector representing an orientation;
    :math:`exp : \mathbb{R}^3 \rightarrow \mathbb{S}^3`, where :math:`\mathbb{S}^3` is a unit sphere in
    :math:`\mathbb{R}^4`.

    The mapping is given by:

    .. math::

        \exp(\pmb{r}) = \left\{ \begin{array}{ll}
                \cos(||\pmb{r}||) + \sin(||\pmb{r}||) \frac{r}{||r||}, & \pmb{r} \neq \pmb{0} \\
                1, & \text{otherwise}
            \end{array} \right.


    where :math:`\pmb{r} \in \mathbb{R}^3`, and we use the following representation for the quaternion
    :math:`q = s + \pmb{v}` where :math:`s \in \mathbb{R}` is its scalar part, and :math:`\pmb{v} \in \mathbb{R}^3` is
    its vector part.

    Args:
        r (np.array[float[3]]): 3d vector

    Returns:
        np.array[float[4]]: quaternion
    """
    if np.allclose(r, np.zeros(3)):
        return quaternion.quaternion(1, 0, 0, 0)
    r_ = np.linalg.norm(r)
    x, y, z = np.sin(r_) * r / r_
    return quaternion.quaternion(np.cos(r_), x, y, z)


def angular_velocity_from_quaternion(q1, q2):
    r"""
    Compute the angular velocity that rotates quaternion :math:`q2` into :math:`q1` within unit time.
    Convert the difference between 2 quaternions using the logarithm map.

    .. math::

        \omega = 2 \log(q_1 * \bar{q}_2)

    where :math:`\omega` is the resulting angular velocity, :math:`\bar{q}_2` is the conjugate of :math:`q_2`, and
    :math:`\log: \mathbb{S}^3 \rightarrow \mathbb{R}^3` is the logarithm map that maps the unit quaternion (that is
    on a unit sphere :math:`\mathbb{S}^3` in :math:`\mathbb{R}^4` into the Euclidean space :math:`\mathbb{R}^3`.

    Args:
        q1: first (desired) quaternion
        q2: second (current) quaternion

    Returns:
        np.array[float[3]]: angular velocity (angular error in :math:`R^3`)
    """
    q1 = quat_converter.convert_to(q1)
    q2 = quat_converter.convert_to(q2)
    return 2 * logarithm_map(q1 * q2.conjugate())


def quaternion_distance(q1, q2):
    r"""
    Compute the distance metric (on :math:`\mathbb{S}^3`) between two quaternions :math:`q_1` and :math:`q_2`:

    Assuming a quaternion :math:`q` is represented as :math:`s + \pmb{v}` where :math:`s \in \mathbb{R}` is the scalar
    part and :math:`\pmb{v} \in \mathbb{R}^3` is the vector part, the distance is given by:

    .. math::

        d(q_1, q_2) = \left\{ \begin{array}{ll}
                2\pi, & q1 * \bar{q}_2 = -1 + [0,0,0]^\top \\
                2 || \log(q_1 * \bar{q}_2) ||, & \text{otherwise}
            \end{array} \right.

    where :math:`-1 + [0,0,0]^\top` is the only singularity on :math:`\mathbb{S}^3`.

    Note that this distance is not a metric on :math:`SO(3)` (the set of all orientations, which is by the way not a
    vector space but a group and a real 3d manifold).

    Args:
        q1: first quaternion
        q2: second quaternion

    Returns:
        float: distance between the 2 given quaternions

    References:
        - [1] "Orientation in Cartesian Space Dynamic Movement Primitives", Ude et al., 2014
        - [2] "Metrics for 3D Rotations: Comparison and Analysis", Huynh, 2009
    """
    q1 = quat_converter.convert_to(q1)
    q2 = quat_converter.convert_to(q2)

    q = q1 * q2.conjugate()
    s, v = q.w, np.array([q.x, q.y, q.z])

    if np.allclose(s, -1) and np.allclose(v, np.zeros(3)):
        return 2 * np.pi
    return 2 * np.linalg.norm(logarithm_map(q))


def trajectory_tracking_error(p_des, p_curr, q_des, q_curr, gamma=1.):
    r"""
    Compute the trajectory tracking error as provided in [1]. This tracking error is given by:

    .. math:: e(p_d, p_c, q_d, q_c) = ||p_d - p_c|| + \gamma d(q_d, q_c),

    where :math:`p_d` and :math:`p_c` are the desired and current 3d position, :math:`q_d` and :math:`q_c` are the
    desired and current orientations represented as quaternions, :math:`\gamma` is a weighting factor, and
    :math:`d(\cdot, \cdot)` is the distance metric on the unit sphere :math:`\mathbb{S}^3` (see `quaternion_distance`
    function for more info).

    Args:
        p_des (np.array[float[3]]): desired position
        p_curr (np.array[float[3]]): current position
        q_des: desired orientation (quaternion)
        q_curr: current orientation (quaternion)

    Returns:
        float: the trajectory tracking error

    References:
        - [1] "Orientation in Cartesian Space Dynamic Movement Primitives", Ude et al., 2014
    """
    return np.linalg.norm(p_des - p_curr) + gamma * quaternion_distance(q_des, q_curr)


def quaternion_derivative(rate, q):
    r"""
    Return the quaternion derivative

    .. math:: \dot{q}(t) = \frac{1}{2} \omega(t) * q(t)

    where :math:`\omega(t)` is the angular velocity a time :math:`t` which is treated here as a quaternion with a 0
    scalar value, :math:`q(t)` is the quaternion at time :math:`t`, :math:`\dot{q}(t)` is the derivative of the
    quaternion and :math:`*` is the quaternion product operator.

    Args:
        rate (np.array[float[3]]): angular velocity at time t.
        q: unit quaternion at time t.

    Returns:
        quaternion: a unit quaternion describing the rotation rate.

    References:
        - [1] "Orientation in Cartesian Space Dynamic Movement Primitives", Ude et al., 2014
    """
    w = quaternion.quaternion(0, rate[0], rate[1], rate[2])
    return 0.5 * get_quaternion_product(w, q)


def quaternion_integrate(rate, q, dt=0):
    r"""
    Integrate a quaternion

    .. math:: q(t + \Delta t) = \exp(\frac{\Delta t}{2} \omega(t)) * q(t)

    where :math:`\omega(t)` is the angular velocity a time :math:`t`, :math:`q(t)` is the quaternion at time :math:`t`,
    :math:`\Delta t` is the time difference to move forward in the future, and :math:`*` is the quaternion product
    operator.

    Args:
        rate (np.array[float[3]]): angular velocity at time t.
        q: unit quaternion at time t.
        dt (float): time difference to move forward in the future.

    Returns:
        quaternion: the resulting unit quaternion after :math:`t + \Delta t`.

    References:
        - [1] "Orientation in Cartesian Space Dynamic Movement Primitives", Ude et al., 2014
    """
    return get_quaternion_product(exponential_map(dt/2. * rate), q)


def slerp(q0, qf, t, t0=0., tf=1.):
    """
    Interpolate between two quaternions using Spherical Linear intERPolation (SLERP).

    Args:
        q0 (np.array[float[4]], quaternion.quaternion): initial quaternion.
        qf (np.array[float[4]], quaternion.quaternion): final quaternion.
        t (float, list[float], np.array[float]): the times to which the quaternions should be interpolated.
        t0 (float): initial time corresponding to the initial quaternion.
        tf (float): final time corresponding to the final quaternion.

    Returns:
        np.array, quaternion, np.array[quaternion]: one or multiple interpolated quaternions

    References:
        - [1] "Understanding Quaternions", https://www.3dgep.com/understanding-quaternions
        - [2] Documentation of numpy-quaternion
    """
    # convert if necessary
    is_input_quaternion = True
    if not isinstance(q0, quaternion.quaternion):
        q0 = quaternion.quaternion(q0[3], q0[0], q0[1], q0[2])
        is_input_quaternion = False
    if not isinstance(qf, quaternion.quaternion):
        qf = quaternion.quaternion(qf[3], qf[0], qf[1], qf[2])

    t = np.asarray(t)

    # interpolate using quaternion library
    qs = quaternion.slerp(q0, qf, t0, tf, t)

    # if we need to convert back to np.array
    if not is_input_quaternion:
        if isinstance(qs, np.ndarray):
            qs = quaternion.as_float_array(qs)
            qs = np.hstack((qs[:, 1:], qs[:, 0, np.newaxis]))
            return qs
        return np.array([qs.x, qs.y, qs.z, qs.w])
    return qs


def squad(quaternions, times, t):
    r"""
    Smoothly interpolate over a list/path of rotations using Spherical and QUADrangle (SQUAD).

    From [2]: "Spherical "quadrangular" interpolation of rotors with a cubic spline

    This is the best way to interpolate rotations.  It uses the analog of a cubic spline, except that the interpolant
    is confined to the rotor manifold in a natural way.  Alternative methods involving interpolation of other
    coordinates on the rotation group or normalization of interpolated values give bad results. The results from this
    method are as natural as any, and are continuous in first and second derivatives.

    The input `R_in` rotors are assumed to be reasonably continuous (no sign flips), and the input `t` arrays are
    assumed to be sorted. No checking is done for either case, and you may get silently bad results if these
    conditions are violated."

    Args:
        quaternions (list[np.array[float[4]]], list[quaternion.quaternion]): A time-series of rotors (unit
            quaternions) to be interpolated
        times (np.array[float], list[float]): the times corresponding to the quaternions.
        t (np.array[float], list[float]): the times to which the quaternions should be interpolated.

    Returns:
        np.array, np.array[quaternion]: interpolated quaternions

    References:
        - [1] "Understanding Quaternions", https://www.3dgep.com/understanding-quaternions
        - [2] Documentation of numpy-quaternion
    """
    # make sure that they are numpy arrays
    quaternions = np.asarray(quaternions)
    times = np.asarray(times)
    t = np.asarray(t)

    # check if we need to convert
    are_input_quaternions = (quaternions.dtype == quaternion.quaternion)
    if not are_input_quaternions:
        qs = quaternions
        quaternions = quaternion.from_float_array(np.hstack((qs[:, 0, np.newaxis], qs[:, 1:])))

    # interpolate
    qs = quaternion.squad(quaternions, times, t)

    # if we need to convert back to np.array
    if not are_input_quaternions:
        if isinstance(qs, np.ndarray):
            qs = quaternion.as_float_array(qs)
            qs = np.hstack((qs[:, 1:], qs[:, 0, np.newaxis]))
            return qs
        return np.array([qs.x, qs.y, qs.z, qs.w])
    return qs


# Tests
if __name__ == "__main__":
    import pybullet
    import tf.transformations as tft

    q = np.array([-0.043, 0.567, 0.368, 0.736])
    rpy = get_rpy_from_quaternion(q)

    print('\nRPY from quaternion: {}'.format(get_rpy_from_quaternion(q)))
    print('RPY <- matrix <- quaternion: {}'.format(get_rpy_from_matrix(get_matrix_from_quaternion(q))))
    print('Using pybullet: {}'.format(pybullet.getEulerFromQuaternion(q)))
    print('Using tf.transformations: {}'.format(tft.euler_from_quaternion(q)))

    print('\nQuaternion from RPY: {}'.format(get_quaternion_from_rpy(rpy)))
    print('Quaternion <- matrix <- RPY: {}'.format(get_quaternion_from_matrix(get_matrix_from_rpy(rpy))))
    print('Using pybullet: {}'.format(pybullet.getQuaternionFromEuler(rpy)))
    print('Using tf.transformations: {}'.format(tft.quaternion_from_euler(*rpy)))

    import quaternion
    q1 = quaternion.quaternion(q[3], q[0], q[1], q[2])
    q2 = q1
    print(q1 * q2)
    print(get_quaternion_product(q, q, convention='xyzw'))
