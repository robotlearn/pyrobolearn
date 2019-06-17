#!/usr/bin/env python
"""Provide utils code to transform orientation expressed in different forms

This includes rotation matrices, euler angles (RPY), axis-angle, and quaternions.

References:
    [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010, chapter 2 and 3
    [2] "Understanding Quaternions", https://www.3dgep.com/understanding-quaternions
"""

import numpy as np
import quaternion
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


def get_homogeneous_transform(position, orientation):
    r"""
    Return the Homogeneous transform matrix given the position vector and the orientation.

    .. math::

        H = [[R, p],
             [zeros(3),1]]

    where :math:`R` is the 3x3 rotation matrix, :math:`p` is the 3x1 position vector.

    Args:
        position (np.array[3]): position vector
        orientation (np.array[4], np.array[3,3], np.array[3]): orientation (expressed as a quaternion [x,y,z,w],
            3x3 rotation matrix, or roll-pitch-yaw angles).

    Returns:
        np.array[4,4]: homogeneous matrix
    """
    if isinstance(orientation, quaternion.quaternion):
        R = quaternion.as_rotation_matrix(orientation)
    else:
        orientation = np.array(orientation)
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


def homogeneous_to_pose(matrix):
    r"""
    Return a pose (7D vector: position + quaternion) from a homogeneous matrix.

    Args:
        matrix (np.array[4,4]): homogeneous matrix

    Returns:
        np.array[7]: pose (position + quaternion [x,y,z,w])
    """
    position = matrix[:3, -1]
    quaternion = get_quaternion_from_matrix(matrix[:3, :3])
    return np.concatenate((position, quaternion))


def pose_to_homogeneous(pose):
    r"""
    Return a homogeneous matrix from a pose (7D vector: concatenation of position and quaternion).

    Args:
        pose (np.array[7]): concatenation of position and orientation (expressed as a quaternion [x,y,z,w])

    Returns:
        np.array[4,4]: homogeneous matrix
    """
    pose = np.array(pose).flatten()
    position, orientation = pose[:3], pose[-4:]
    return get_homogeneous_transform(position=position, orientation=orientation)


def get_matrix_from_axis_angle(axis, angle):
    """Return the rotation matrix from the specified axis and angle.

    Args:
        axis (np.float[3], list of 3 float): 3d axis vector.
        angle (float): angle.

    Returns:
        np.float[3,3]: rotation matrix.
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
        axis (np.float[3], list of 3 float, list of 3 sympy.Symbol): 3d axis vector.
        angle (float, sympy.Symbol): angle.

    Returns:
        np.float[3,3]: rotation matrix.
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
        R (np.float[3,3]): 3-by-3 rotation matrix.

    Returns:
        float: angle.
        np.float[3]: 3d axis vector.
    """
    angle = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.)
    axis = 1. / (2. * np.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle, axis


def get_symbolic_axis_angle_from_matrix(R):
    """Return the symbolic axis and angle from the specified rotation matrix.

    Args:
        R (np.array of sympy.Symbol): 3-by-3 rotation matrix.

    Returns:
        sympy.Symbol: angle.
        np.array of 3 sympy.Symbol: 3d axis vector.
    """
    angle = sympy.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.)
    axis = 1. / (2. * sympy.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle, axis


def get_quaternion_from_axis_angle(axis, angle, convert_to_quat=False, convention='xyzw'):
    """Get the quaternion associated from the axis/angle representation.

    Args:
        axis (np.float[3]): 3d axis vector.
        angle (float): angle.
        convert_to_quat (bool): If True, it will return an instance of `quaternion.quaternion`.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[4], quaternion.quaternion: quaternion.
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
        axis (np.float[3], np.array of 3 sympy.Symbol): 3d axis vector.
        angle (float, sympy.Symbol): angle.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[4]: symbolic quaternion.
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
        R (np.float[3,3]): 3-by-3 rotation matrix.

    Returns:
        np.float[3]: roll-pitch-yaw angle values.
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
        R (np.float[3,3], np.array of sympy.Symbol): symbolic 3-by-3 rotation matrix.

    Returns:
        np.array of 3 sympy.Symbol: symbolic roll-pitch-yaw angles.
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
        rpy (np.float[3]): roll-pitch-yaw angles

    Returns:
        np.float[3,3]: rotation matrix.
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
        rpy (np.float[3], np.array of 3 sympy.Symbol): roll-pitch-yaw angles.

    Returns:
        3-by-3 np.array of sympy.Symbol: symbolic rotation matrix
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
        R (np.float[3,3]): rotation matrix.
        convert_to_quat (bool): If True, it will return an instance of `quaternion.quaternion`.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[4], quaternion.quaternion: quaternion
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
        R (3-by-3 np.array of sympy.Symbol, np.float[3,3]): (symbolic) rotation matrix
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array of 4 sympy.Symbol: symbolic quaternion.
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
        q (np.array[4], quaternion.quaternion): quaternion
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.float[3,3]: rotation matrix.
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
        q (np.array of 4 sympy.Symbol, np.array[4]): (symbolic) quaternion.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        3-by-3 np.array of sympy.Symbol: symbolic rotation matrix.
    """
    return get_matrix_from_quaternion(q, convention=convention)


def get_rpy_from_quaternion(q, convention='xyzw'):
    """
    Get the Roll-Pitch-Yaw angle from the given quaternion.

    Args:
        q (np.array[4], quaternion.quaternion): quaternion
        convention: convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.float[3]: roll-pitch-yaw angles.
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
    roll = np.arctan2(2*(w*x + y*z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w*y - z*x))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2))
    return np.array([roll, pitch, yaw])


def get_symbolic_rpy_from_quaternion(q, convention='xyzw'):
    """
    Get the symbolic Roll-Pitch-Yaw angle from the given quaternion.

    Args:
        q (np.array[4], np.array of 4 sympy.Symbol): quaternion
        convention: convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array of 3 sympy.Symbol: symbolic roll-pitch-yaw angles.
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
        rpy (np.float[3]): roll-pitch-yaw angles
        convert_to_quat (bool): If True, it will return an instance of `quaternion.quaternion`.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[4], quaternion.quaternion: quaternion
    """
    r, p, y = rpy
    cr, sr = np.cos(r/2.), np.sin(r/2.)
    cp, sp = np.cos(p/2.), np.sin(p/2.)
    cy, sy = np.cos(y/2.), np.sin(y/2.)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    if convert_to_quat:
        return quaternion.quaternion(w, x, y, z)
    else:
        if convention == 'xyzw':
            return np.array([x, y, z, w])
        elif convention == 'wxyz':
            return np.array([w, x, y, z])
        else:
            raise NotImplementedError("Asking for a convention that has not been implemented")


def get_symbolic_quaternion_from_rpy(rpy, convention='xyzw'):
    """
    Get symbolic quaternion from Roll-Pitch-Yaw angle.

    Args:
        rpy (np.float[3], np.array of 3 sympy.Symbol): (symbolic) roll-pitch-yaw angles
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array of 4 sympy.Symbol: symbolic quaternion
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
        vector (np.array[3]): 3D vector

    Returns:
        np.array[3,3]: skew-symmetric matrix

    References:
        [1] Wikipedia: https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product
        [2] "Robotics: Modelling, Planning and Control" (sec 3.1.1), by Siciliano et al., 2010
    """
    x, y, z = np.array(vector).flatten()
    return np.array([[0., -z, y],
                     [z, 0., -x],
                     [-y, x, 0.]])


def vector_from_skew_matrix(matrix):
    r"""
    Convert skew-symmetric matrix to vector; this is the inverse of the function `skew_matrix`.

    Warnings: this function does not check if the given matrix is skew-symmetric.

    Args:
        matrix (np.array[3,3]): skew-symmetric matrix

    Returns:
        np.array[3]: vector which produced the skew-symmetric matrix

    References:
        [1] Wikipedia: https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product
        [2] "Robotics: Modelling, Planning and Control" (sec 3.1.1), by Siciliano et al., 2010
    """
    return 0.5 * np.array([matrix[2, 1] - matrix[1, 2],
                           matrix[0, 2] - matrix[2, 0],
                           matrix[1, 0] - matrix[0, 1]])


def rotation_matrix_x(angle):
    """
    Return the rotation matrix around the x-axis by the given angle.

    Args:
        angle (float): angle in radians

    Returns:
        np.array[3,3]: rotation matrix around the x-axis
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
        np.array[3,3]: rotation matrix around the y-axis
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
        np.array[3,3]: rotation matrix around the z-axis
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
        rotation (np.array[3,3]): rotation matrix
        position (np.array[3]): position of body

    Returns:
        np.array[6,6]: spatial transformation matrix
    """
    pass


###############
# Quaternions #
###############

# reference: https://www.3dgep.com/understanding-quaternions/

quat_converter = QuaternionNumpyConverter(convention=1)


def get_quaternion_conjugate(q, convention='xyzw'):
    r"""Return the conjugate of the given quaternion; i.e. if the quaternion is given by q = [x,y,z,w] where [x,y,z]
    is the vector part and

    Args:
        q (np.array[4], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[4], quaternion.quaternion: quaternion inverse.
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
        q (np.array[4], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)

    Returns:
        float: norm of a quaternion

    References:
        [1] https://www.3dgep.com/understanding-quaternions/#Quaternions
    """
    return np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)


def normalize_quaternion(q):
    r"""
    Return the normalized quaternion; the quaternion divided by its norm.

    Args:
        q (np.array[4], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)

    Returns:
        np.array[4], quaternion.quaternion: normalized quaternion.
    """
    return q / get_quaternion_norm(q)


def get_quaternion_inverse(q, convention='xyzw'):
    """Return the inverse of the given quaternion.

    Note: the inverse of a quaternion is the conjugate of the quaternion divided by the square norm of that quaternion.

    Args:
        q (np.array[4], quaternion.quaternion): quaternion.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[4], quaternion.quaternion: quaternion inverse.
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


def get_quaternion_product(q1, q2, convention='xyzw'):
    """Return the quaternion product between two quaternions.

    Args:
        q1 (np.array[4], quaternion.quaternion): first quaternion
        q2 (np.array[4], quaternion.quaternion): second quaternion
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.array[4], quaternion.quaternion: resulting quaternion.
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
    """
    Quaternion error between two quaternions.

    Args:
        quat_des (np.array[4], quaternion.quaternion): desired quaternion
        quat_cur (np.array[4], quaternion.quaternion): current quaternion

    Returns:

    """
    diff = quat_cur[-1] * quat_des[:3] - quat_des[-1] * quat_cur[:3] - skew_matrix(quat_des[:3]).dot(quat_cur[:3])
    return diff


def logarithm_map(q):
    r"""
    Apply the logarithm map to a quaternion; :math:`log : S^3 \rightarrow R^3`.

    Args:
        q (float[4]): quaternion

    Returns:
        float[3]: resulting 3d vector
    """
    q = quat_converter.convert_to(q)
    v, u = q.w, np.array([q.x, q.y, q.z])

    zero = np.zeros(3)
    if np.allclose(u, zero):
        return zero
    return np.arccos(v) * u / np.linalg.norm(u)


def exponential_map(r):
    r"""
    Apply the exponential map to a 3d vector representing an orientation; :math:`exp : R^3 \rightarrow S^3`

    Args:
        r (float[3]): 3d vector

    Returns:
        float[4]: quaternion
    """
    if np.allclose(r, np.zeros(3)):
        return quaternion.quaternion(1, 0, 0, 0)
    r_ = np.linalg.norm(r)
    x, y, z = np.sin(r_) * r / r_
    return quaternion.quaternion(np.cos(r_), x, y, z)


def angular_velocity_from_quaternion(q1, q2):
    r"""
    Compute the angular velocity that rotates quaternion q2 into q1 within unit time.
    Convert the difference between 2 quaternions using the logarithm map.

    Args:
        q1: first (desired) quaternion
        q2: second (current) quaternion

    Returns:
        float[3]: angular velocity (angular error in :math:`R^3`)
    """
    q1 = quat_converter.convert_to(q1)
    q2 = quat_converter.convert_to(q2)
    return 2 * logarithm_map(q1 * q2)


def slerp(q0, qf):
    """
    Interpolate between two quaternions using Spherical Linear intERPolation (SLERP).

    Args:
        q1:
        q2:

    Returns:

    """
    pass


def squad(quaternions):
    r"""
    Smoothly interpolate over a list/path of rotations using Spherical and QUADrangle (SQUAD).

    Args:
        quaternions (list of np.array, list of quaternion.quaternion):

    Returns:

    """
    pass


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
