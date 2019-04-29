#!/usr/bin/env python
"""Provide utils code to transform orientation expressed in different forms

This includes rotation matrices, euler angles (RPY), axis-angle, and quaternions.

References:
    [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010, chapter 2
"""

import numpy as np
import quaternion
import sympy
from collections import Iterable

from pyrobolearn.utils.converter import QuaternionNumpyConverter

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


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
        np.float[4], quaternion.quaternion: quaternion.
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
        np.float[4]: symbolic quaternion.
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
        np.float[4], quaternion.quaternion: quaternion
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
        q (np.float[4], quaternion.quaternion): quaternion
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
        q (np.array of 4 sympy.Symbol, np.float[4]): (symbolic) quaternion.
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
        q (np.float[4], quaternion.quaternion): quaternion
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
        q (np.float[4], np.array of 4 sympy.Symbol): quaternion
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
        np.float[4], quaternion.quaternion: quaternion
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
    x, y, z = vector
    return np.array([[0., -z, y],
                     [z, 0., -x],
                     [-y, x, 0.]])


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


###############
# Quaternions #
###############

quat_converter = QuaternionNumpyConverter(convention=1)


def get_quaternion_inverse(q, convention='xyzw'):
    """Return the inverse of the given quaternion.

    Args:
        q (np.float[4], quaternion.quaternion): quaternion.
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.float[4], quaternion.quaternion: quaternion inverse.
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
        q1 (np.float[4], quaternion.quaternion): first quaternion
        q2 (np.float[4], quaternion.quaternion): second quaternion
        convention (str): convention to be adopted when representing the quaternion. You can choose between 'xyzw' or
            'wxyz'.

    Returns:
        np.float[4], quaternion.quaternion: resulting quaternion.
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


def logarithm_map(q):
    r"""
    Apply the logarithm map to a quaternion; :math:`log : S^3 \rightarrow R^3`.

    Args:
        q (float[4]): quaternion

    Returns:
        float[3]: resulting 3d vector
    """
    q = quat_converter.convert_to(q)
    v, u = q.w,  np.array([q.x, q.y, q.z])

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
