# utils code to transform orientation expressed in different forms
# This includes rotation matrices, euler angles (RPY), axis-angle, and quaternions

import numpy as np
import quaternion
import sympy
from collections import Iterable

from converter import QuaternionNumpyConverter


def getMatrixFromAxisAngle(axis, angle):
    x, y, z = axis
    a = angle
    c, s = np.cos(a), np.sin(a)
    c1 = 1 - c
    R = np.array([[x ** 2 * c1 + c, x * y * c1 - z * s, x * z * c1 + y * s],
                  [x * y * c1 + z * s, y ** 2 * c1 + c, y * z * c1 - x * s],
                  [x * z * c1 - y * s, y * z * c1 + x * s, z ** 2 * c1 + c]])
    return R


def getSymbolicMatrixFromAxisAngle(axis, angle):
    x, y, z = axis
    a = angle
    c, s = sympy.cos(a), sympy.sin(a)
    c1 = 1 - c
    R = np.array([[x**2 * c1 + c, x * y * c1 - z * s, x * z * c1 + y * s],
                  [x * y * c1 + z * s, y**2 * c1 + c, y * z * c1 - x * s],
                  [x * z * c1 - y * s, y * z * c1 + x * s, z**2 * c1 + c]])
    return R


def getAxisAngleFromMatrix(R):
    angle = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.)
    axis = 1. / (2. * np.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle, axis


def getSymbolicAxisAngleFromMatrix(R):
    angle = sympy.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.)
    axis = 1. / (2. * sympy.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle, axis


def getQuaternionFromAxisAngle(axis, angle, convert_to_quat=False, convention='xyzw'):
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


def getSymbolicQuaternionFromAxisAngle(axis, angle, convention='xyzw'):
    w = sympy.cos(angle / 2.)
    x, y, z = sympy.sin(angle / 2.) * axis
    if convention == 'xyzw':
        return np.array([x, y, z, w])
    elif convention == 'wxyz':
        return np.array([w, x, y, z])
    else:
        raise NotImplementedError("Asking for a convention that has not been implemented")


def getRPYFromMatrix(R):
    r = np.arctan2(R[1, 0], R[0, 0])
    p = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    y = np.arctan2(R[2, 1], R[2, 2])
    return np.array([r, p, y])


def getSymbolicRPYFromMatrix(R):
    r = sympy.atan2(R[1, 0], R[0, 0])
    p = sympy.atan2(-R[2, 0], sympy.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    y = sympy.atan2(R[2, 1], R[2, 2])
    return np.array([r, p, y])


def getMatrixFromRPY(rpy):
    cr, cp, cy = [np.cos(i) for i in rpy]
    sr, sp, sy = [np.sin(i) for i in rpy]
    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp, cp*sr, cp*cr]])
    return R


def getSymbolicMatrixFromRPY(rpy):
    cr, cp, cy = [sympy.cos(i) for i in rpy]
    sr, sp, sy = [sympy.sin(i) for i in rpy]
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    return R


def getQuaternionFromMatrix(R, convert_to_quat=False, convention='xyzw'):
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


def getSymbolicQuaternionFromMatrix(R, convention='xyzw'):
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


def getMatrixFromQuaternion(q, convention='xyzw'):
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


def getSymbolicMatrixFromQuaternion(q, convention='xyzw'):
    return getMatrixFromQuaternion(q, convention=convention)


def skew(vector):
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


def RotX(angle):
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


def RotY(angle):
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


def RotZ(angle):
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


def getQuaternionInverse(q, convention='xyzw'):
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


def getQuaternionProduct(q1, q2, convention='xyzw'):
    if type(q1) != type(q2):
        raise TypeError("Expecting q1 and q2 to be of the same type")
    if isinstance(q1, quaternion.quaternion):
        return q1*q2
    elif isinstance(q1, Iterable):
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
    q = quat_converter.convertTo(q)
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
    """
    Convert the difference between 2 quaternions using the logarithm map.

    Args:
        q1: first (desired) quaternion
        q2: second (current) quaternion

    Returns:
        float[3]: angular velocity (angular error in :math:`R^3`)
    """
    q1 = quat_converter.convertTo(q1)
    q2 = quat_converter.convertTo(q2)
    return 2 * logarithm_map(q1 * q2)
