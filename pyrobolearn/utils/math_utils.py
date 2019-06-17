#!/usr/bin/env python
"""Defines mathematical operations.
"""

import numpy as np
import copy

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def exp(x):
    if callable(x):
        y = copy.copy(x)

        def exp_():
            return np.exp(x())

        y.__call__ = exp_
        return y
    else:
        return np.exp(x)


class Plane(object):
    r"""Plane class.

    A plane is defined by its initial point and its normal vector.
    .. math:: \pi \equiv \overline{n} \cdot (\overline{x} - \overline{x}_0) = 0
    where :math:`\cdot` is the scalar product operator, :math:`\overline{n}` is the normal vector to the plane
    :math:`\pi`, :math:`\overline{x_0}` is the initial point on the plane, and :math:`\overline{x}` is an arbitrary
    point on the plane. Basically, this equation states that any vector on the plane is perpendicular to the normal
    vector.

    Given a 3D point in the space :math:`\overline{x}_1 = [x_1,y_1,z_1]`, if you wish to know the intersection of
    the line perpendicular to the plane :math:`\pi` and passing through this point, you can use the fact that this
    intersection point :math:`\overline{x} = [x,y,z]` has to satisfy the line and plane equations.
    That is, the line is given by :math:`\overline{x} &= \overline{x}_1 + \lambda \overline{n}`, and by replacing
    it in the plane equation, and solving it for :math:`\lambda`, and then finally re-incorporating this one into
    the line equation will give you:
    .. math:: `\overline{x} = \overline{x}_1 + \overline{n} \cdot (\overline{x}_0 - \overline{x}_1) \overline{n}`
    """

    def __init__(self, x0, normal):
        self.threshold = 1e-12
        self.x0 = x0
        self.normal = normal

    @staticmethod
    def convert_to_array(pt):
        if isinstance(pt, (tuple, list)):
            pt = np.array(pt)
        if not isinstance(pt, np.ndarray):
            raise TypeError("Expecting a numpy array of shape 3")
        else:
            if len(pt.shape) > 1:
                raise ValueError("Expecting an array")
            if pt.shape != (3,):
                raise ValueError("Expecting a shape 3")
        return pt

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0):
        self._x0 = self.convert_to_array(x0)

    @property
    def normal(self):
        return self._normal

    @normal.setter
    def normal(self, normal):
        normal = self.convert_to_array(normal)
        # normalize
        norm = np.linalg.norm(normal)
        if norm < self.threshold:
            raise ValueError("The norm of the normal vector is too close to zero.")
        self._normal = normal / norm

    def __contains__(self, point):
        """Check if the given point is in the plane."""
        point = self.convert_to_array(point)

        # scalar product between the normal and (point-x0) vectors
        val = self.normal.T.dot(point - self.x0)

        if val < self.threshold:
            return True
        return False

    def get_intersection_point(self, point):
        """
        Get the intersection of the plane with a line that starts at the given point and is parallel to the normal.
        """
        point = self.convert_to_array(point)
        return point + self.normal.T.dot(self.x0 - point) * self.normal