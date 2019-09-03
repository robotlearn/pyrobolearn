#!/usr/bin/env python
r"""Provide the Convex Hull constraint.

From the documentation of the framework of [1]: "this constraint implements a constraint of the type:

.. math:: A_{CH} J_{CoM} \dot{q} \leq b_{CH}

where the number of row for :math:`A_{CH} \in \mathbb{R}^{F \times 3}` and :math:`b_{CH} \in \mathbb{F}` are the
number of facets :math:`F` in the convex hull."

This formulation can be rewritten as a upper unilateral inequality constraint :math:`A_{ineq} x \leq b_u` in QP,
with :math:`x = \dot{q}`, :math:`A_{ineq} = A_{CH} J_{CoM}`, and :math:`b_u = b_{CH}`.

Note that computing the ConvexHull at each time step can be quite expensive from a computing point of view, as
such you can specify the number of ticks to sleep before the next computation.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2), but we use the
`scipy.spatial.ConvexHull` class [2].

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] ConvexHull: https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.ConvexHull.html
"""

import numpy as np
from scipy.spatial import ConvexHull

from pyrobolearn.priorities.constraints.constraint import UpperUnilateralConstraint, JointVelocityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ConvexHullConstraint(UpperUnilateralConstraint, JointVelocityConstraint):
    r"""Convex Hull constraint.

    From the documentation of the framework of [1]: "this constraint implements a constraint of the type:

    .. math:: A_{CH} J_{CoM} \dot{q} \leq b_{CH}

    where the number of row for :math:`A_{CH} \in \mathbb{R}^{F \times 3}` and :math:`b_{CH} \in \mathbb{F}` are the
    number of facets :math:`F` in the convex hull."

    This formulation can be rewritten as a upper unilateral inequality constraint :math:`A_{ineq} x \leq b_u` in QP,
    with :math:`x = \dot{q}`, :math:`A_{ineq} = A_{CH} J_{CoM}`, and :math:`b_u = b_{CH}`.

    Note that computing the ConvexHull at each time step can be quite expensive from a computing point of view, as
    such you can specify the number of ticks to sleep before the next computation.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2), but we use the
    `scipy.spatial.ConvexHull` class [2].

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
        - [2] ConvexHull: https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.ConvexHull.html
    """

    def __init__(self, model, points=[], ticks=20):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            points (list[np.array[float[3]]]): list of 3D contact points. The convex hull will be built using these.
            ticks (ticks): the number of ticks to sleep before updating. Calculating the convex hull can be quite
              computing demanding.
        """
        super(ConvexHullConstraint, self).__init__(model)

        self.ticks = ticks
        self._cnt = 0
        self._hull = None

        self.points = points

    ##############
    # Properties #
    ##############

    @property
    def ticks(self):
        """Return the number of ticks to sleep before the next update."""
        return self._ticks

    @ticks.setter
    def ticks(self, ticks):
        """Set the number of ticks to sleep before the next update."""
        if not isinstance(ticks, int):
            raise TypeError("Expecting the given 'ticks' to be a int, but got instead: {}".format(type(ticks)))
        if ticks < 1:
            raise ValueError("Expecting the given 'ticks' to be bigger or equal to 1, but got: {}".format(ticks))
        self._ticks = ticks

    @property
    def points(self):
        """Return the list of 3D points."""
        return self._points

    @points.setter
    def points(self, points):
        """Set the list of 3D points."""
        if not isinstance(points, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'points' to be a list of 3D points.")
        if isinstance(points, np.ndarray):
            points = points.reshape(-1, 3)  # (M,3)
        self._points = points

    @property
    def hull(self):
        """Return the convex hull instance."""
        return self._hull

    @property
    def vertices(self):
        """Return the list of vertices that forms the convex hull."""
        if self._hull is not None:
            return self._hull.vertices

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the :math:`A_{ineq}` matrix and the :math:`b_u` vector"""
        # if time to update
        if self._cnt % self._ticks == 0:
            # convex hull
            hull = ConvexHull(self._points)  # compute convex hull
            self._hull = hull

            # convex hull equations
            A = hull.equations[:, :-1]
            b = 1 * hull.equations[:, -1]

            # get jacobian
            jacobian = self.model.get_com_jacobian(full=False)  # shape: (3,N)

            # constraint matrix and vector
            self._A_ineq = A.dot(jacobian)  # (F,N)
            self._b_upper_bound = b  # (F,)

            # reset counter
            self._cnt = 0

        # update counter
        self._cnt += 1
