#!/usr/bin/env python
r"""Provide the Convex Hull constraint.

This convex hull constraints make sure that the projected CoM position (in the x-y directions) belongs to the
convex hull (i.e. support polygon). This can be defined as:

.. math:: A_{CH} J_{CoM} \dot{q} dt \leq b_{CH} - A_{CH} x_{CoM} - d

where :math:`A_{CH} \in \mathbb{R}^{F \times 2}` and :math:`b_{CH} \in \mathbb{R}^F` are the matrix and vector
that appears in the convex hull hyperplane equation :math:`A_{CH} x \leq b_{CH}`, where :math:`F` is the number
of facets in the convex hull, :math:`J_{CoM} \in \mathbb{R}^{2 \times N}` is the truncated CoM Jacobian that only
accounts for the x and y direction components, :math:`\dot{q}` are the joint velocities being optimized,
:math:`x_{CoM} \in \mathbb{R}^2` is the current position of the CoM (due to the current joint configuration
:math:`q`) in the x-y directions, :math:`dt` is the integration time step, and :math:`d` is a safety margin
distance. Note that :math:`A_{CH}` has also be truncated to only take into account the x-y components (i.e.
:math:`A_{CH} \in \mathbb{R}^{F \times 2}` and not :math:`A_{CH} \in \mathbb{R}^{F \times 3}`).

This formulation can be rewritten as a upper unilateral inequality constraint :math:`A_{ineq} x \leq b_u` in QP,
with :math:`x = \dot{q}`, :math:`A_{ineq} = A_{CH} J_{CoM} dt`, and :math:`b_u = b_{CH} - A_{CH} x_{CoM} - d`.

Note that computing the ConvexHull at each time step can be quite expensive from a computing point of view, as
such you can specify the number of ticks to sleep before the next computation.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2), but we use the
`scipy.spatial.ConvexHull` class [2].

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] ConvexHull: https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.ConvexHull.html
"""

import numpy as np
import scipy.spatial.qhull as qhull

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

    This convex hull constraints make sure that the projected CoM position (in the x-y directions) belongs to the
    convex hull (i.e. support polygon). This can be defined as:

    .. math:: A_{CH} J_{CoM} \dot{q} dt \leq b_{CH} - A_{CH} x_{CoM} - d

    where :math:`A_{CH} \in \mathbb{R}^{F \times 2}` and :math:`b_{CH} \in \mathbb{R}^F` are the matrix and vector
    that appears in the convex hull hyperplane equation :math:`A_{CH} x \leq b_{CH}`, where :math:`F` is the number
    of facets in the convex hull, :math:`J_{CoM} \in \mathbb{R}^{2 \times N}` is the truncated CoM Jacobian that only
    accounts for the x and y direction components, :math:`\dot{q}` are the joint velocities being optimized,
    :math:`x_{CoM} \in \mathbb{R}^2` is the current position of the CoM (due to the current joint configuration
    :math:`q`) in the x-y directions, :math:`dt` is the integration time step, and :math:`d` is a safety margin
    distance. Note that :math:`A_{CH}` has also be truncated to only take into account the x-y components (i.e.
    :math:`A_{CH} \in \mathbb{R}^{F \times 2}` and not :math:`A_{CH} \in \mathbb{R}^{F \times 3}`).

    This formulation can be rewritten as a upper unilateral inequality constraint :math:`A_{ineq} x \leq b_u` in QP,
    with :math:`x = \dot{q}`, :math:`A_{ineq} = A_{CH} J_{CoM} dt`, and :math:`b_u = b_{CH} - A_{CH} x_{CoM} - d`.

    Note that computing the ConvexHull at each time step can be quite expensive from a computing point of view, as
    such you can specify the number of ticks to sleep before the next computation.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2), but we use the
    `scipy.spatial.ConvexHull` class [2].

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
        - [2] ConvexHull: https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.ConvexHull.html
    """

    def __init__(self, model, dt, points=[], ticks=20, safety_margin=0.):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            dt (float): integration time step used to compute :math:`dq = \dot{q} dt`.
            points (list[np.array[float[3]]]): list of 3D contact points. The convex hull will be built using these.
            ticks (ticks): the number of ticks to sleep before updating. Calculating the convex hull can be quite
              computing demanding.
            safety_margin (float): safety margin distance. This will be removed from the computed b vector.
        """
        super(ConvexHullConstraint, self).__init__(model)

        # set variables
        self.dt = dt
        self.ticks = ticks
        self._cnt = 0
        self._hull = None
        self.safety_margin = safety_margin
        self.points = points

        # first update
        self.update()
        self._cnt = 0

    ##############
    # Properties #
    ##############

    @property
    def dt(self):
        """Return the integration time step."""
        return self._dt

    @dt.setter
    def dt(self, dt):
        """Set the integration time step."""
        if not isinstance(dt, (float, int)):
            raise TypeError("Expecting the integration time step `dt` to be a float or int.")
        if dt <= 0:
            raise ValueError("Expecting the integration time step `dt` to be bigger than 0.")
        self._dt = float(dt)

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

        if len(points) > 2:  # we need at least 3 (different) points to construct the convex hull
            self.disable()
        else:
            self.enable()

    @property
    def safety_margin(self):
        """Return the safety margin distance."""
        return self._margin

    @safety_margin.setter
    def safety_margin(self, margin):
        """Set the safety margin distance."""
        self._margin = float(margin) if margin >= 0 else 0.

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
            # compute convex hull
            try:
                self._hull = qhull.ConvexHull(self._points)
            except qhull.QhullError:
                print("Not enough different points to compute the convex hull, using the old one.")

            if self._hull is not None:

                # convex hull equations
                A = self._hull.equations[:, :-2]  # shape: (F,2) - we only care about x and y (and not z))
                b = 1 * self._hull.equations[:, -1] - self.safety_margin

                # get jacobian (note that we only care about x and y (and not z))
                jacobian = self.model.get_com_jacobian(full=False)[:2]  # shape: (2,N)

                # get current com position
                x_com = self.model.get_com_position()[:2]  # shape: (2,)

                # constraint matrix and vector
                self._A_ineq = A.dot(jacobian)  # (F,N)
                self._b_upper_bound = b - self._margin - A.dot(x_com)  # (F,)

            # reset counter
            self._cnt = 0

        # update counter
        self._cnt += 1
