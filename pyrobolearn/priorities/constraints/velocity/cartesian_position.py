#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the Cartesian Position constraint.

The bilateral inequality cartesian position constraint is given by:

.. math:: x_{lb} \leq x + J(q) \dot{q} * dt \leq x_{ub}

where :math:`x_{lb}, x_{ub}` are the lower and upper bound on the cartesian positions of a given distal link,
:math:`x` is the current cartesian position of the distal link wrt the base link, :math:`\dot{q}` are the joint
velocities being optimized, :math:`J(q)` is the Jacobian from the base to the distal link, and :math:`dt` is the
integration time step.

This formulation can be rewritten as a bilateral inequality constraint :math:`b_l \leq A_{ineq} x \leq b_u` in QP,
with :math:`x = \dot{q}`, :math:`A_{ineq} = J(q) * dt`, :math:`b_l = (x_{lb} - x)` and :math:`b_u = (x_{ub} - x)`.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import BilateralConstraint, JointVelocityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CartesianPositionConstraint(BilateralConstraint, JointVelocityConstraint):
    r"""Cartesian Position constraint.

    The bilateral inequality cartesian position constraint is given by:

    .. math:: x_{lb} \leq x + J(q) \dot{q} * dt \leq x_{ub}

    where :math:`x_{lb}, x_{ub}` are the lower and upper bound on the cartesian positions of a given distal link,
    :math:`x` is the current cartesian position of the distal link wrt the base link, :math:`\dot{q}` are the joint
    velocities being optimized, :math:`J(q)` is the Jacobian from the base to the distal link, and :math:`dt` is the
    integration time step.

    This formulation can be rewritten as a bilateral inequality constraint :math:`b_l \leq A_{ineq} x \leq b_u` in QP,
    with :math:`x = \dot{q}`, :math:`A_{ineq} = J(q) * dt`, :math:`b_l = (x_{lb} - x)` and :math:`b_u = (x_{ub} - x)`.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, dt, distal_link, base_link=None, local_position=(0, 0, 0), x_lower_bound=None,
                 x_upper_bound=None):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            dt (float): integration time step: use to compute :math:`x = x + v dt`.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the world.
            local_position (np.array[float[3]]): local position on the distal link.
            x_lower_bound (np.array[float[3]], None): the lower bound on the cartesian position of the distal link wrt
              the base link. If None, it will not be considered.
            x_upper_bound (np.array[float[3]], None): the upper bound on the cartesian position of the distal link wrt
              the base link. If None, it will not be considered.
        """
        super(CartesianPositionConstraint, self).__init__(model)

        # set time
        self.dt = dt

        # define variables
        self.distal_link = self.model.get_link_id(distal_link)
        self.base_link = self.model.get_link_id(base_link) if base_link is not None else base_link
        self.local_position = local_position

        self.position_lower_bound = x_lower_bound
        self.position_upper_bound = x_upper_bound

        # first update
        self.update()

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
    def position_lower_bound(self):
        """Get the lower bound on the cartesian position of the distal link wrt the base link."""
        return self._x_lower_bound

    @position_lower_bound.setter
    def position_lower_bound(self, bound):
        """Set the lower bound on the cartesian position of the distal link wrt the base link."""
        if bound is not None:
            if not isinstance(bound, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the given lower cartesian position bound to be a np.array, instead got: "
                                "{}".format(bound))
            bound = np.asarray(bound)
            if len(bound) != 3:
                raise ValueError("Expecting the length of the lower bound to be 3, but got instead a length of: "
                                 "{}.".format(len(bound)))
        self._x_lower_bound = bound

    @property
    def position_upper_bound(self):
        """Get the upper bound on the cartesian position of the distal link wrt the base link."""
        return self._x_upper_bound

    @position_upper_bound.setter
    def position_upper_bound(self, bound):
        """Set the upper bound on the cartesian position of the distal link wrt the base link."""
        if bound is not None:
            if not isinstance(bound, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the given upper cartesian position bound to be a np.array, instead got: "
                                "{}".format(bound))
            bound = np.asarray(bound)
            if len(bound) != 3:
                raise ValueError("Expecting the length of the upper bound to be 3, but got instead a length of: "
                                 "{}.".format(len(bound)))
        self._x_upper_bound = bound

    ###########
    # Methods #
    ###########

    def _update(self):
        """Update the inequality matrix and vectors."""
        if self._x_lower_bound is None and self._x_upper_bound is None:
            raise ValueError("Expecting at least the lower or upper bounds of the cartesian position to be "
                             "specified, but instead got None for both of them.")

        self._A_ineq = self.model.get_jacobian(link=self.distal_link, wrt_link=self.base_link,
                                               point=self.local_position)[:3] * self.dt
        x = self.model.get_position(link=self.distal_link, wrt_link=self.base_link)

        if self._x_lower_bound is not None:
            self._b_lower_bound = self._x_lower_bound - x
        if self._x_upper_bound is not None:
            self._b_upper_bound = self._x_upper_bound - x
