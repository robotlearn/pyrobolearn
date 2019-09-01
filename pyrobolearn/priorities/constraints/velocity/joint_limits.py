#!/usr/bin/env python
r"""Provide the joint position limits constraint.

This provides bounds/limits on the joint positions, which are given by:

.. math:: q_{lb} \leq q + \dot{q} dt \leq q_{ub}

where :math:`(q_{lb}, q_{ub})` are the lower and upper bound of the joint positions respectively, :math:`q` are
the current joint positions, :math:`\dot{q}` are the joint velocities being optimized, and :math:`dt` is the
integration time step.

This formulation can be rewritten as the inequality constraint math:`lb \leq x \leq ub` in QP, with
:math:`lb = (q_{lb} - q) / dt`, :math:`ub = (q_{ub} - q) / dt`, and :math:`x = \dot{q}`. This can
also be rewritten as :math:`Gx \leq h`, with :math:`G = [-dt*I, dt*I]^\top` and
:math:`h = [(q - q_{lb})^\top, (q_{ub} - q)^\top]^\top` where :math:`I` is the square identity matrix.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import BoundConstraint, JointVelocityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointPositionLimitsConstraint(BoundConstraint, JointVelocityConstraint):
    r"""Joint Position Limits constraint.

    This provides bounds/limits on the joint positions, which are given by:

    .. math:: q_{lb} \leq q + \dot{q} dt \leq q_{ub}

    where :math:`(q_{lb}, q_{ub})` are the lower and upper bound of the joint positions respectively, :math:`q` are
    the current joint positions, :math:`\dot{q}` are the joint velocities being optimized, and :math:`dt` is the
    integration time step.

    This formulation can be rewritten as the inequality constraint math:`lb \leq x \leq ub` in QP, with
    :math:`lb = (q_{lb} - q) / dt`, :math:`ub = (q_{ub} - q) / dt`, and :math:`x = \dot{q}`. This can
    also be rewritten as :math:`Gx \leq h`, with :math:`G = [-dt*I, dt*I]^\top` and
    :math:`h = [(q - q_{lb})^\top, (q_{ub} - q)^\top]^\top` where :math:`I` is the square identity matrix.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, dt, q_lower_bound=None, q_upper_bound=None):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            dt (float): integration time step: use to compute :math:`q = q + \dot{q} dt`.
            q_lower_bound (np.array[float[N]], None): joint position lower limits. If None, it will take the lower
              joint limits specified in the model. Note that if the lower limits are equal to the upper limits, they
              will be set to -10 and 10 by default.
            q_upper_bound (np.array[float[N]], None): joint position upper limits. If None, it will take the upper
              joint limits specified in the model. Note that if the upper limits are equal to the lower limits, they
              will be set to -10 and 10 by default.
        """
        super(JointPositionLimitsConstraint, self).__init__(model)

        # set time
        self.dt = dt

        # set variables
        if q_lower_bound is None or q_upper_bound is None:
            q_lb, q_ub = self.model.get_joint_limits()
            if q_lower_bound is None:
                q_lower_bound = q_lb
            if q_upper_bound is None:
                q_upper_bound = q_ub
            if np.allclose(q_lower_bound, q_upper_bound):
                print("WARNING: the joint position lower and upper limits are the same, by default they will be set "
                      "to -10 and 10.")
                q_lower_bound = -10. * np.ones(len(q_lower_bound))
                q_upper_bound = 10. * np.ones(len(q_upper_bound))

        self.q_lower_bounds = q_lower_bound
        self.q_upper_bounds = q_upper_bound

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
    def q_lower_bounds(self):
        """Get the lower joint position limits."""
        return self._q_lb

    @q_lower_bounds.setter
    def q_lower_bounds(self, q_lb):
        """Set the lower joint position limits."""
        if q_lb is None:
            q_lb = self.model.get_joint_limits()[0]
        if not isinstance(q_lb, np.ndarray):
            raise TypeError("Expecting the given lower joint position limits to be a np.array, instead got: "
                            "{}".format(q_lb))
        if len(q_lb) != self.x_size:
            raise ValueError("Expecting the length of the lower joint position limits (={}) to be the same length as "
                             "the number of variables being optimized =({}).".format(len(q_lb), self.x_size))
        self._q_lb = q_lb

    @property
    def q_upper_bounds(self):
        """Get the upper joint position limits."""
        return self._q_ub

    @q_upper_bounds.setter
    def q_upper_bounds(self, q_ub):
        """Set the upper joint position limits."""
        if q_ub is None:
            q_ub = self.model.get_joint_limits()[1]
        if not isinstance(q_ub, np.ndarray):
            raise TypeError("Expecting the given upper joint position limits to be a np.array, instead got: "
                            "{}".format(q_ub))
        if len(q_ub) != self.x_size:
            raise ValueError("Expecting the length of the upper joint position limits (={}) to be the same length as "
                             "the number of variables being optimized =({}).".format(len(q_ub), self.x_size))
        self._q_ub = q_ub

    ###########
    # Methods #
    ###########

    def _update(self):
        r"""
        Update the lower and upper bounds.
        """
        q = self.model.get_joint_positions()
        self._lower_bound = (self._q_lb - q) / self.dt
        self._upper_bound = (self._q_ub - q) / self.dt
