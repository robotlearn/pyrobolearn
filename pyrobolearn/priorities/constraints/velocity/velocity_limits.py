# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the velocity limits constraint.

This provides bounds/limits on the joint velocities

.. math::  \dot{q}_{lb} \leq \dot{q} \leq \dot{q}_{ub}

where :math:`(\dot{q}_{lb}, \dot{q}_{ub})` are the lower and upper bound on the joint velocities, and
:math:`\dot{q}` are the joint velocities being optimized.

This formulation can be rewritten as the inequality constraint math:`lb \leq x \leq ub` in QP, with
:math:`lb = \dot{q}_{lb}`, :math:`ub = \dot{q}_{ub}`, and :math:`x = \dot{q}`. This can also be rewritten as
:math:`Gx \leq h`, with :math:`G = [-I, I]^\top` and :math:`h = [-\dot{q}_{lb}^\top, \dot{q}_{ub}^\top]^\top`
where :math:`I` is the square identity matrix.

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


class JointVelocityLimitsConstraint(BoundConstraint, JointVelocityConstraint):
    r"""Joint velocity limits constraint.

    This provides bounds/limits on the joint velocities:

    .. math::  \dot{q}_{lb} \leq \dot{q} \leq \dot{q}_{ub}

    where :math:`(\dot{q}_{lb}, \dot{q}_{ub})` are the lower and upper bound on the joint velocities, and
    :math:`\dot{q}` are the joint velocities being optimized.

    This formulation can be rewritten as the inequality constraint math:`lb \leq x \leq ub` in QP, with
    :math:`lb = \dot{q}_{lb}`, :math:`ub = \dot{q}_{ub}`, and :math:`x = \dot{q}`. This can also be rewritten as
    :math:`Gx \leq h`, with :math:`G = [-I, I]^\top` and :math:`h = [-\dot{q}_{lb}^\top, \dot{q}_{ub}^\top]^\top`
    where :math:`I` is the square identity matrix.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, dq_lower_bound=None, dq_upper_bound=None):
        """
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            dq_lower_bound (np.array[float[N]], None): joint velocity lower limits. If None, it will take the lower
              joint limits specified in the model. Note that if the lower limits are equal to the upper limits, they
              will be set to -10 and 10 by default.
            dq_upper_bound (np.array[float[N]], None): joint velocity upper limits. If None, it will take the upper
              joint limits specified in the model. Note that if the upper limits are equal to the lower limits, they
              will be set to -10 and 10 by default.
        """
        super(JointVelocityLimitsConstraint, self).__init__(model)

        # set variables
        if dq_lower_bound is None or dq_upper_bound is None:
            dq_lb, dq_ub = self.model.get_joint_velocity_limits()
            if dq_lower_bound is None:
                dq_lower_bound = dq_lb
            if dq_upper_bound is None:
                dq_upper_bound = dq_ub
            if np.allclose(dq_lower_bound, dq_upper_bound):
                print("WARNING: the joint velocity lower and upper limits are the same, by default they will be set "
                      "to -10 and 10.")
                dq_lower_bound = -10. * np.ones(len(dq_lower_bound))
                dq_upper_bound = 10. * np.ones(len(dq_upper_bound))

        self.dq_lower_bounds = dq_lower_bound
        self.dq_upper_bounds = dq_upper_bound

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def dq_lower_bounds(self):
        """Get the lower joint velocity limits."""
        return self._dq_lb

    @dq_lower_bounds.setter
    def dq_lower_bounds(self, dq_lb):
        """Set the lower joint velocity limits."""
        if dq_lb is None:
            dq_lb = self.model.get_joint_limits()[0]
        if not isinstance(dq_lb, np.ndarray):
            raise TypeError("Expecting the given lower joint velocity limits to be a np.array, instead got: "
                            "{}".format(dq_lb))
        if len(dq_lb) != self.x_size:
            raise ValueError("Expecting the length of the lower joint velocity limits (={}) to be the same length as "
                             "the number of variables being optimized =({}).".format(len(dq_lb), self.x_size))
        self._dq_lb = dq_lb

    @property
    def dq_upper_bounds(self):
        """Get the upper joint velocity limits."""
        return self._dq_ub

    @dq_upper_bounds.setter
    def dq_upper_bounds(self, dq_ub):
        """Set the upper joint velocity limits."""
        if dq_ub is None:
            dq_ub = self.model.get_joint_limits()[1]
        if not isinstance(dq_ub, np.ndarray):
            raise TypeError("Expecting the given upper joint velocity limits to be a np.array, instead got: "
                            "{}".format(dq_ub))
        if len(dq_ub) != self.x_size:
            raise ValueError("Expecting the length of the upper joint velocity limits (={}) to be the same length as "
                             "the number of variables being optimized =({}).".format(len(dq_ub), self.x_size))
        self._dq_ub = dq_ub

    ###########
    # Methods #
    ###########

    def _update(self):
        r"""
        Update the lower and upper bounds.
        """
        self._lower_bound = self._dq_lb
        self._upper_bound = self._dq_ub
