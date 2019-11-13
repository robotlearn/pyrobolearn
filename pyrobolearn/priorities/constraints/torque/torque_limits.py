#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the torque limits constraint.

This provides bounds/limits on the joint torques:

.. math::  \tau_{lb} \leq \tau \leq \tau_{ub}

where :math:`(\tau_{lb}, \tau_{ub})` are the lower and upper bound on the joint torques, and :math:`\tau` are the
joint torques being optimized.

This formulation can be rewritten as the inequality constraint math:`lb \leq x \leq ub` in QP, with
:math:`lb = \tau_{lb}`, :math:`ub = \tau_{ub}`, and :math:`x = \tau`. This can also be rewritten as :math:`Gx \leq h`,
with :math:`G = [-I, I]^\top` and :math:`h = [-\tau_{lb}^\top, \tau_{ub}^\top]^\top` where :math:`I` is the square
identity matrix.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import BoundConstraint, JointTorqueConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointTorqueLimitsConstraint(BoundConstraint, JointTorqueConstraint):
    r"""Torque Limits constraint.

    This provides bounds/limits on the joint torques:

    .. math::  \tau_{lb} \leq \tau \leq \tau_{ub}

    where :math:`(\tau_{lb}, \tau_{ub})` are the lower and upper bound on the joint torques, and
    :math:`\tau` are the joint torques being optimized.

    This formulation can be rewritten as the inequality constraint math:`lb \leq x \leq ub` in QP, with
    :math:`lb = \tau_{lb}`, :math:`ub = \tau_{ub}`, and :math:`x = \tau`. This can also be rewritten as
    :math:`Gx \leq h`, with :math:`G = [-I, I]^\top` and :math:`h = [-\tau_{lb}^\top, \tau_{ub}^\top]^\top`
    where :math:`I` is the square identity matrix.
    
    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, torque_lower_bound=None, torque_upper_bound=None):
        """
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            torque_lower_bound (np.array[float[N]], None): joint velocity lower limits. If None, it will take the lower
              joint limits specified in the model. Note that if the lower limits are equal to the upper limits, they
              will be set to -100 and 100 by default.
            torque_upper_bound (np.array[float[N]], None): joint velocity upper limits. If None, it will take the upper
              joint limits specified in the model. Note that if the upper limits are equal to the lower limits, they
              will be set to -100 and 100 by default.
        """
        super(JointTorqueLimitsConstraint, self).__init__(model)

        # set variables
        if torque_lower_bound is None or torque_upper_bound is None:
            tau_lb, tau_ub = self.model.get_joint_velocity_limits()
            if torque_lower_bound is None:
                torque_lower_bound = tau_lb
            if torque_upper_bound is None:
                torque_upper_bound = tau_ub
            if np.allclose(torque_lower_bound, torque_upper_bound):
                print("WARNING: the joint velocity lower and upper limits are the same, by default they will be set "
                      "to -10 and 10.")
                torque_lower_bound = -10. * np.ones(len(torque_lower_bound))
                torque_upper_bound = 10. * np.ones(len(torque_upper_bound))

        self.torque_lower_bounds = torque_lower_bound
        self.torque_upper_bounds = torque_upper_bound

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def torque_lower_bounds(self):
        """Get the lower joint velocity limits."""
        return self._tau_lb

    @torque_lower_bounds.setter
    def torque_lower_bounds(self, torque_lb):
        """Set the lower joint velocity limits."""
        if torque_lb is None:
            torque_lb = self.model.get_joint_limits()[0]
        if not isinstance(torque_lb, np.ndarray):
            raise TypeError("Expecting the given lower joint velocity limits to be a np.array, instead got: "
                            "{}".format(torque_lb))
        if len(torque_lb) != self.x_size:
            raise ValueError("Expecting the length of the lower joint velocity limits (={}) to be the same length as "
                             "the number of variables being optimized =({}).".format(len(torque_lb), self.x_size))
        self._tau_lb = torque_lb

    @property
    def torque_upper_bounds(self):
        """Get the upper joint velocity limits."""
        return self._tau_ub

    @torque_upper_bounds.setter
    def torque_upper_bounds(self, torque_ub):
        """Set the upper joint velocity limits."""
        if torque_ub is None:
            torque_ub = self.model.get_joint_limits()[1]
        if not isinstance(torque_ub, np.ndarray):
            raise TypeError("Expecting the given upper joint velocity limits to be a np.array, instead got: "
                            "{}".format(torque_ub))
        if len(torque_ub) != self.x_size:
            raise ValueError("Expecting the length of the upper joint velocity limits (={}) to be the same length as "
                             "the number of variables being optimized =({}).".format(len(torque_ub), self.x_size))
        self._tau_ub = torque_ub

    ###########
    # Methods #
    ###########

    def _update(self):
        r"""
        Update the lower and upper bounds.
        """
        self._lower_bound = self._tau_lb
        self._upper_bound = self._tau_ub
