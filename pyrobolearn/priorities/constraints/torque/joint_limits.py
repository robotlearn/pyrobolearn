#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the joint limits constraint.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import BoundConstraint, JointTorqueConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointLimitsConstraint(BoundConstraint, JointTorqueConstraint):
    r"""Joint Limits constraint.

    This provides bounds/limits on the joint torques (based on a PD control feedback law):

    .. math::  k_p (q_{lb} - q) - k_d \dot{q} \leq \tau \leq k_p (q_{ub} - q) - k_d \dot{q}

    where :math:`q_{lb}, q_{ub}` are the lower and upper joint position limits, :math:`kp` and :math:`kd` are the
    position and velocity gains respectively, :math:`q, \dot{q}` are the current joint positions and velocities, and
    :math:`\tau` are the torques that are being optimized.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, q_lower_bound=None, q_upper_bound=None, kp=15000., kd=1000.):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            q_lower_bound (np.array[float[N]], None): joint position lower limits. If None, it will take the lower
              joint limits specified in the model. Note that if the lower limits are equal to the upper limits, they
              will be set to -10 and 10 by default.
            q_upper_bound (np.array[float[N]], None): joint position upper limits. If None, it will take the upper
              joint limits specified in the model. Note that if the upper limits are equal to the lower limits, they
              will be set to -10 and 10 by default.
            kp (float, np.array[float[N]]): position gain(s).
            kd (float, np.array[float[N]]): velocity gain(s).
        """
        super(JointLimitsConstraint, self).__init__(model)

        # set gains
        self.kp = kp
        self.kd = kd

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

    @property
    def kp(self):
        """Return the position gain."""
        return self._kp

    @kp.setter
    def kp(self, kp):
        """Set the position gain."""
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given position gain kp to be an int, float, np.array, instead got: "
                            "{}".format(type(kp)))
        if isinstance(kp, np.ndarray) and len(kp) != self.x_size:
            raise ValueError("Expecting the given position gain matrix kp to be of length {}, but instead got "
                             "a length of: {}".format(self.x_size, len(kp)))
        if np.any(kp < 0):
            raise ValueError("The position gain(s) should all be bigger or equal than 0, but found some negative "
                             "gains")
        self._kp = kp

    @property
    def kd(self):
        """Return the linear velocity gain."""
        return self._kd

    @kd.setter
    def kd(self, kd):
        """Set the linear velocity gain."""
        if not isinstance(kd, (float, int, np.ndarray)):
            raise TypeError("Expecting the given velocity gain kd to be an int, float, np.array, instead got: "
                            "{}".format(type(kd)))
        if isinstance(kd, np.ndarray) and len(kd) != self.x_size:
            raise ValueError("Expecting the given velocity gain matrix kd to be of length {}, but instead got "
                             "a length of: {}".format(self.x_size, len(kd)))
        if np.any(kd < 0):
            raise ValueError("The velocity gain(s) should all be bigger or equal than 0, but found some negative "
                             "gains")
        self._kd = kd

    ###########
    # Methods #
    ###########

    def _update(self):
        r"""
        Update the lower and upper bounds.
        """
        q = self.model.get_joint_positions()
        dq = self.model.get_joint_velocities()
        self._lower_bound = (self._q_lb - q) - self.kd * dq
        self._upper_bound = (self._q_ub - q) - self.kd * dq

