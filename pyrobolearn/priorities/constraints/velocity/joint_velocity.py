#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the differential kinematics constraint.

This provides the joint velocity constraints which is given by:

.. math:: J(q) \dot{q} = v

where :math:`J(q)` is the jacobian from a base link to a distal link, :math:`\dot{q}` are the joint velocities
being optimized, and :math:`v` is the imposed cartesian velocity imposed on the distal link.

This formulation can be rewritten as the inequality constraint :math:`A_{eq} x = b_{eq}` used in QP, with
:math:`A_{eq} = J(q)`, :math:`x = \dot{q}`, and :math:`b_{eq} = v`.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import EqualityConstraint, JointVelocityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DifferentialKinematicsConstraint(EqualityConstraint, JointVelocityConstraint):
    r"""Differential kinematics constraint.

    This provides the joint velocity constraints which is given by:

    .. math:: J(q) \dot{q} = v

    where :math:`J(q)` is the jacobian from a base link to a distal link, :math:`\dot{q}` are the joint velocities
    being optimized, and :math:`v` is the imposed cartesian velocity imposed on the distal link.

    This formulation can be rewritten as the inequality constraint :math:`A_{eq} x = b_{eq}` used in QP, with
    :math:`A_{eq} = J(q)`, :math:`x = \dot{q}`, and :math:`b_{eq} = v`.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, distal_link, base_link=None, local_position=(0., 0., 0.), linear_velocity=None,
                 angular_velocity=None):
        r"""
        Initialize the constraint.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the world.
            local_position (np.array[float[3]]): local position on the distal link.
            linear_velocity (np.array[float[3]], None): imposed linear velocity. If None, it will not be considered.
            angular_velocity (np.array[float[3]], None): imposed angular velocity. If None, it will not be considered.
        """
        super(DifferentialKinematicsConstraint, self).__init__(model)

        # define variables
        self.distal_link = self.model.get_link_id(distal_link)
        self.base_link = self.model.get_link_id(base_link) if base_link is not None else base_link
        self.local_position = local_position

        # set velocities
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

        # first update
        self.update()
        
    ##############
    # Properties #
    ##############

    @property
    def linear_velocity(self):
        """Get the cartesian linear velocity of the distal link wrt the base."""
        return self._lin_vel

    @linear_velocity.setter
    def linear_velocity(self, velocity):
        """Set the cartesian linear velocity of the distal link wrt the base."""
        if velocity is not None:
            if not isinstance(velocity, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given linear velocity to be a np.array, instead got: "
                                "{}".format(type(velocity)))
            velocity = np.asarray(velocity)
            if len(velocity) != 3:
                raise ValueError("Expecting the given linear velocity array to be of length 3, but instead "
                                 "got: {}".format(len(velocity)))
        self._lin_vel = velocity

    @property
    def angular_velocity(self):
        """Get the cartesian angular velocity of the distal link wrt the base."""
        return self._ang_vel

    @angular_velocity.setter
    def angular_velocity(self, velocity):
        """Set the cartesian angular velocity of the distal link wrt the base."""
        if velocity is not None:
            if not isinstance(velocity, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given angular velocity to be a np.array, instead got: "
                                "{}".format(type(velocity)))
            velocity = np.asarray(velocity)
            if len(velocity) != 3:
                raise ValueError("Expecting the given angular velocity array to be of length 3, but instead "
                                 "got: {}".format(len(velocity)))
        self._ang_vel = velocity
    
    ###########
    # Methods #
    ###########

    def _update(self):
        r"""
        Update the constraint by computing :math:`A_{eq}` and :math:`b_{eq}`.
        """
        if self._lin_vel is None and self._ang_vel is None:
            raise ValueError("Expecting at least the linear or angular velocity to be specified, but none are "
                             "provided.")
        self._A_eq = self.model.get_jacobian(link=self.distal_link, wrt_link=self.base_link, point=self.local_position)

        if self._lin_vel is None:
            self._A_eq = self._A_eq[3:]
            self._b_eq = self._ang_vel
        elif self._ang_vel is None:
            self._A_eq = self._A_eq[:3]
            self._b_eq = self._lin_vel
        else:
            self._b_eq = np.concatenate((self._lin_vel, self._ang_vel))
