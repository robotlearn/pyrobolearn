#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the center of mass velocity task.

The CoM task tries to impose a desired position and (linear) velocity of the CoM with respect to the world frame.

.. math:: ||J_{CoM} \dot{q} - (K_p (x_d - x) + \dot{x}_d)||^2

where :math:`J_{CoM}` is the CoM Jacobian, :math:`\dot{q}` are the joint velocities being optimized, :math:`K_p`
is the stiffness gain, :math:`x_d` and :math:`x` are the desired and current cartesian CoM position
respectively, and :math:`\dot{x}_d` is the desired linear velocity of the CoM.

This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=J_{CoM}`,
:math:`x=\dot{q}`, and :math:`b = K_p (x_d - x) + \dot{x}_d`.

Note that you can only specify the center of mass linear velocity if you wish.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CoMTask(JointVelocityTask):
    r"""Center of Mass Velocity Task

    The CoM task tries to impose a desired position and (linear) velocity of the CoM with respect to the world frame.

    .. math:: ||J_{CoM} \dot{q} - (K_p (x_d - x) + \dot{x}_d)||^2

    where :math:`J_{CoM}` is the CoM Jacobian, :math:`\dot{q}` are the joint velocities being optimized, :math:`K_p`
    is the stiffness gain, :math:`x_d` and :math:`x` are the desired and current cartesian CoM position
    respectively, and :math:`\dot{x}_d` is the desired linear velocity of the CoM.

    This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=J_{CoM}`,
    :math:`x=\dot{q}`, and :math:`b = K_p (x_d - x) + \dot{x}_d`.

    Note that you can only specify the center of mass linear velocity if you wish.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, desired_position=None, desired_velocity=None, kp=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            desired_position (np.array[float[3]], None): desired CoM position. If None, it will be set to 0.
            desired_velocity (np.array[float[3]], None): desired CoM linear velocity. If None, it will be set to 0.
            kp (float, np.array[float[3,3]]): stiffness gain.
            weight (float, np.array[float[3,3]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(CoMTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define variable
        self.kp = kp

        # define desired references
        self.desired_position = desired_position
        self.desired_velocity = desired_velocity

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def desired_position(self):
        """Get the desired CoM position."""
        return self._des_pos

    @desired_position.setter
    def desired_position(self, position):
        """Set the desired CoM position."""
        if position is not None:
            if not isinstance(position, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given desired position to be a np.array, instead got: "
                                "{}".format(type(position)))
            position = np.asarray(position)
            if len(position) != 3:
                raise ValueError("Expecting the given desired position array to be of length 3, but instead got: "
                                 "{}".format(len(position)))
        self._des_pos = position

    @property
    def desired_velocity(self):
        """Get the desired CoM linear velocity."""
        return self._des_vel

    @desired_velocity.setter
    def desired_velocity(self, velocity):
        """Set the desired CoM linear velocity."""
        if velocity is None:
            velocity = np.zeros(3)
        elif not isinstance(velocity, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired linear velocity to be a np.array, instead got: "
                            "{}".format(type(velocity)))
        velocity = np.asarray(velocity)
        if len(velocity) != 3:
            raise ValueError("Expecting the given desired linear velocity array to be of length 3, but instead "
                             "got: {}".format(len(velocity)))
        self._des_vel = velocity

    @property
    def x_desired(self):
        """Get the desired CoM position."""
        return self._des_pos

    @x_desired.setter
    def x_desired(self, x_d):
        """Set the desired CoM position."""
        self.desired_position = x_d

    @property
    def dx_desired(self):
        """Get the desired CoM linear velocity."""
        return self._des_vel

    @dx_desired.setter
    def dx_desired(self, dx_d):
        """Set the desired CoM linear velocity."""
        self.desired_velocity = dx_d

    @property
    def kp(self):
        """Return the stiffness gain."""
        return self._kp

    @kp.setter
    def kp(self, kp):
        """Set the stiffness gain."""
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given stiffness gain kp to be an int, float, np.array, instead got: "
                            "{}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (3, 3):
            raise ValueError("Expecting the given stiffness gain matrix kp to be of shape {}, but instead got "
                             "shape: {}".format((self.x_size, self.x_size), kp.shape))
        self._kp = kp

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, dx_des=None, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[float[3]], None): desired CoM position. If None, it will be set to 0.
            dx_des (np.array[float[3]], None): desired CoM linear velocity. If None, it will be set to 0.
        """
        self.x_desired = x_des
        self.dx_desired = dx_des

    def get_desired_references(self):
        """Return the desired references.

        Returns:
            np.array[float[3]]: desired CoM position.
            np.array[float[3]]: desired CoM linear velocity.
        """
        return self.x_desired, self.dx_desired

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """

        self._A = self.model.get_com_jacobian(full=False)  # shape: (3, N)
        if self._des_pos is not None:
            x = self.model.get_com_position()
            self._b = np.dot(self.kp, (self._des_pos - x)) + self._des_vel  # shape: (3,)
        else:
            self._b = self._des_vel  # shape: (3,)
