#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the cartesian (velocity) task.

The cartesian task tries to impose a desired pose (position and orientation) and velocity of a distal link with
respect to a base link or the world frame. The minimization problem is given by:

.. math:: || ^bJ_d(q) \dot{q} - (K_p e + \dot{x}_d) ||^2

where :math:`^bJ_d(q) \in \mathbb{R}^{6 \times N}` is the Jacobian taken from the base to the distal link,
:math:`\dot{q}` are the joint velocities being optimized, :math:`K_p` is the stiffness gain,
:math:`e \in \mathbb{R}^{6}` is the error which is the concatenation of the position error given by
:math:`e_{p} = (x_d - x)` (with :math:`x_d` being the desired position, and :math:`x` the current position), and the
orientation error given by (if expressed as quaternions :math:`o = {s, v}` where :math:`s` is the real scalar part,
and :math:`v` is the vector part) :math:`e_{o} = s v_d - s_d v - v_d \cross v`, and :math:`\dot{x}_d` is the
desired cartesian velocity for the distal link with respect to the base link.

This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A = ^bJ_d(q)`,
:math:`x = \dot{q}`, and :math:`b = K_p e + \dot{x}_d`.

Important notes:

- You don't have to specify the whole pose, you can also only specify the position or orientation.
- You can also only specify the cartesian velocities without providing the cartesian position or orientation.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask
from pyrobolearn.utils.transformation import quaternion_error


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CartesianTask(JointVelocityTask):
    r"""Cartesian (velocity) Task

    The cartesian task tries to impose a desired pose (position and orientation) and velocity of a distal link with
    respect to a base link or the world frame. The minimization problem is given by:

    .. math:: || ^bJ_d(q) \dot{q} - (K_p e + \dot{x}_d) ||^2

    where :math:`^bJ_d(q) \in \mathbb{R}^{6 \times N}` is the Jacobian taken from the base to the distal link,
    :math:`\dot{q}` are the joint velocities being optimized, :math:`K_p` is the stiffness gain,
    :math:`e \in \mathbb{R}^{6}` is the error which is the concatenation of the position error given by
    :math:`e_{p} = (x_d - x)` (with :math:`x_d` being the desired position, and :math:`x` the current position), and
    the orientation error given by (if expressed as quaternions :math:`o = {s, v}` where :math:`s` is the real scalar
    part, and :math:`v` is the vector part) :math:`e_{o} = s v_d - s_d v - v_d \cross v`, and :math:`\dot{x}_d` is the
    desired cartesian velocity for the distal link with respect to the base link.

    This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A = ^bJ_d(q)`,
    :math:`x = \dot{q}`, and :math:`b = K_p e + \dot{x}_d`.

    Important notes:

    - You don't have to specify the whole pose, you can also only specify the position or orientation.
    - You can also only specify the cartesian velocities without providing the cartesian position or orientation.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, distal_link, base_link=None, local_position=(0, 0, 0), desired_position=None,
                 desired_orientation=None, desired_linear_velocity=None, desired_angular_velocity=None,
                 kp_position=1., kp_orientation=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the world.
            local_position (np.array[float[3]]): local position on the distal link.
            desired_position (np.array[float[3]], None): desired position of distal link wrt the base. If None, it
              will not be taken into account.
            desired_orientation (np.array[float[4]], None): desired orientation (expressed as quaternion [x,y,z,w]) of
              distal link wrt the base. If None, it will not be taken into account.
            desired_linear_velocity (np.array[float[3]], None): desired linear velocity of distal link wrt the base.
              If None, it will be set to zero.
            desired_angular_velocity (np.array[float[3]], None): desired angular velocity of distal link wrt the base.
              If None, it will be set to zero.
            kp_position (float, np.array[float[3,3]]): position stiffness gain.
            kp_orientation (float, np.array[float[3,3]]): orientation stiffness gain.
            weight (float, np.array[float[6,6]], np.array[float[3,3]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(CartesianTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define variables
        self.distal_link = self.model.get_link_id(distal_link)
        self.base_link = self.model.get_link_id(base_link) if base_link is not None else base_link
        self.local_position = local_position

        # gains
        self.kp_position = kp_position
        self.kp_orientation = kp_orientation

        # define desired references
        self.desired_position = desired_position
        self.desired_orientation = desired_orientation
        self.desired_linear_velocity = desired_linear_velocity
        self.desired_angular_velocity = desired_angular_velocity

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def desired_position(self):
        """Get the desired cartesian position for the distal link wrt the base."""
        return self._des_pos

    @desired_position.setter
    def desired_position(self, position):
        """Set the desired cartesian position for the distal link wrt the base."""
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
    def desired_orientation(self):
        """Get the desired cartesian orientation (expressed as a quaternion [x,y,z,w]) for the distal link wrt the
        base."""
        return self._des_quat

    @desired_orientation.setter
    def desired_orientation(self, orientation):
        """Set the desired cartesian orientation (expressed as a quaternion [x,y,z,w]) for the distal link wrt the
        base."""
        if orientation is not None:
            if not isinstance(orientation, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given desired orientation to be a np.array, instead got: "
                                "{}".format(type(orientation)))
            orientation = np.asarray(orientation)
            if len(orientation) != 4:
                raise ValueError(
                    "Expecting the given desired orientation array to be of length 4, but instead got: "
                    "{}".format(len(orientation)))
        self._des_quat = orientation

    @property
    def desired_linear_velocity(self):
        """Get the desired cartesian linear velocity of the distal link wrt the base."""
        return self._des_lin_vel

    @desired_linear_velocity.setter
    def desired_linear_velocity(self, velocity):
        """Set the desired cartesian linear velocity of the distal link wrt the base."""
        if velocity is None:
            velocity = np.zeros(3)
        elif not isinstance(velocity, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired linear velocity to be a np.array, instead got: "
                            "{}".format(type(velocity)))
        velocity = np.asarray(velocity)
        if len(velocity) != 3:
            raise ValueError("Expecting the given desired linear velocity array to be of length 3, but instead "
                             "got: {}".format(len(velocity)))
        self._des_lin_vel = velocity

    @property
    def desired_angular_velocity(self):
        """Get the desired cartesian angular velocity of the distal link wrt the base."""
        return self._des_ang_vel

    @desired_angular_velocity.setter
    def desired_angular_velocity(self, velocity):
        """Set the desired cartesian angular velocity of the distal link wrt the base."""
        if velocity is None:
            velocity = np.zeros(3)
        elif not isinstance(velocity, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired angular velocity to be a np.array, instead got: "
                            "{}".format(type(velocity)))
        velocity = np.asarray(velocity)
        if len(velocity) != 3:
            raise ValueError("Expecting the given desired angular velocity array to be of length 3, but instead "
                             "got: {}".format(len(velocity)))
        self._des_ang_vel = velocity

    @property
    def desired_velocity(self):
        """Return the linear and angular velocity."""
        return np.concatenate((self._des_lin_vel, self._des_ang_vel))

    @property
    def x_desired(self):
        """Get the desired cartesian pose for the distal link wrt to the base."""
        position = self.desired_position
        orientation = self.desired_orientation
        if position is not None:
            if orientation is not None:
                return np.concatenate((position, orientation))
            return position
        return orientation

    @x_desired.setter
    def x_desired(self, x_d):
        """Set the desired cartesian pose for the distal link wrt to the base."""
        if x_d is not None:
            if not isinstance(x_d, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given desired pose to be a np.array, instead got: {}".format(type(x_d)))
            x_d = np.asarray(x_d)
            if len(x_d) == 3:  # only position is provided
                x_d = np.concatenate((x_d, np.array([0., 0., 0., 1.])))
            elif len(x_d) == 4:  # only orientation is provided
                x_d = np.concatenate((np.zeros(3), x_d))
            if len(x_d) != 7:
                raise ValueError("Expecting the given desired pose array to be of length 7 (3 for the position, and 4 "
                                 "for the orientation expressed as a quaternion [x,y,z,w]), instead got a length of: "
                                 "{}".format(len(x_d)))
            self._des_pos = x_d[:3]
            self._des_quat = x_d[3:]

    @property
    def dx_desired(self):
        """Get the desired cartesian velocity for the distal link wrt to the base."""
        return np.concatenate((self._des_lin_vel, self._des_ang_vel))

    @dx_desired.setter
    def dx_desired(self, dx_d):
        """Set the desired cartesian velocity for the distal link wrt to the base."""
        if dx_d is not None:
            if not isinstance(dx_d, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given desired velocity to be a np.array, instead got: "
                                "{}".format(type(dx_d)))
            dx_d = np.asarray(dx_d)
            if len(dx_d) == 3:  # assume that it is the linear velocity
                dx_d = np.concatenate((dx_d, np.zeros(3)))
            if len(dx_d) != 6:
                raise ValueError("Expecting the given desired velocity array to be of length 6 (3 for the linear and "
                                 "3 for the angular part), instead got a length of: {}".format(len(dx_d)))
            self._des_lin_vel = dx_d[:3]
            self._des_ang_vel = dx_d[3:]

    @property
    def kp_position(self):
        """Return the position stiffness gain."""
        return self._kp_pos

    @kp_position.setter
    def kp_position(self, kp):
        """Set the position stiffness gain."""
        if kp is None:
            kp = 1.
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given position stiffness gain kp to be an int, float, np.array, instead "
                            "got: {}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (3, 3):
            raise ValueError("Expecting the given position stiffness gain matrix kp to be of shape {}, but instead "
                             "got shape: {}".format((3, 3), kp.shape))
        self._kp_pos = kp

    @property
    def kp_orientation(self):
        """Return the orientation stiffness gain."""
        return self._kp_quat

    @kp_orientation.setter
    def kp_orientation(self, kp):
        """Set the orientation stiffness gain."""
        if kp is None:
            kp = 1.
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given orientation stiffness gain kp to be an int, float, np.array, instead "
                            "got: {}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (3, 3):
            raise ValueError("Expecting the given orientation stiffness gain matrix kp to be of shape {}, but instead "
                             "got shape: {}".format((3, 3), kp.shape))
        self._kp_quat = kp

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, dx_des=None, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[float[7]], None): desired cartesian pose (position and quaternion [x,y,z,w]) of distal
              link wrt the base. If None, it will let the initial desired pose unchanged.
            dx_des (np.array[float[6]], None): desired cartesian velocity of distal link wrt the base. If None, it
              will let the initial desired velocities unchanged.
        """
        self.x_desired = x_des
        self.dx_desired = dx_des

    def get_desired_references(self):
        """Return the desired references.

        Returns:
            np.array[float[7]]: desired cartesian pose (position and quaternion [x,y,z,w]) of distal link wrt the base.
            np.array[float[6]]: desired cartesian velocity of distal link wrt the base.
        """
        return self.x_desired, self.dx_desired

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        x = self.model.get_pose(self.distal_link, self.base_link)
        self._A = self.model.get_jacobian(link=self.distal_link, wrt_link=self.base_link,
                                          point=self.local_position)  # shape: (6,N)

        if self._des_quat is None:  # only position and/or velocities
            if self._des_pos is None:  # only velocities
                self._b = np.concatenate((self._des_lin_vel, self._des_ang_vel))
            else:  # only position
                self._A = self._A[:3]
                # compute position error
                error = (self._des_pos - x[:3])
                # compute b vector
                self._b = np.dot(self.kp_position, error) + self._des_lin_vel
        elif self._des_pos is None:  # only orientation
            self._A = self._A[3:]
            # compute orientation error
            error = quaternion_error(quat_des=self._des_quat, quat_cur=x[3:])
            # compute b vector
            self._b = np.dot(self.kp_orientation, error) + self._des_ang_vel
        else:  # both
            # compute position/orientation error
            position_error = (self._des_pos - x[:3])
            orientation_error = quaternion_error(quat_des=self._des_quat, quat_cur=x[3:])

            # compute b vector
            b_position = np.dot(self.kp_position, position_error) + self._des_lin_vel
            b_orientation = np.dot(self.kp_orientation, orientation_error) + self._des_ang_vel
            self._b = np.concatenate((b_position, b_orientation))
