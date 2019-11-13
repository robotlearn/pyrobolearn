#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the cartesian impedance control task.

The cartesian impedance control task optimizes the joint torques such that it applies the necessary torques to
move a distal link with respect to a base:

.. math:: || J(q) H(q)^{-1} \tau - J(q) H(q)^{-1} J(q)^\top f ||^2 = || J(q) H(q)^{-1} (\tau - J(q)^\top f) ||^2

where :math:`J(q) \in \mathbb{R}^{6 \times N}` is the Jacobian matrix, :math:`H(q) \in \mathbb{R}^{N \times N}` is
the joint inertia matrix, :math:`\tau \in \mathbb{R}^N` are the torques being optimized, and
:math:`f \in \mathbb{R}^6` is the desired wrench computed from:

.. math:: f = K_p e + K_d (\dot{x}_d - \dot{x})

where :math:`K_p` and :math:`K_d` are the stiffness and damping gains, :math:`e \in \mathbb{R}^{6}` is the error
which is the concatenation of the position error given by :math:`e_{p} = (x_d - x)` (with :math:`x_d` being the
desired pose, and :math:`x` the current pose), and the orientation error given by (if expressed as quaternions
:math:`o = {s, v}` where :math:`s` is the real scalar part, and :math:`v` is the vector part)
:math:`e_{o} = s v_d - s_d v - v_d \cross v`, and :math:`\dot{x}_d \in \mathbb{R}^{6}` is the desired cartesian
velocity for the distal link with respect to the base link.

The above optimization problem is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
:math:`A = J(q) H(q)^{-1}`, :math:`x = \tau`, and :math:`b = J(q) H(q)^{-1} J(q)^\top f`.

Note that :math:`||J(q) H(q)^{-1} (\tau - J(q)^\top f)||^2  \leq  ||J(q) H(q)^{-1}|| ||\tau - J(q)^\top f||^2`.

.. seealso:: `tasks/velocity/cartesian.py`

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointTorqueTask
from pyrobolearn.utils.transformation import quaternion_error


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CartesianImpedanceControlTask(JointTorqueTask):
    r"""Cartesian Impedance Control Task

    The cartesian impedance control task optimizes the joint torques such that it applies the necessary torques to
    move a distal link with respect to a base:

    .. math:: || J(q) H(q)^{-1} \tau - J(q) H(q)^{-1} J(q)^\top f ||^2 = || J(q) H(q)^{-1} (\tau - J(q)^\top f) ||^2

    where :math:`J(q) \in \mathbb{R}^{6 \times N}` is the Jacobian matrix, :math:`H(q) \in \mathbb{R}^{N \times N}` is
    the joint inertia matrix, :math:`\tau \in \mathbb{R}^N` are the torques being optimized, and
    :math:`f \in \mathbb{R}^6` is the desired wrench computed from:

    .. math:: f = K_p e + K_d (\dot{x}_d - \dot{x})

    where :math:`K_p` and :math:`K_d` are the stiffness and damping gains, :math:`e \in \mathbb{R}^{6}` is the error
    which is the concatenation of the position error given by :math:`e_{p} = (x_d - x)` (with :math:`x_d` being the
    desired pose, and :math:`x` the current pose), and the orientation error given by (if expressed as quaternions
    :math:`o = {s, v}` where :math:`s` is the real scalar part, and :math:`v` is the vector part)
    :math:`e_{o} = s v_d - s_d v - v_d \cross v`, and :math:`\dot{x}_d \in \mathbb{R}^{6}` is the desired cartesian
    velocity for the distal link with respect to the base link.

    The above optimization problem is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
    :math:`A = J(q) H(q)^{-1}`, :math:`x = \tau`, and :math:`b = J(q) H(q)^{-1} J(q)^\top f`.

    Note that :math:`||J(q) H(q)^{-1} (\tau - J(q)^\top f)||^2  \leq  ||J(q) H(q)^{-1}|| ||\tau - J(q)^\top f||^2`.

    .. seealso:: `tasks/velocity/cartesian.py`


    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, distal_link, base_link=None, local_position=(0, 0, 0), desired_position=None,
                 desired_orientation=None, desired_linear_velocity=None, desired_angular_velocity=None,
                 kp_position=1., kp_orientation=1., kd_linear=1., kd_angular=1., weight=1., constraints=[]):
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
            kd_linear (float, np.array[float[3,3]]): linear velocity damping gain.
            kd_angular (float, np.array[float[3,3]]): angular velocity damping gain.
            kp (float, np.array[float[6,6]]): stiffness gain.
            kd (float, np.array[float[6,6]]): damping gain.
            weight (float, np.array[float[6,6]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(CartesianImpedanceControlTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define variables
        self.distal_link = self.model.get_link_id(distal_link)
        self.base_link = self.model.get_link_id(base_link) if base_link is not None else base_link
        self.local_position = local_position

        # gains
        self.kp_position = kp_position
        self.kp_orientation = kp_orientation
        self.kd_linear = kd_linear
        self.kd_angular = kd_angular

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
                raise TypeError(
                    "Expecting the given desired pose to be a np.array, instead got: {}".format(type(x_d)))
            x_d = np.asarray(x_d)
            if len(x_d) == 3:  # only position is provided
                x_d = np.concatenate((x_d, np.array([0., 0., 0., 1.])))
            elif len(x_d) == 4:  # only orientation is provided
                x_d = np.concatenate((np.zeros(3), x_d))
            if len(x_d) != 7:
                raise ValueError("Expecting the given desired pose array to be of length 7 (3 for the position, "
                                 "and 4 for the orientation expressed as a quaternion [x,y,z,w]), instead got a "
                                 "length of: {}".format(len(x_d)))
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
                raise ValueError("Expecting the given desired velocity array to be of length 6 (3 for the linear "
                                 "and 3 for the angular part), instead got a length of: {}".format(len(dx_d)))
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
            raise TypeError("Expecting the given orientation stiffness gain kp to be an int, float, np.array, "
                            "instead got: {}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (3, 3):
            raise ValueError("Expecting the given orientation stiffness gain matrix kp to be of shape {}, but "
                             "instead got shape: {}".format((3, 3), kp.shape))
        self._kp_quat = kp

    @property
    def kd_linear(self):
        """Return the linear velocity damping gain."""
        return self._kd_lin

    @kd_linear.setter
    def kd_linear(self, kd):
        """Set the linear velocity damping gain."""
        if kd is None:
            kd = 1.
        if not isinstance(kd, (float, int, np.ndarray)):
            raise TypeError("Expecting the given linear velocity damping gain kd to be an int, float, np.array, "
                            "instead got: {}".format(type(kd)))
        if isinstance(kd, np.ndarray) and kd.shape != (3, 3):
            raise ValueError("Expecting the given linear velocity damping gain matrix kd to be of shape {}, but "
                             "instead got shape: {}".format((3, 3), kd.shape))
        self._kd_lin = kd

    @property
    def kd_angular(self):
        """Return the angular velocity damping gain."""
        return self._kd_ang

    @kd_angular.setter
    def kd_angular(self, kd):
        """Set the angular velocity damping gain."""
        if kd is None:
            kd = 1.
        if not isinstance(kd, (float, int, np.ndarray)):
            raise TypeError("Expecting the given angular velocity damping gain kd to be an int, float, np.array, "
                            "instead got: {}".format(type(kd)))
        if isinstance(kd, np.ndarray) and kd.shape != (3, 3):
            raise ValueError("Expecting the given angular velocity damping gain matrix kd to be of shape {}, but "
                             "instead got shape: {}".format((3, 3), kd.shape))
        self._kd_ang = kd

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, dx_des=None, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[float[7]], None): desired cartesian pose (position and quaternion [x,y,z,w]) of distal 
              link wrt the base. If None, it will let the initial desired pose unchanged.
            dx_des (np.array[float[6]], None): desired cartesian velocity of distal link wrt the base. If None,
              it will let the initial desired accelerations unchanged.
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
        # get useful variables
        x = self.model.get_pose(link=self.distal_link, wrt_link=self.base_link)  # (7,)
        dx = self.model.get_velocity(link=self.distal_link, wrt_link=self.base_link)  # (6,)
        jac = self.model.get_jacobian(link=self.distal_link, wrt_link=self.base_link,
                                      point=self.local_position)  # shape: (6,N)
        H = self.model.get_inertia_matrix()  # shape: (N,N)
        H_inv = np.linalg.inv(H)

        if self._des_quat is None:  # only position and/or velocities
            if self._des_pos is None:  # only velocities
                force = np.concatenate((np.dot(self.kd_linear, (self._des_lin_vel - dx[:3])),
                                        np.dot(self.kd_angular, (self._des_ang_vel - dx[3:]))))
            else:  # only position
                jac = jac[:3]
                position = np.dot(self.kp_position, (self._des_pos - x[:3]))
                lin_vel = np.dot(self.kd_linear, (self._des_lin_vel - dx[:3]))
                force = position + lin_vel
        elif self._des_pos is None:  # only orientation
            jac = jac[3:]
            orientation = np.dot(self.kp_orientation, quaternion_error(quat_des=self._des_quat, quat_cur=x[3:]))
            ang_vel = np.dot(self.kd_angular, (self._des_ang_vel - dx[3:]))
            force = orientation + ang_vel
        else:  # both
            # compute position/orientation error
            position = np.dot(self.kp_position, (self._des_pos - x[:3]))
            orientation = np.dot(self.kp_orientation, quaternion_error(quat_des=self._des_quat, quat_cur=x[3:]))
            # compute velocities
            lin_vel = np.dot(self.kd_linear, (self._des_lin_vel - dx[:3]))
            ang_vel = np.dot(self.kd_angular, (self._des_ang_vel - dx[3:]))
            force = np.concatenate((position + lin_vel, orientation + ang_vel))

        # compute A matrix and b vector
        self._A = jac.dot(H_inv)
        self._b = self._A.dot(jac.T.dot(force))
