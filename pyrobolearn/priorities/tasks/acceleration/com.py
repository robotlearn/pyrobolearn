#!/usr/bin/env python
r"""Provide the Cartesian CoM acceleration task.

The CoM task tries to impose a desired position of the CoM with respect to the world frame.

.. math:: ||J_{CoM} \dot{q} - (K_p (x_d - x) + \dot{x}_d)||^2

where :math:`J_{CoM}` is the CoM Jacobian, :math:`\dot{q}` are the joint velocities being optimized, :math:`K_p`
is the stiffness gain, :math:`x_d` and :math:`x` are the desired and current cartesian CoM position
respectively, and :math:`\dot{x}_d` is the desired linear velocity of the CoM.

This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=J_{CoM}`,
:math:`x=\dot{q}`, and :math:`b = K_p (x_d - x) + \dot{x}_d`.

Note that you can only specify the center of mass linear velocity if you wish.

The CoM acceleration task tries to impose a desired pose, velocity and acceleration profiles for the CoM with respect
to the world frame.

Before presenting the optimization problem, here is a small reminder. The acceleration is the time derivative of
the velocity, i.e. :math:`a = \frac{dv}{dt}` where the cartesian velocities are related to joint velocities by
:math:`v_{CoM} = J_{CoM}(q) \dot{q}` where :math:`J_{CoM}(q)` is the CoM Jacobian, thus deriving that expression wrt
time gives us:

.. math:: a = \frac{d}{dt} v = \frac{d}{dt} J_{CoM}(q) \dot{q} = J_{CoM}(q) \ddot{q} + \dot{J}_{CoM}(q) \dot{q}.

Now, we can formulate our minimization problem as:

.. math:: || J_{CoM}(q) \ddot{q} + \dot{J}_{CoM} \dot{q} - (a_d + K_d (v_d - v) + K_p (x_d - x)) ||^2,

where :math:`\ddot{q}` are the joint accelerations being optimized, :math:`a_d` are the desired cartesian linear
accelerations, :math:`v_d` and :math:`v` are the desired and current cartesian linear velocities,
:math:`J_{CoM}(q) \in \mathbb{R}^{3 \times N}` is the CoM Jacobian, :math:`K_p` and :math:`K_d` are the stiffness and
damping gains respectively, and :math:`x_d` and :math:`x` are the desired and current cartesian CoM position
respectively.


The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
:math:`A = J_{CoM}(q)`, :math:`x = \ddot{q}`, and
:math:`b = - \dot{J}_{CoM} \dot{q} + (a_d + K_d (v_d - v) + K_p (x_d - x))`.


Inverse dynamics
----------------

Once the optimal joint accelerations :math:`\ddot{q}^*` have been computed, we can use inverse dynamics to
compute the corresponding torques to apply on the joints. This is given by:

.. math:: \tau = H(q) \ddot{q} + N(q,\dot{q)}

where :math:`H(q)` is the inertia joint matrix, and N(q, \dot{q}) is a vector force that accounts for all the
other non-linear forces acting on the system (Coriolis, centrifugal, gravity, external forces, friction, etc.).


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointAccelerationTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Songyan Xin (insight)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CoMAccelerationTask(JointAccelerationTask):
    r"""CoM Acceleration Task

    The CoM task tries to impose a desired position of the CoM with respect to the world frame.

    .. math:: ||J_{CoM} \dot{q} - (K_p (x_d - x) + \dot{x}_d)||^2

    where :math:`J_{CoM}` is the CoM Jacobian, :math:`\dot{q}` are the joint velocities being optimized, :math:`K_p`
    is the stiffness gain, :math:`x_d` and :math:`x` are the desired and current cartesian CoM position
    respectively, and :math:`\dot{x}_d` is the desired linear velocity of the CoM.

    This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=J_{CoM}`,
    :math:`x=\dot{q}`, and :math:`b = K_p (x_d - x) + \dot{x}_d`.

    Note that you can only specify the center of mass linear velocity if you wish.

    The CoM acceleration task tries to impose a desired pose, velocity and acceleration profiles for the CoM wrt the
    world frame.

    Before presenting the optimization problem, here is a small reminder. The acceleration is the time derivative of
    the velocity, i.e. :math:`a = \frac{dv}{dt}` where the cartesian velocities are related to joint velocities by
    :math:`v_{CoM} = J_{CoM}(q) \dot{q}` where :math:`J_{CoM}(q)` is the CoM Jacobian, thus deriving that expression
    wrt time gives us:

    .. math:: a = \frac{d}{dt} v = \frac{d}{dt} J_{CoM}(q) \dot{q} = J_{CoM}(q) \ddot{q} + \dot{J}_{CoM}(q) \dot{q}.

    Now, we can formulate our minimization problem as:

    .. math:: || J_{CoM}(q) \ddot{q} + \dot{J}_{CoM} \dot{q} - (a_d + K_d (v_d - v) + K_p (x_d - x)) ||^2,

    where :math:`\ddot{q}` are the joint accelerations being optimized, :math:`a_d` are the desired cartesian linear
    accelerations, :math:`v_d` and :math:`v` are the desired and current cartesian linear velocities,
    :math:`J_{CoM}(q) \in \mathbb{R}^{3 \times N}` is the CoM Jacobian, :math:`K_p` and :math:`K_d` are the stiffness
    and damping gains respectively, and :math:`x_d` and :math:`x` are the desired and current cartesian CoM position
    respectively.


    The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
    :math:`A = J_{CoM}(q)`, :math:`x = \ddot{q}`, and
    :math:`b = - \dot{J}_{CoM} \dot{q} + (a_d + K_d (v_d - v) + K_p (x_d - x))`.


    Inverse dynamics
    ----------------

    Once the optimal joint accelerations :math:`\ddot{q}^*` have been computed, we can use inverse dynamics to
    compute the corresponding torques to apply on the joints. This is given by:

    .. math:: \tau = H(q) \ddot{q} + N(q,\dot{q)}

    where :math:`H(q)` is the inertia joint matrix, and N(q, \dot{q}) is a vector force that accounts for all the
    other non-linear forces acting on the system (Coriolis, centrifugal, gravity, external forces, friction, etc.).


    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, desired_position=None, desired_velocity=None, desired_acceleration=None, kp=1., kd=1.,
                 weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            desired_position (np.array[float[3]], None): desired CoM position. If None, it will be set to 0.
            desired_velocity (np.array[float[3]], None): desired CoM linear velocity. If None, it will be set to 0.
            desired_acceleration (np.array[float[3]], None): desired CoM linear acceleration. If None, it will be set
              to 0.
            kp (float, np.array[float[3,3]]): position gain.
            kd (float, np.array[float[3,3]]): linear velocity gain.
            weight (float, np.array[float[3,3]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(CoMAccelerationTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define variable
        self.kp = kp
        self.kd = kd

        # define desired references
        self.desired_position = desired_position
        self.desired_velocity = desired_velocity
        self.desired_acceleration = desired_acceleration

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
    def desired_acceleration(self):
        """Get the desired CoM linear acceleration."""
        return self._des_acc

    @desired_acceleration.setter
    def desired_acceleration(self, acceleration):
        """Set the desired CoM linear acceleration."""
        if acceleration is None:
            acceleration = np.zeros(3)
        elif not isinstance(acceleration, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired linear acceleration to be a np.array, instead got: "
                            "{}".format(type(acceleration)))
        acceleration = np.asarray(acceleration)
        if len(acceleration) != 3:
            raise ValueError("Expecting the given desired linear acceleration array to be of length 3, but instead "
                             "got: {}".format(len(acceleration)))
        self._des_acc = acceleration

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
    def ddx_desired(self):
        """Get the desired CoM linear acceleration."""
        return self._des_acc

    @ddx_desired.setter
    def ddx_desired(self, ddx_d):
        """Set the desired CoM linear acceleration."""
        self.desired_acceleration = ddx_d

    @property
    def kp(self):
        """Return the position gain."""
        return self._kp

    @kp.setter
    def kp(self, kp):
        """Set the position gain."""
        if kp is None:
            kp = 1.
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given position gain kp to be an int, float, np.array, instead got: "
                            "{}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (3, 3):
            raise ValueError("Expecting the given position gain matrix kp to be of shape {}, but instead got "
                             "shape: {}".format((self.x_size, self.x_size), kp.shape))
        self._kp = kp

    @property
    def kd(self):
        """Return the linear velocity gain."""
        return self._kd

    @kd.setter
    def kd(self, kd):
        """Set the linear velocity gain."""
        if kd is None:
            kd = 1.
        if not isinstance(kd, (float, int, np.ndarray)):
            raise TypeError("Expecting the given linear velocity gain kd to be an int, float, np.array, "
                            "instead got: {}".format(type(kd)))
        if isinstance(kd, np.ndarray) and kd.shape != (3, 3):
            raise ValueError("Expecting the given linear velocity gain matrix kd to be of shape {}, but "
                             "instead got shape: {}".format((3, 3), kd.shape))
        self._kd = kd

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, dx_des=None, ddx_des=None, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[float[3]], None): desired CoM position. If None, it will be set to 0.
            dx_des (np.array[float[3]], None): desired CoM linear velocity. If None, it will be set to 0.
            ddx_des (np.array[float[3]], None): desired CoM linear velocity. If None, it will be set to 0.
        """
        self.x_desired = x_des
        self.dx_desired = dx_des
        self.ddx_desired = ddx_des

    def get_desired_references(self):
        """Return the desired references.

        Returns:
            np.array[float[3]]: desired CoM position.
            np.array[float[3]]: desired CoM linear velocity.
        """
        return self.x_desired, self.dx_desired, self.ddx_desired

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """

        self._A = self.model.get_com_jacobian(full=False)  # shape: (3, N)
        jdotqdot = self.model.compute_com_JdotQdot()[:3]  # shape: (3,)
        vel = self.model.get_com_velocity()  # shape: (3,)
        self._b = -jdotqdot + self._des_acc + np.dot(self.kd, (self._des_vel - vel))  # shape: (3,)

        if self._des_pos is not None:
            x = self.model.get_com_position()
            self._b = self._b + np.dot(self.kp, (self._des_pos - x))  # shape: (3,)
