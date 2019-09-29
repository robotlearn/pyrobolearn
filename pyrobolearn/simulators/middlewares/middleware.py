# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the abstract middleware API.

Dependencies in PRL:
* NONE
"""

# TODO
import os
import subprocess
import psutil
import signal
import importlib
import inspect


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MiddleWare(object):
    r"""Middleware (abstract) class

    Middlewares can be provided to simulators which can then use them to send/receive messages.
    """

    def __init__(self, subscribe=False, publish=False, teleoperate=False):
        """
        Initialize the middleware to communicate.

        Args:
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robots, and will read
              the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robots.
            teleoperate (bool): if True, it will move the robot based on the received or sent values based on the 2
              previous attributes :attr:`subscribe` and :attr:`publish`.
        """
        # set variables
        self.subscribe = subscribe
        self.publish = publish
        self.teleoperate = teleoperate

    ##############
    # Properties #
    ##############

    @property
    def subscribe(self):
        return self._subscribe

    @subscribe.setter
    def subscribe(self, subscribe):
        self._subscribe = bool(subscribe)

    @property
    def publish(self):
        return self._publish

    @publish.setter
    def publish(self, publish):
        self._publish = bool(publish)

    @property
    def teleoperate(self):
        return self._teleoperate

    @teleoperate.setter
    def teleoperate(self, teleoperate):
        self._teleoperate = bool(teleoperate)

    ###########
    # Methods #
    ###########

    def has_sensor(self, body_id, name):
        """
        Check if the specified robot has the given sensor.

        Args:
            body_id (int): body unique id.
            name (str): name of the sensor.

        Returns:
            bool: True if the specified robot has the given sensor.
        """
        pass

    def get_sensor_values(self, body_id, name):
        """
        Return the sensor.

        Args:
            body_id (int): body unique id.
            name (str): name of the sensor.

        Returns:
            np.array, dict, list, None: sensor values. None if it didn't have anything.
        """
        pass

    def get_joint_positions(self, body_id, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        pass

    def set_joint_positions(self, body_id, joint_ids, positions, velocities=None, kps=None, kds=None, forces=None,
                            check_teleoperate=False):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            positions (float, np.array[float[N]]): desired position, or list of desired positions [rad]
            velocities (None, float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            kps (None, float, np.array[float[N]]): position gain(s)
            kds (None, float, np.array[float[N]]): velocity gain(s)
            forces (None, float, np.array[float[N]]): maximum motor force(s)/torque(s) used to reach the target values.
            check_teleoperate (bool): if True, it will check if the given `teleoperate` argument has been set to True,
              and if so, it will set the joint positions. If the `teleoperate` argument has been set to False, it
              won't set the joint positions.
        """
        pass

    def get_joint_velocities(self, body_id, joint_ids):
        """
        Get the velocity of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.array[float[N]]: joint velocities [rad/s]
        """
        pass

    def set_joint_velocities(self, body_id, joint_ids, velocities, max_force=None, check_teleoperate=False):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            velocities (float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            max_force (None, float, np.array[float[N]]): maximum motor forces/torques.
            check_teleoperate (bool): if True, it will check if the given `teleoperate` argument has been set to True,
              and if so, it will set the joint velocities. If the `teleoperate` argument has been set to False, it
              won't set the joint velocities.
        """
        pass

    def get_joint_torques(self, body_id, joint_ids):
        """
        Get the applied torque(s) on the given joint(s). "This is the motor torque applied during the last `step`.
        Note that this only applies in VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the
        applied joint motor torque is exactly what you provide, so there is no need to report it separately." [1]

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.array[float[N]]: torques associated to the given joints [Nm]
        """
        pass

    def set_joint_torques(self, body_id, joint_ids, torques, check_teleoperate=False):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            torques (float, list[float]): desired torque(s) to apply to the joint(s) [N].
            check_teleoperate (bool): if True, it will check if the given `teleoperate` argument has been set to True,
              and if so, it will set the joint torques. If the `teleoperate` argument has been set to False, it won't
              set the joint torques.
        """
        pass

    def get_jacobian(self, body_id, link_id, local_position=None, q=None):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q), J_{ang}(q)]^T`, such that:

        .. math:: v = [\dot{p}, \omega]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            body_id (int): unique body id.
            link_id (int): link id.
            local_position (np.array[float[3]]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
            q (np.array[float[N]]): joint positions of size N, where N is the number of DoFs.

        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: full geometric (linear and angular) Jacobian matrix. The
                number of columns depends if the base is fixed or floating.
        """
        pass

    def get_inertia_matrix(self, body_id, q):
        r"""
        Return the mass/inertia matrix :math:`H(q)`, which is used in the rigid-body equation of motion (EoM) in joint
        space given by (see [1]):

        .. math:: \tau = H(q)\ddot{q} + C(q,\dot{q})

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Warnings: If the base is floating, it will return a [6+N,6+N] inertia matrix, where N is the number of actuated
            joints. If the base is fixed, it will return a [N,N] inertia matrix

        Args:
            body_id (int): body unique id.
            q (np.array[float[N]]): joint positions of size N, where N is the total number of DoFs.

        Returns:
            np.array[float[N,N]], np.array[float[6+N,6+N]]: inertia matrix
        """
        pass
