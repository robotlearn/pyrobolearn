# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the various joint actuators used in robotics.
"""

import copy
import numpy as np
from abc import ABCMeta

from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.robots.actuators.actuator import Actuator
from pyrobolearn.robots.base import Body

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointActuator(Actuator):
    r"""Joint Actuators (abstract)

    This defined the joint actuator class; this is an actuator which is attached to a joint and outputs the torque
    to be applied on it using a specific control scheme (e.g. PD control).
    For instance, given a target joint position value, the actuator computes the necessary torque to be applied on
    the joint using a simple PD control (with certain gains).
    """
    __metaclass__ = ABCMeta

    def __init__(self, simulator, body_id, joint_ids, ticks=1, latency=None):
        """
        Initialize the joint actuator.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            ticks (int): number of steps to wait/sleep before acting in the world.
            latency (int, float, None): latency time / step.
        """
        super(JointActuator, self).__init__(ticks=ticks, latency=latency)

        # setting simulator
        if not isinstance(simulator, Simulator):
            raise TypeError("Expecting the given 'simulator' to be an instance of `Simulator`, but got instead: "
                            "{}".format(type(simulator)))
        self.sim = simulator

        # set the body id
        if isinstance(body_id, Body):
            body_id = body_id.id
        elif not isinstance(body_id, (int, long)):
            raise TypeError("Expecting the given 'body_id' to be an int or an instance of `Body`, but got instead: "
                            "{}".format(type(body_id)))
        if body_id < 0:
            raise ValueError("Expecting the given 'body_id' to be a positive integer, but got instead: "
                             "{}".format(body_id))
        self.body_id = body_id

        # set the joint ids
        if joint_ids is None:
            # get actuated joints
            joint_ids = []
            for joint_id in range(self.sim.num_joints(self.body_id)):
                joint_info = self.sim.get_joint_info(self.body_id, joint_id)
                if joint_info[2] != self.sim.JOINT_FIXED:  # if not a fixed joint
                    joint_ids.append(joint_info[0])
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        elif isinstance(joint_ids, (tuple, list)):
            for i, joint in enumerate(joint_ids):
                if not isinstance(joint, int):
                    raise TypeError("Expecting the given {}th joint id to be an int, instead got: {}".format(i, joint))
        else:
            raise TypeError("Expecting the given 'joint_ids' to be an int or list of int, instead got: "
                            "{}".format(joint_ids))
        self.joint_ids = joint_ids
        self.q_indices = self.sim.get_q_indices(self.body_id, self.joint_ids)

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self.sim

    #############
    # Operators #
    #############

    def __copy__(self):
        """Return a shallow copy of the actuator. This can be overridden in the child class."""
        return self.__class__(simulator=self.sim, body_id=self.body_id, joint_ids=self.joint_ids, ticks=self._ticks,
                              latency=self._latency)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the actuator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        simulator = memo.get(self.simulator, self.simulator)  # copy.deepcopy(self.simulator, memo)
        joint_ids = copy.deepcopy(self.joint_ids)
        actuator = self.__class__(simulator=simulator, body_id=self.body_id, joint_ids=joint_ids, ticks=self._ticks,
                                  latency=self._latency)
        memo[self] = actuator
        return actuator


class JointPositionActuator(JointActuator):
    r"""Joint position actuator

    """

    def __init__(self, simulator, body_id, joint_ids, kps=None, kds=None, forces=None, ticks=1, latency=None):
        """
        Initialize the joint position actuator.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            kps (None, float, np.array[N]): position gain(s)
            kds (None, float, np.array[N]): velocity gain(s)
            forces (None, float, np.array[N]): maximum motor force(s)/torque(s) used to reach the target values.
            ticks (int): number of steps to wait/sleep before acting in the world.
            latency (int, float, None): latency time / step.
        """
        super(JointPositionActuator, self).__init__(simulator=simulator, body_id=body_id, joint_ids=joint_ids,
                                                    ticks=ticks, latency=latency)
        # set gains
        self._kps = kps
        self._kds = kds
        self._forces = forces

    def set_joint_positions(self, positions, velocities=None):
        r"""
        Set the position of the given joint(s) (using position control).

        Args:
            positions (float, np.array[N]): desired position, or list of desired positions [rad]
            velocities (float, np.array[N], None): desired velocity, or list of desired velocities [rad/s]
        """
        self.sim.set_joint_positions(body_id=self.body_id, joint_ids=self.joint_ids, positions=positions,
                                     velocities=velocities, kps=self._kps, kds=self._kds, forces=self._forces)

    def _act(self):
        """Act using the actuator by setting the joint positions."""
        self.set_joint_positions(positions=self._data)


class JointVelocityActuator(JointActuator):
    r"""Joint velocity actuator

    """

    def __init__(self, simulator, body_id, joint_ids, max_force=None, ticks=1, latency=None):
        """
        Initialize the joint velocity actuator.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            max_force (float, float[N], None): maximum allowed force/torque for each joint.
            ticks (int): number of steps to wait/sleep before acting in the world.
            latency (int, float, None): latency time / step.
        """
        super(JointVelocityActuator, self).__init__(simulator=simulator, body_id=body_id, joint_ids=joint_ids,
                                                    ticks=ticks, latency=latency)

        # set max force
        self._max_force = max_force

    def set_joint_velocities(self, velocities):
        r"""
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            velocities (float, np.array[N]): desired velocity, or list of desired velocities [rad/s]
        """
        self.sim.set_joint_velocities(body_id=self.body_id, joint_ids=self.joint_ids, velocities=velocities,
                                      max_force=self._max_force)

    def _act(self):
        """Act using the actuator by setting the joint velocities."""
        self.set_joint_velocities(velocities=self._data)


class JointPositionVelocityActuator(JointPositionActuator):
    r"""Joint position velocity actuator

    """

    def __init__(self, simulator, body_id, joint_ids, kps=None, kds=None, forces=None, ticks=1, latency=None):
        """
        Initialize the joint position velocity actuator.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            kps (None, float, np.array[N]): position gain(s)
            kds (None, float, np.array[N]): velocity gain(s)
            forces (None, float, np.array[N]): maximum motor force(s)/torque(s) used to reach the target values.
            ticks (int): number of steps to wait/sleep before acting in the world.
            latency (int, float, None): latency time / step.
        """
        super(JointPositionVelocityActuator, self).__init__(simulator=simulator, body_id=body_id, joint_ids=joint_ids,
                                                            ticks=ticks, latency=latency)
        # set gains
        self._kps = kps
        self._kds = kds
        self._forces = forces

    def _act(self):
        """Act using the actuator by setting the joint positions and velocities."""
        middle_idx = int(len(self._data) / 2)
        positions, velocities = self._data[:middle_idx], self._data[middle_idx:]
        self.set_joint_positions(positions=positions, velocities=velocities)


class JointTorqueActuator(JointActuator):
    r"""Joint torque actuator

    """

    def __init__(self, simulator, body_id, joint_ids, ticks=1, latency=None):
        """
        Initialize the joint torque actuator.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will get all the actuated joints.
            ticks (int): number of steps to wait/sleep before acting in the world.
            latency (int, float, None): latency time / step.
        """
        super(JointTorqueActuator, self).__init__(simulator=simulator, body_id=body_id, joint_ids=joint_ids,
                                                  ticks=ticks, latency=latency)

    def set_joint_torques(self, torques=None):
        r"""
        Set the torque to the given joint(s) (using force/torque control).

        Args:
            torques (float, np.array[N], None): desired torque(s) to apply to the joint(s) [N]. If None, it will apply
                a torque of 0 to the given joint(s).
        """
        if torques is None:
            torques = [0] * len(self.joint_ids)
        elif isinstance(torques, (int, float)):
            torques = [torques] * len(self.joint_ids)

        self.sim.set_joint_torques(self.body_id, joint_ids=self.joint_ids, torques=torques)

    def _act(self):
        """Act using the actuator by setting the joint torques."""
        self.set_joint_torques(torques=self._data)


############################################################

# class PDJointActuator(JointActuator):  # see also utils/feedback.py
#     r"""PD Joint Actuator
#
#     Compute the torque to be applied on the joint using a PD controller: :math:`\tau = k_p (q_d - q) - k_d \dot{q}`,
#     where :math:`q` and :math:`\dot{q}` are the current joint position and velocity respectively, :math:`q_d` is
#     the desired joint position, and :math:`k_p` and :math:`k_d` are the PD gains.
#     """
#
#     def __init__(self, joint_id, kp=0, kd=0, min_torque=-np.infty, max_torque=np.infty, latency=0):
#         """
#         Initialize the PD joint actuator.
#
#         Args:
#             joint_id (int): joint id.
#             kp (float): position gain
#             kd (float): velocity gain
#             min_torque (float): minimum torque
#             max_torque (float): maximum torque
#             latency (int, float, None): latency time / step.
#         """
#         super(PDJointActuator, self).__init__(joint_id, latency=latency)
#         self.kp = kp
#         self.kd = kd
#         self.min_torque = min_torque
#         self.max_torque = max_torque
#
#     def compute(self, qd, q, dq):
#         """
#         Compute and return the torque using the PD control scheme.
#
#         Args:
#             qd (float): desired joint position
#             q (float): current joint position
#             dq (float): current joint velocity
#
#         Returns:
#             float: computed torque using PD control
#         """
#         torque = self.kp * (qd - q) - self.kd * dq
#         torque = np.clip(torque, self.min_torque, self.max_torque)
#         return torque
#
#     def __copy__(self):
#         """Return a shallow copy of the actuator. This can be overridden in the child class."""
#         return self.__class__(joint_id=self.joint_id, kp=self.kp, kd=self.kd, min_torque=self.min_torque,
#                               max_torque=self.max_torque, latency=self.latency)
#
#     def __deepcopy__(self, memo={}):
#         """Return a deep copy of the actuator. This can be overridden in the child class.
#
#         Args:
#             memo (dict): memo dictionary of objects already copied during the current copying pass
#         """
#         joint_id = copy.deepcopy(self.joint_id)
#         kp = copy.deepcopy(self.kp)
#         kd = copy.deepcopy(self.kd)
#         min_torque = copy.deepcopy(self.min_torque)
#         max_torque = copy.deepcopy(self.max_torque)
#         latency = copy.deepcopy(self.latency)
#         actuator = self.__class__(joint_id=joint_id, kp=kp, kd=kd, min_torque=min_torque, max_torque=max_torque,
#                                   latency=latency)
#         memo[self] = actuator
#         return actuator
#
#
# class GearedActuator(JointActuator):
#     r"""Geared Actuator
#     """
#
#     def __init__(self, joint_id):
#         super(GearedActuator, self).__init__(joint_id)
#
#
# class DirectDriveActuator(JointActuator):
#     r"""Direct Drive Actuator
#     """
#
#     def __init__(self, joint_id):
#         super(DirectDriveActuator, self).__init__(joint_id)
#
#
# class SEA(JointActuator):
#     r"""Series Elastic Actuators
#
#     This actuator has multiple components including springs, gears, encoders, and an electric motors, resulting in
#     complex dynamics. Specifically, it is composed of an electric motor, a high gear ratio transmission, an elastic
#     element, and two rotary encoders to measure spring deflection and output position. [2]
#
#     References:
#         [1] "Series elastic actuators", Pratt et al., 1995
#         [2] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
#     """
#
#     def __init__(self, joint_id):
#         super(SEA, self).__init__(joint_id)
#
#
# class HydraulicActuator(JointActuator):
#     r"""Hydraulic Actuator
#     """
#
#     def __init__(self, joint_id):
#         super(HydraulicActuator, self).__init__(joint_id)
#
#
# class JointActuatorApproximator(JointActuator):
#     r"""Joint Actuator Approximator.
#
#     This is a joint actuator that uses a function approximator to output the torque values to be applied on the
#     actuator given for instance the joint positions. This function approximator has been trained on real data obtained
#     from the real actuator and can thus be a better approximation of the way the actual actuator works.
#     """
#
#     def __init__(self, joint_id, approximator=None):
#         super(JointActuatorApproximator, self).__init__(joint_id)
#         self.approximator = approximator
#
#     def __copy__(self):
#         """Return a shallow copy of the actuator. This can be overridden in the child class."""
#         return self.__class__(joint_id=self.joint_id, approximator=self.approximator)
#
#     def __deepcopy__(self, memo={}):
#         """Return a deep copy of the actuator. This can be overridden in the child class.
#
#         Args:
#             memo (dict): memo dictionary of objects already copied during the current copying pass
#         """
#         joint_id = copy.deepcopy(self.joint_id)
#         approximator = copy.deepcopy(self.approximator, memo)
#         actuator = self.__class__(joint_id=joint_id, approximator=approximator)
#         memo[self] = actuator
#         return actuator
#
#
# class ActuatorNet(JointActuatorApproximator):
#     r"""Actuator Neural Network
#
#     References:
#         [1] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
#     """
#
#     def __init__(self, joint_id, nn_model=None):
#         super(ActuatorNet, self).__init__(joint_id, approximator=nn_model)
#
#
# class CoupledJointActuatorApproximator(JointActuator):
#     r"""Coupled Joint Actuator Approximator
#
#     Multiple joint ids.
#     """
#     pass
#
#
# class CoupledActuatorNet(CoupledJointActuatorApproximator):
#     r"""Coupled Actuator Neural Network
#
#     """
#     pass
