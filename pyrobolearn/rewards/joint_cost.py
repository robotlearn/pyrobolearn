#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the costs used on joint states / actions.
"""

from abc import ABCMeta
import numpy as np

import pyrobolearn as prl
from pyrobolearn.rewards.cost import Cost
from pyrobolearn.utils.transformation import min_angle_difference


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointCost(Cost):

    r"""(Abstract) Joint Cost."""
    __metaclass__ = ABCMeta

    def __init__(self, update_state=False):
        """
        Initialize the abstract joint cost.

        Args:
            update_state (bool): if True it will update the given states before computing the cost.
        """
        super(JointCost, self).__init__()
        self.update_state = update_state

    @staticmethod
    def _check_state(state, cls, update_state=False, **kwargs):
        """
        Check that the given state is an instance of the given class. If not, check if it can be constructed.

        Args:
            state (Robot, State, list/tuple[State]): the state or robot instance that we have to check.
            cls (State class): the state class that the state should belong to.
            update_state (bool): if the state should be updated or not by default.
            **kwargs (dict): dictionary of arguments passed to the `cls` class  if the state is a `Robot` instance.

        Returns:
            State: an instance of the specified class `cls`.
            bool: if the state should be updated or not.
        """
        # check given state
        if isinstance(state, prl.robots.Robot):  # if robot, instantiate state class with robot as param.
            state = cls(robot=state, **kwargs)
            update_state = True
        if not isinstance(state, cls):  # if not an instance of the given state class, look for it (the first instance)
            if isinstance(state, prl.states.State):
                state = state.lookfor(cls)
            elif isinstance(state, (tuple, list)):
                for s in state:
                    if isinstance(s, cls):
                        state = s
                    elif isinstance(s, prl.states.State):
                        state = s.lookfor(cls)

                    if state is not None:
                        break
            else:
                raise TypeError("Expecting the given 'state' to be an instance of `Robot`, `{}`, `State` or a list of "
                                "`State`, but instead got: {}".format(cls.__name__, type(state)))

            if state is None:
                raise ValueError("Couldn't find the specified state class `{}` in the given "
                                 "state.".format(cls.__name__))
        return state, update_state

    @staticmethod
    def _check_target_state(state, target_state, cls, update_state=False, **kwargs):
        """
        Check that the given target state is an instance of the given target state class. If not, check if it can be
        constructed from it.

        Args:
            state (Robot, State, list/tuple[State]): the state associated to the given target state. This is used if
                the target state is an int, float, or np.ndarray.
            target_state (None, int, float, np.array, Robot, State, list/tuple[State]): the target state or robot
                instance that we have to check.
            cls (State class): the state class that the state should belong to.
            update_state (bool): if the state should be updated or not by default.
            **kwargs (dict): dictionary of arguments passed to the `cls` class  if the target state is a `Robot`
                instance.

        Returns:
            State: an instance of the specified class `cls`.
            bool: if the state should be updated or not.
        """
        # check given target state
        if target_state is None:  # if the target is None, initialize it zero
            target_state = np.zeros(state.total_size())
        if isinstance(target_state, (int, float, np.ndarray)):  # if target is a np.array/float/int, create FixedState
            # TODO: check shape
            target_state = prl.states.FixedState(value=target_state)
            update_state = True
        elif isinstance(target_state, prl.robots.Robot):  # if robot, instantiate state class with robot as param.
            target_state = cls(robot=target_state, **kwargs)
            update_state = True
        elif not isinstance(target_state, cls):  # if not an instance of the given state class, look for it
            if isinstance(target_state, prl.states.State):
                target_state = target_state.lookfor(cls)
            elif isinstance(target_state, (tuple, list)):
                for s in target_state:
                    if isinstance(s, cls):
                        target_state = s
                    elif isinstance(s, prl.states.State):
                        target_state = s.lookfor(cls)
                    if target_state is not None:
                        break
            else:
                raise TypeError("Expecting the given 'target_state' to be None, a np.array, an instance of `Robot`, "
                                "`{}`, `State`, or a list of `State`, but instead got: "
                                "{}".format(cls.__name__, type(target_state)))

            if target_state is None:
                raise ValueError("Couldn't find the specified target state class `{}` in the given "
                                 "target_state.".format(cls.__name__))

        return target_state, update_state


class JointAngleDifferenceCost(JointCost):
    r"""Joint Angle Difference Cost

    Return the cost such that measures the L2 norm between the current joint positions and the target joint positions:

    .. math:: ||d(q_{target},q)||^2,

    where :math:`d(\cdot, \cdot) \in [-\pi, \pi]` is the minimum distance between two angles as described in [1,2].

    References:
        - [1] OpenAI Gym
        - [2] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, state, target_state, joint_ids=None, update_state=False):
        r"""
        Initialize the joint position cost.

        Args:
            state (JointPositionState, Robot): joint position state, or robot instance.
            target_state (JointPositionState, np.array[float[N]], None): target joint position state. If None, it will
                be set to 0.
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.
        """
        super(JointAngleDifferenceCost, self).__init__(update_state)

        # check given joint position state
        self.q, self.update_state = self._check_state(state, prl.states.JointPositionState,
                                                      update_state=self.update_state, joint_ids=joint_ids)

        # check target joint position state
        self.q_target, self.update_target_state = self._check_target_state(self.q, target_state,
                                                                           prl.states.JointPositionState,
                                                                           self.update_state)

        if self.q.total_size() != self.q_target.total_size():
            raise ValueError("The given state and target_state do not have the same size: "
                             "{} != {}".format(self.q.total_size(), self.q_target.total_size()))

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_state:
            self.q()
        if self.update_target_state:
            self.q_target()
        return - np.sum(min_angle_difference(self.q.data[0], self.q_target.data[0])**2)


class JointPositionCost(JointCost):
    r"""Joint Position Cost

    Return the cost such that measures the L2 norm between the current joint positions and the target joint positions:

    .. math:: ||q_{target} - q||^2`.
    """

    def __init__(self, state, target_state, joint_ids=None, update_state=False):
        r"""
        Initialize the joint position cost.

        Args:
            state (JointPositionState, Robot): joint position state, or robot instance.
            target_state (JointPositionState, np.array[float[N]], None): target joint position state. If None, it will
                be set to 0.
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.
        """
        super(JointPositionCost, self).__init__(update_state)

        # check given joint position state
        self.q, self.update_state = self._check_state(state, prl.states.JointPositionState,
                                                      update_state=self.update_state, joint_ids=joint_ids)

        # check target joint position state
        self.q_target, self.update_target_state = self._check_target_state(self.q, target_state,
                                                                           prl.states.JointPositionState,
                                                                           self.update_state)

        if self.q.total_size() != self.q_target.total_size():
            raise ValueError("The given state and target_state do not have the same size: "
                             "{} != {}".format(self.q.total_size(), self.q_target.total_size()))

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_state:
            self.q()
        if self.update_target_state:
            self.q_target()
        return - np.sum(min_angle_difference(self.q.data[0], self.q_target.data[0])**2)


class JointVelocityCost(JointCost):
    r"""Joint Velocity Cost

    Return the cost due to the joint velocities given by:

    .. math:: c = || \dot{q}_{target} - \dot{q} ||^2

    where :math:`\dot{q}_{target}` can be set to zero if wished.
    """

    def __init__(self, state, target_state=None, joint_ids=None, update_state=False):
        """
        Initialize the joint velocity cost.

        Args:
            state (JointVelocityState, Robot): joint velocity state, or robot instance.
            target_state (JointVelocityState, np.array[float[N]], Robot, None): target joint velocity state. If None,
                it will be set to 0.
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.
        """
        super(JointVelocityCost, self).__init__(update_state)

        # check given joint velocity state
        self.dq, self.update_state = self._check_state(state, prl.states.JointVelocityState,
                                                       update_state=self.update_state, joint_ids=joint_ids)

        # check target joint velocity state
        self.dq_target, self.update_target_state = self._check_target_state(self.dq, target_state,
                                                                            prl.states.JointVelocityState,
                                                                            self.update_state)

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_state:
            self.dq()
        if self.update_target_state:
            self.dq_target()
        return - np.sum((self.dq_target.data[0] - self.dq.data[0])**2)


class JointAccelerationCost(JointCost):
    r"""Joint Acceleration Cost

    Return the joint acceleration cost defined as:

    .. math:: c = || \ddot{q}_{target} - \ddot{q} ||^2

    where :math:`\ddot{q}_{target}` can be set to zero if wished.
    """

    def __init__(self, state, target_state=None, joint_ids=None, update_state=False):
        """
        Initialize the joint acceleration cost.

        Args:
            state (JointAccelerationState, Robot): joint acceleration state.
            target_state (JointAccelerationState, np.array[float[N]], Robot, None): target joint acceleration state.
                If None, it will be set to 0.
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.
        """
        super(JointAccelerationCost, self).__init__(update_state=update_state)

        # check given joint acceleration state
        self.ddq, self.update_state = self._check_state(state, prl.states.JointAccelerationState,
                                                        update_state=self.update_state, joint_ids=joint_ids)

        # check target joint acceleration state
        self.ddq_target, self.update_target_state = self._check_target_state(self.ddq, target_state,
                                                                             prl.states.JointAccelerationState,
                                                                             self.update_state)

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_state:
            self.ddq()
        if self.update_target_state:
            self.ddq_target()
        return - np.sum((self.ddq_target.data[0] - self.ddq.data[0])**2)


class JointTorqueCost(JointCost):
    r"""Torque Cost

    Return the cost due to the joint torques given by:

    .. math:: c = || \tau_{target} - \tau ||^2

    where :math:`\tau_{target}` can be set to zero if wished.
    """

    def __init__(self, state, target_state=None, joint_ids=None, update_state=False):
        """
        Initialize the joint torque cost.

        Args:
            state (JointForceTorqueState, Robot): joint torque state.
            target_state (JointForceTorqueState, np.array[float[N]], Robot, None): target joint torque state. If None,
                it will be set to 0.
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.
        """
        super(JointTorqueCost, self).__init__(update_state)

        # check given joint torque state
        self.tau, self.update_state = self._check_state(state, prl.states.JointForceTorqueState,
                                                        update_state=self.update_state, joint_ids=joint_ids)

        # check target joint torque state
        self.tau_target, self.update_target_state = self._check_target_state(self.tau, target_state,
                                                                             prl.states.JointForceTorqueState,
                                                                             self.update_state)

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_state:
            self.tau()
        if self.update_target_state:
            self.tau_target()
        return - np.sum((self.tau_target.data[0] - self.tau.data[0])**2)


class JointPowerCost(JointCost):
    r"""Joint Power Cost

    Return the joint power cost given by:

    .. math:: c = ||\tau \cdot \dot{q}||^2

    where :math:`\tau \in \mathcal{R}^N` are the torques, and :math:`\dot{q} \in \mathcal{R}^N` are the joint
    velocities.
    """

    def __init__(self, state, joint_ids=None, update_state=False):
        """
        Initialize the Joint Power Consumption cost.

        Args:
            state (Robot, State): robot instance, or the state. The state must contains the `JointForceTorqueState`
                and the `JointVelocityState`. Note that if they are multiple torque or velocity states, it will look
                for the first instance.
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.
        """
        super(JointPowerCost, self).__init__(update_state)

        # check given joint torque and velocity state
        self.tau, self.update_torque_state = self._check_state(state, prl.states.JointForceTorqueState,
                                                               update_state=self.update_state, joint_ids=joint_ids)
        self.vel, self.update_velocity_state = self._check_state(state, prl.states.JointVelocityState,
                                                                 update_state=self.update_state, joint_ids=joint_ids)

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_torque_state:
            self.tau()
        if self.update_velocity_state:
            self.vel()
        return - np.sum((self.tau.data[0] * self.vel.data[0])**2)


class JointPowerConsumptionCost(JointCost):
    r"""Joint Power Consumption Cost

    Return the joint power consumption cost given by [1]:

    .. math:: c = \sum_i^N max(\tau_i \dot{q}_i, 0)

    where :math:`\tau \in \mathcal{R}^N` are the torques, and :math:`\dot{q} \in \mathcal{R}^N` are the joint
    velocities.

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, state, joint_ids=None, update_state=False):
        """
        Initialize the Joint Power Consumption cost.

        Args:
            state (Robot, State): robot instance, or the state. The state must contains the `JointForceTorqueState`
                and the `JointVelocityState`. Note that if they are multiple torque or velocity states, it will look
                for the first instance.
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.
        """
        super(JointPowerConsumptionCost, self).__init__(update_state)

        # check given joint torque and velocity state
        self.tau, self.update_torque_state = self._check_state(state, prl.states.JointForceTorqueState,
                                                               update_state=self.update_state, joint_ids=joint_ids)
        self.vel, self.update_velocity_state = self._check_state(state, prl.states.JointVelocityState,
                                                                 update_state=self.update_state, joint_ids=joint_ids)

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_torque_state:
            self.tau()
        if self.update_velocity_state:
            self.vel()
        return - np.sum(np.maximum(self.tau.data[0] * self.vel.data[0], 0))


class JointSpeedLimitCost(JointCost):
    r"""Joint Speed Limit Cost

    Return the joint speed cost as computed in [1].

    .. math:: c = || \max(\dot{q}_{max} - |\dot{q}|, 0) ||^2

    where :math:``

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, state, max_joint_speed=None, joint_ids=None, update_state=False):
        """
        Initialize the joint speed state.

        Args:
            state (JointVelocityState, Robot): joint velocity state, or robot instance.
            max_joint_speed (int, float, np.array[float[N]], None):
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.

        """
        super(JointSpeedLimitCost, self).__init__(update_state)
        self.dq = state
        self.dq_max = max_joint_speed
        if max_joint_speed is None:
            self.dq_max = state.max

    def _compute(self):
        """Compute and return the cost value."""
        return - np.sum(np.maximum(self.dq_max - np.abs(self.dq.data[0]), 0)**2)


class JointEnergyCost(JointCost):
    r"""Joint Energy Cost

    Return the joint energy cost given by:

    .. math:: c = | \tau \cdot \dot{q}| * dt

    where :math:`\tau \in \mathcal{R}^N` are the torques, :math:`\dot{q} \in \mathcal{R}^N` are the joint velocities,
    and :math:`dt` is the simulation time step.
    """

    def __init__(self, state, dt, joint_ids=None, update_state=False):
        """
        Initialize the joint energy cost.

        Args:
            state (Robot, State): robot instance, or the state. The state must contains the `JointForceTorqueState`
                and the `JointVelocityState`. Note that if they are multiple torque or velocity states, it will look
                for the first instance.
            dt (float): simulation time.
            joint_ids (None, int, list[int]): joint ids. This used if `state` is a `Robot` instance.
            update_state (bool): if True, it will update the given states before computing the cost.
        """
        super(JointEnergyCost, self).__init__(update_state)

        # check given joint torque and velocity state
        self.tau, self.update_torque_state = self._check_state(state, prl.states.JointForceTorqueState,
                                                               update_state=self.update_state, joint_ids=joint_ids)
        self.vel, self.update_velocity_state = self._check_state(state, prl.states.JointVelocityState,
                                                                 update_state=self.update_state, joint_ids=joint_ids)

        self.dt = dt

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_torque_state:
            self.tau()
        if self.update_velocity_state:
            self.vel()
        return np.abs(np.dot(self.tau.data[0], self.vel.data[0])) * self.dt
