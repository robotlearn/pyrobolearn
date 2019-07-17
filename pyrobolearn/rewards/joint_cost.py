#!/usr/bin/env python
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
    def _check_state(state, cls, update_state=False):
        # check given state
        if isinstance(state, prl.robots.Robot):  # if robot, instantiate state class with robot as param.
            state = cls(robot=state)
            update_state = True
        if not isinstance(state, cls):  # if not an instance of the given state, class, raise error
            raise TypeError("Expecting the given 'state' to be an instance of `Robot` or `" + cls.__name__ + "`, "
                            "but instead got: {}".format(type(state)))
        return state, update_state

    @staticmethod
    def _check_target_state(state, target_state, cls, update_state=False):
        # check given target state
        if target_state is None:  # if the target is None, initialize it zero
            target_state = np.zeros(state.total_size())
        if isinstance(target_state, (int, float, np.ndarray)):  # if target is a np.array/float/int, create FixedState
            target_state = prl.states.FixedState(value=target_state)
            update_state = True
        elif isinstance(target_state, prl.robots.Robot):  # if robot, instantiate state class with robot as param.
            target_state = cls(robot=target_state)
            update_state = True
        elif not isinstance(target_state, cls):  # if not an instance of the given state class, raise error
            raise TypeError("Expecting the given 'target_state' to be None, a np.array, or an instance of "
                            "`Robot` or `" + cls.__name__ + "`, but instead got: {}".format(type(target_state)))
        return target_state, update_state


# class JointPositionErrorCost(JointCost):
#     r"""Joint Position Error Cost
#
#     Return the joint position error as defined in [1] as :math:`d(\hat{\phi}, \phi) \in [0, \pi]` where :math:`d(.,.)`
#     is the minimum angle difference between two angles, and :math:`\hat{\phi}` and :math:`\phi` are the target and
#     current angles.
#
#     References:
#         - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
#     """
#
#     def __init__(self, joint_state, target_joint_state, update_state=False):
#         """
#         Initialize the joint position error cost.
#
#         Args:
#             joint_state (JointPositionState):
#             target_joint_state (JointPositionState):
#             update_state (bool): if True it will update the given states before computing the cost.
#         """
#         super(JointPositionErrorCost, self).__init__(update_state=update_state)
#         self.state = joint_state
#         self.target_state = target_joint_state
#
#     def _compute(self):
#         return - min_angle_difference(self.state.data[0], self.target_state.data[0])


class JointPositionCost(JointCost):
    r"""Joint Position Cost

    Return the cost such that measures the L2 norm between the current joint positions and the target joint positions:
    :math:`||d(q_{target},q)||^2` where :math:`d(\cdot, \cdot) \in [-\pi, \pi]` is the minimum distance between two
    angles.

    References:
        - [1] OpenAI Gym
        - [2] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, state, target_state, update_state=False):
        r"""
        Initialize the joint position cost.

        Args:
            state (JointPositionState, Robot): joint position state.
            target_state (JointPositionState, np.array[N], None): target joint position state. If None, it will be set
                to 0.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        super(JointPositionCost, self).__init__(update_state)

        # check given joint position state
        self.q, self.update_state = self._check_state(state, prl.states.JointPositionState,
                                                      update_state=self.update_state)

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

    Return the cost due to the joint velocities: :math:`|| \dot{q}_{target} - \dot{q} ||^2`, where
    :math:`\dot{q}_{target}` can be set to zero if wished.
    """

    def __init__(self, state, target_state=None, update_state=False):
        """
        Initialize the joint velocity cost.

        Args:
            state (JointVelocityState, Robot): joint velocity state.
            target_state (JointVelocityState, np.array[N], Robot, None): target joint velocity state. If None, it
                will be set to 0.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        super(JointVelocityCost, self).__init__(update_state)

        # check given joint velocity state
        self.dq, self.update_state = self._check_state(state, prl.states.JointVelocityState,
                                                       update_state=self.update_state)

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

    Return the joint acceleration cost defined notably in [1] as :math:`cost = || \ddot{q}_{target} - \ddot{q} ||^2`,
    where :math:`\ddot{q}_{target}` can be set to zero if wished.

    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
    """

    def __init__(self, state, target_state=None, update_state=False):
        """
        Initialize the joint acceleration cost.

        Args:
            state (JointAccelerationState, Robot): joint acceleration state.
            target_state (JointAccelerationState, np.array[N], Robot, None): target joint acceleration state. If None,
                it will be set to 0.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        super(JointAccelerationCost, self).__init__(update_state=update_state)

        # check given joint acceleration state
        self.ddq, self.update_state = self._check_state(state, prl.states.JointAccelerationState,
                                                        update_state=self.update_state)

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

    Return the cost due to the joint torques; :math:`|| \tau_{target} - \tau ||^2`, where :math:`\tau_{target}` can
    be set to zero if wished.
    """

    def __init__(self, state, target_state=None, update_state=False):
        """
        Initialize the joint torque cost.

        Args:
            state (JointForceTorqueState, Robot): joint torque state.
            target_state (JointForceTorqueState, np.array[N], Robot, None): target joint torque state. If None, it
                will be set to 0.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        super(JointTorqueCost, self).__init__(update_state)

        # check given joint torque state
        self.tau, self.update_state = self._check_state(state, prl.states.JointForceTorqueState,
                                                        update_state=self.update_state)

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
    r"""Joint Power Consumption Cost

    Return the joint power consumption cost, where the power is computed as the torque times the velocity.
    """

    def __init__(self, state, joint_ids=None, update_state=False):
        """
        Initialize the Joint Power Consumption cost.

        Args:
            state (Robot, State): robot instance, or the state. The state must contains the `JointForceTorqueState`
                and the `JointVelocityState`. Note that if they are multiple torque or velocity states, it will look
                for the first instance.
            joint_ids (None, int, list of int): joint ids. This used if `torque` is a `Robot` instance.
            update_state (bool): if True it will update the given states before computing the cost.
        """
        self.update_state = update_state

        # Check the state
        # if the given state is a robot, create the torque and velocity states
        if isinstance(state, prl.robots.Robot):
            torque = prl.states.JointForceTorqueState(state, joint_ids=joint_ids)
            velocity = prl.states.JointVelocityState(state, joint_ids=joint_ids)
            self.update_state = True
            # state = torque + velocity

        # elif the given state is a composite state, look for the torque and velocity states.
        else:

            if isinstance(state, prl.states.State):
                state = [state]

            # if the given state is a list of states, check each one of them by looking for the torque/velocity state
            if isinstance(state, (list, tuple)):
                # for each state, check if it is a torque, velocity or composite state
                torque, velocity = None, None
                for s in state:
                    if isinstance(s, prl.states.JointForceTorqueState):
                        if torque is None:
                            torque = s
                    elif isinstance(s, prl.states.JointVelocityState):
                        if velocity is None:
                            velocity = s
                    elif isinstance(s, prl.states.State):
                        if torque is None:
                            torque = s.lookfor(prl.states.JointForceTorqueState)
                        if velocity is None:
                            velocity = s.lookfor(prl.states.JointVelocityState)
                    else:
                        raise TypeError("Expecting the state to be an instance of `State` or `Robot`, or a list of "
                                        "`State`, instead got: {}".format(type(s)))
                    # if we have found the states, get out of the loop
                    if torque is not None and velocity is not None:
                        break

                # check that we have the torque and velocity states
                if torque is None:
                    raise ValueError("Didn't find a `JointForceTorqueState` instance in the given states.")
                if velocity is None:
                    raise ValueError("Didn't find a `JointVelocityState` instance in the given states.")
            else:
                raise TypeError("Expecting the state to be an instance of `State` or `Robot`, instead got: "
                                "{}".format(type(state)))

        super(JointPowerCost, self).__init__()
        self.tau = torque
        self.vel = velocity
        self.update_state = update_state

    def _compute(self):
        """Compute and return the cost value."""
        if self.update_state:
            self.tau()
            self.vel()
        return - np.sum((self.tau.data[0] * self.vel.data[0])**2)


# class JointSpeedCost(Cost):
#     r"""Joint Speed Cost
#
#     Return the joint speed cost as computed in [1].
#
#     .. math:: \text{cost} = || \max(\dot{q}_{max} - |\dot{q}|, 0) ||^2
#
#     References:
#         - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
#     """
#
#     def __init__(self, state, max_joint_speed=None):
#         """
#         Initialize the joint speed state.
#
#         Args:
#             state: joint velocity state.
#             max_joint_speed:
#         """
#         super(JointSpeedCost, self).__init__()
#         self.dq = state
#         self.dq_max = max_joint_speed
#         if max_joint_speed is None:
#             self.dq_max = state.max
#
#     def _compute(self):
#         """Compute and return the cost value."""
#         return - np.sum(np.maximum(self.dq_max - np.abs(self.dq.data[0]), 0)**2)
