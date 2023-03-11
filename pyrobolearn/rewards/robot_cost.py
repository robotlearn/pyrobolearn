#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define some basic robot costs used in reinforcement learning and optimization.

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

from abc import ABCMeta
import numpy as np

import pyrobolearn as prl
from pyrobolearn.robots.robot import Robot
from pyrobolearn.rewards.cost import Cost


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotCost(Cost):
    r"""Robot reward (abstract).

    Abstract reward class that accepts as input the state and/or action which must depends on a robotic platform.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state=None, update_state=False):
        """
        Initialize the Robot reward.

        Args:
            state (State, Robot): robot state.
            update_state (bool): if we should call the state and update its value.
        """
        if isinstance(state, prl.states.State):
            super(RobotCost, self).__init__(state)
        else:
            super(RobotCost, self).__init__()
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
        if isinstance(state, Robot):  # if robot, instantiate state class with robot as param.
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
    def normalize(x):
        """
        Normalize the given vector.
        """
        if np.allclose(x, 0):
            return x
        return x / np.linalg.norm(x)


class DriftCost(RobotCost):
    """Drift cost.

    Calculates the drift of a moving robot wrt a specified direction.
    """

    def __init__(self, state, direction=(1, 0, 0), normalize=False, update_state=False):  # TODO: use direction.
        """
        Initialize the drift cost.

        Args:
            state (BasePositionState, Robot): robot or base position state.
            direction (np.array[float[3]], None): forward direction vector. If None, it will take the initial forward
              vector.
            normalize (bool): if we should normalize the direction vector.
            update_state (bool): if we should call the state and update its value.
        """
        super(DriftCost, self).__init__(state=state, update_state=update_state)

        # check given base position state
        self.state, self.update_state = self._check_state(state, prl.states.BasePositionState,
                                                          update_state=self.update_state)

        # if no direction specified, take the body forward vector
        if direction is None:
            self.direction = self.state.body.forward_vector
        else:
            self.direction = np.array(direction)

        # normalize the direction vector if specified
        if normalize:
            self.direction = self.normalize(self.direction)

        # remember current position
        self.prev_pos = np.copy(self.state.data[0])
        self.value = 0

    def _compute(self):
        """Compute the difference vector between the current and previous position (i.e. ~ velocity vector), and
        compute the dot product between this velocity vector and the direction vector."""
        if self.update_state:
            self.state()
        curr_pos = self.state.data[0]
        velocity = curr_pos - self.prev_pos
        self.value = -np.abs(velocity[1])
        self.prev_pos = np.copy(curr_pos)
        return self.value


class ShakeCost(RobotCost):
    """Shake cost.

    Calculates the shaking cost of a moving robot wrt a specified direction.
    """

    def __init__(self, state, direction=(1, 0, 0), normalize=False, update_state=False):
        """
        Initialize the shake cost.

        Args:
            state (BasePositionState, Robot): robot or base position state.
            direction (np.array[float[3]], None): forward direction vector. If None, it will take the initial forward
              vector.
            normalize (bool): if we should normalize the direction vector.
            update_state (bool): if we should call the state and update its value.
        """
        super(ShakeCost, self).__init__(state=state, update_state=update_state)

        # check given base position state
        self.state, self.update_state = self._check_state(state, prl.states.BasePositionState,
                                                          update_state=self.update_state)

        # if no direction specified, take the body forward vector
        if direction is None:
            self.direction = self.state.body.forward_vector
        else:
            self.direction = np.array(direction)

        # normalize the direction vector if specified
        if normalize:
            self.direction = self.normalize(self.direction)

        # remember current position
        self.prev_pos = np.copy(self.state.data[0])
        self.value = 0

    def _compute(self):
        """Compute the difference vector between the current and previous position (i.e. ~ velocity vector), and
        compute the dot product between this velocity vector and the direction vector."""
        if self.update_state:
            self.state()
        curr_pos = self.state.data[0]
        velocity = curr_pos - self.prev_pos
        self.value = -np.abs(velocity[2])
        self.prev_pos = np.copy(curr_pos)
        return self.value
