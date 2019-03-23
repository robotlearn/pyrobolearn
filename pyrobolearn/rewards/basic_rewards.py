#!/usr/bin/env python
"""Define some basic rewards used in reinforcement learning and optimization.

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

import numpy as np

# from objective import Objective
from pyrobolearn.robots.robot import Robot
import pyrobolearn.states as states
import pyrobolearn.actions as actions
from pyrobolearn.rewards.reward import Reward

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FixedReward(Reward):
    r"""Fixed reward.

    This is a dummy class which always returns a fixed reward. This is fixed initially.
    """

    def __init__(self, value):
        super(FixedReward, self).__init__()
        if not isinstance(value, (int, float)):
            raise TypeError("Expecting a number")
        self.value = value

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, str(self.value))

    def compute(self):
        return self.value


class FunctionalReward(Reward):
    r"""Functional reward.

    This is a reward class which calls a given function/class to compute the reward.
    """
    def __init__(self, function):
        super(FunctionalReward, self).__init__()
        self.function = function

    def __repr__(self):
        return self.function.__name__

    def compute(self):
        return self.function()


class ForwardProgressReward(Reward):
    r"""Forward progress reward

    Compute the forward progress based on a forward direction, a previous and current positions.
    """

    def __init__(self, state, direction=(1, 0, 0), normalize=False, update_state=False):
        """
        Initialize the Forward Progress Reward.

        Args:
            state (BasePositionState, PositionState, Robot): robot or base position state.
            direction (np.float[3], None): forward direction vector. If None, it will take the initial forward vector.
            normalize (bool): if we should normalize the direction vector.
            update_state (bool): if we should call the state and update its value.
        """
        # check state argument
        self.update_state = update_state
        if isinstance(state, Robot):
            state = states.BasePositionState(state)
            self.update_state = True
        elif not isinstance(state, (states.BasePositionState, states.PositionState)):
            raise TypeError("Expecting the state to be an instance of `BasePositionState`, `PositionState`, or `Robot`"
                            ", instead got: {}".format(type(state)))
        super(ForwardProgressReward, self).__init__(state=state)

        # if no direction specified, take the body forward vector
        if direction is None:
            self.direction = state.body.forward_vector
        else:
            self.direction = np.array(direction)

        # normalize the direction vector if specified
        if normalize:
            self.direction = self.normalize(self.direction)

        # remember current position
        self.prev_pos = np.copy(self.state.data[0])
        self.value = 0

    @staticmethod
    def normalize(x):
        """
        Normalize the given vector.
        """
        if np.allclose(x, 0):
            return x
        return x / np.linalg.norm(x)

    def compute(self):
        """Compute the difference vector between the current and previous position (i.e. ~ velocity vector), and
        compute the dot product between this velocity vector and the direction vector."""
        if self.update_state:
            self.state()
        curr_pos = self.state.data[0]
        velocity = curr_pos - self.prev_pos
        self.value = self.direction.dot(velocity)
        self.prev_pos = np.copy(curr_pos)
        return self.value


class DirectiveReward(Reward):
    r"""Directive Reward

    Provide reward if the vector state is in the specified direction. Specifically, it computes the dot product
    between the state vector and the specified direction.

    If normalize, the reward is between -1 and 1.
    """

    def __init__(self, state, direction=(1, 0, 0), normalize=True):
        super(DirectiveReward, self).__init__(state=state)

        self.normalize = normalize
        if self.normalize:
            self.direction = self.norm(np.array(direction))

        # TODO uncomment
        # if not isinstance(state, (PositionState, BasePositionState)):
        #     raise ValueError("Expecting state to be a PositionState or BasePositionState")
        self.value = 0

    @staticmethod
    def norm(x):
        """
        Normalize the given vector.
        """
        if np.allclose(x, 0):
            return x
        return x / np.linalg.norm(x)

    def compute(self):
        pos = self.state.data[0]
        if self.normalize:
            pos = self.norm(pos)
        self.value = self.direction.dot(pos)
        return self.value


class L2SimilarityReward(Reward):
    """
    Compute the square of the L2 norm between two vectors.
    """

    def __init__(self):
        super(L2SimilarityReward, self).__init__()

    def value(self, vector1, vector2):
        return np.dot(vector1, vector2)


class ImitationReward(Reward):

    def __init__(self, human, robot):
        super(ImitationReward, self).__init__()
        self.human = human  # instance of HumanKinematic class
        self.robot = robot  # instance of Robot class

    def compute(self):
        # check
        pass
