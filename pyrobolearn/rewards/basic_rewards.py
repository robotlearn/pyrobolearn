#!/usr/bin/env python
"""Define some basic rewards used in reinforcement learning and optimization.

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

import numpy as np

from pyrobolearn.robots.robot import Robot
import pyrobolearn.states as states
from pyrobolearn.rewards.reward import Reward


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FixedReward(Reward):
    r"""Fixed reward.

    This is a dummy class which always returns a fixed reward. This is fixed initially.
    """

    def __init__(self, value, range=None):
        """
        Initialize the fixed reward.

        Args:
            value (int, float): initial value.
            range (None, tuple[float/int]): A tuple corresponding to the min and max possible rewards. By default,
                it is [value, value]. The initial value must be included in the given range.
        """
        super(FixedReward, self).__init__()
        if not isinstance(value, (int, float)):
            raise TypeError("Expecting a number")
        self.value = value
        self.range = (value, value) if range is None or (isinstance(range, (tuple, list)) and len(range) == 0) \
            else range

        if value < self.range[0] or value > self.range[1]:
            raise ValueError("The given value (={}) is not in the specified range = {}".format(value, self.range))

    def __str__(self):
        """Return a string describing the reward."""
        return '%s(%s, range=%s)' % (self.__class__.__name__, str(self.value), str(self.range))

    def _compute(self):
        return self.value


# class FunctionalReward(Reward):
#     r"""Functional reward.
#
#     This is a reward class which calls a given function/class to compute the reward.
#     """
#     def __init__(self, function):
#         super(FunctionalReward, self).__init__()
#         self.function = function
#         # TODO: range
#
#     def __repr__(self):
#         return self.function.__name__
#
#     def _compute(self):
#         return self.function()


class DirectiveReward(Reward):
    r"""Directive Reward

    Provide reward if the vector state is in the specified direction. Specifically, it computes the dot product
    between the state vector and the specified direction.

    If normalize, the reward is between -1 and 1.
    """

    def __init__(self, state, direction=(1, 0, 0), normalize=True, range=None):
        super(DirectiveReward, self).__init__(state=state, range=range)

        self.normalize = normalize
        if self.normalize:
            self.direction = self.norm(np.array(direction))

        # TODO uncomment
        # if not isinstance(state, (PositionState, BasePositionState)):
        #     raise ValueError("Expecting state to be a PositionState or BasePositionState")
        self.value = 0

        if self.normalize:
            self.range = (-1., 1.)

    @staticmethod
    def norm(x):
        """Normalize the given vector."""
        if np.allclose(x, 0):
            return x
        return x / np.linalg.norm(x)

    def _compute(self):
        """Compute the reward."""
        pos = self.state.data[0]
        if self.normalize:
            pos = self.norm(pos)
        self.value = self.direction.dot(pos)
        return self.value


# class L2SimilarityReward(Reward):
#     """
#     Compute the square of the L2 norm between two vectors.
#     """
#
#     def __init__(self, vector1, vector2, range=None):
#         super(L2SimilarityReward, self).__init__(range=range)
#         self.vector1 = vector1
#         self.vector2 = vector2
#
#     def _compute(self):
#         return np.linalg.norm(self.vector1, self.vector2)
#
#
# class ImitationReward(Reward):
#
#     def __init__(self, human, robot, range=None):
#         super(ImitationReward, self).__init__(range=range)
#         self.human = human  # instance of HumanKinematic class
#         self.robot = robot  # instance of Robot class
#
#     def _compute(self):
#         # check
#         pass


# Test
if __name__ == '__main__':
    from pyrobolearn.rewards import cos

    reward = 2*FixedReward(1, range=(-1., 1.)) - FixedReward(3)**2 - 10
    reward += FixedReward(2)
    print(reward)
    print("\n2*FixedReward(1, range=(-1., 1.)) + FixedReward(3)**2 - 10 + FixedReward(2) = {}".format(reward()))
    print("Is an instance of Reward? {}".format(isinstance(reward, Reward)))
    print("Inner rewards: {}".format(reward.rewards))
    print("Range of reward: {}".format(reward.range))

    reward = FixedReward(-10, range=(-20, 20))
    print("\nInitial fixed reward: {}".format(reward()))
    print("Range: {}".format(reward.range))
    reward = abs(reward)
    print("abs(reward) = {}".format(reward()))
    print("range(abs(reward)) = {}".format(reward.range))
    print("FixedReward(10) == FixedReward(10)? {}".format(FixedReward(10) == FixedReward(10)))

    reward = FixedReward(2) + FixedReward(1)
    print("\nreward = FixedReward(2) + FixedReward(1) = {}".format(reward()))
    reward = cos(reward)
    print("cos(reward) = {}".format(reward()))
    # print(type(reward))
    # print(reward.rewards)

    reward = FixedReward(1, range=(-1., 1.))
    print("\nreward = FixedReward(1, range=(-1.,1.)")
    reward.value = -2.
    print("reward.value = {}".format(reward.value))
    print("Calling reward (it should automatically be clipped): {}".format(reward()))
