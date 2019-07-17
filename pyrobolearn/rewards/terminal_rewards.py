#!/usr/bin/env python
"""Define terminal rewards used in RL.
"""

import numpy as np

from pyrobolearn.rewards.reward import Reward
from pyrobolearn.rewards.basic_rewards import FixedReward
from pyrobolearn.terminal_conditions.terminal_condition import TerminalCondition

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TerminalReward(Reward):
    r"""Terminal reward.

    This computes the provided subreward until the terminal condition is not fulfilled. Once it has achieved it, it
    computes the final given reward function. This reward is useful specially for sparse rewards that only returns a
    value once the goal has been achieved (e.g. games).
    """

    def __init__(self, terminal_condition, subreward, final_reward):
        r"""
        Terminal reward.

        Args:
            terminal_condition (TerminalCondition): terminal condition.
            subreward (Reward, float, int): sub reward that is called until the terminal condition is not fulfilled.
            final_reward (Reward, float, int): final reward that is called when the terminal condition has been reached.
        """
        super(TerminalReward, self).__init__()

        # set the attributes
        self.terminal_condition = terminal_condition
        self.subreward = subreward
        self.final_reward = final_reward

    ##############
    # Properties #
    ##############

    @property
    def terminal_condition(self):
        """Return the terminal condition instance."""
        return self._terminal_condition

    @terminal_condition.setter
    def terminal_condition(self, condition):
        """Set the terminal condition instance."""
        if not isinstance(condition, TerminalCondition):
            raise TypeError("Expecting the given 'terminal_condition' to be an instance of `TerminalCondition`, "
                            "instead got: {}".format(type(condition)))
        self._terminal_condition = condition

    @property
    def subreward(self):
        """Return the sub-reward instance."""
        return self._reward

    @subreward.setter
    def subreward(self, reward):
        """Set the sub-reward instance."""
        if isinstance(reward, (int, float)):
            reward = FixedReward(value=reward)
        elif not isinstance(reward, Reward):
            raise TypeError("Expecting the given 'subreward' to be an instance of `Reward`, instead got: "
                            "{}".format(type(reward)))
        self._reward = reward

    @property
    def final_reward(self):
        """Return the final reward instance."""
        return self._final_reward

    @final_reward.setter
    def final_reward(self, reward):
        """Set the final reward instance."""
        if isinstance(reward, (int, float)):
            reward = FixedReward(value=reward)
        elif not isinstance(reward, Reward):
            raise TypeError("Expecting the given 'final_reward' to be an instance of `Reward`, instead got: "
                            "{}".format(type(reward)))
        self._final_reward = reward

    ###########
    # Methods #
    ###########

    def _compute(self):
        """Compute the terminal reward."""
        done = self.terminal_condition()
        if done:
            return self.final_reward()
        return self.subreward()
