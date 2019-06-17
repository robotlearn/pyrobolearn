#!/usr/bin/env python
"""Provide the Gym wrapper reward.
"""

import gym

from pyrobolearn.rewards.basic_rewards import FixedReward


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GymReward(FixedReward):
    r"""OpenAI Gym reward

    This provides a wrapper around the gym reward value. It is pretty much the same as the FixedReward. Its value is
    set outside the reward function because the `gym.Env.step(actions)` has to be called to compute the reward. There
    are however several problems with that:
    1. we need to provide the `actions` to that method
    2. it computes the next observations, and other stuffs while we just want the reward value.
    Thus, it is left to the user to set the reward value to the `GymReward` instance.
    """

    def __init__(self, value, range=None):
        """
        Initialize the Gym reward.

        Args:
            value (float, int): current value of the reward.
            range (tuple of float/int, gym.Env): range for the reward function.
        """
        # set the range
        if range is not None:
            if isinstance(range, gym.Env):
                self.range = range.reward_range
            else:
                self.range = range

        super(GymReward, self).__init__(value, range=range)
