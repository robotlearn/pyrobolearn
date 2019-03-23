#!/usr/bin/env python
"""Provide the Gym wrapper reward.
"""

from pyrobolearn.rewards.basic_rewards import FixedReward

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GymReward(FixedReward):
    r"""OpenAI Gym reward

    This provides a wrapper around the gym reward value. It is the same as the FixedReward.
    """
    pass
