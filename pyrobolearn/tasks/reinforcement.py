#!/usr/bin/env python
"""Define the reinforcement learning task.
"""

from pyrobolearn.tasks.task import Task

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RLTask(Task):
    r"""Reinforcement Learning Task

    Reinforcement learning consists for an agent to learn to perform a certain task by maximizing the expected total
    reward [1,2].

    References:
        [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 2018
        [2] "A Survey on Policy Search for Robotics", Deisenroth et al., 2013
    """

    def __init__(self, environment, policies):
        super(RLTask, self).__init__(environment, policies)
