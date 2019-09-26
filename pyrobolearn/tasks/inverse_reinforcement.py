# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the inverse reinforcement learning task.
"""

from pyrobolearn.tasks.imitation import ILTask

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class IRLTask(ILTask):
    r"""Inverse Reinforcement Learning Task.

    The goal of the inverse reinforcement learning task is to approximate the reward function from demonstrations
    by the user.
    """

    def __init__(self, environment, policies, reward_approximator, interface=None, recorder=None):
        """Initialize the inverse reinforcement learning task.

        Args:
            environment (Env): environment of the task (which contains the world)
            policies (Policy): rl to be trained by the task
            reward_approximator (Approximator): reward function approximator
            interface (Interface): input/output interface that allows to interact with the world.
            recorder (Recorder): if the interface doesn't have a recorder, it can be supplemented here.
                If the interface doesn't have a recorder, and it is not specified, it will create a recorder that
                record the states and actions (inferred from the rl).
        """
        super(IRLTask, self).__init__(environment, policies, interface, recorder)
