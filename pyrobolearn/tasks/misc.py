# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the miscellaneous tasks.
"""

from pyrobolearn.tasks import Task, ILTask, RLTask

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CombinedTask(Task):

    def __init__(self, tasks):
        super(CombinedTask, self).__init__(simulator, env)

    def __rshift__(self, other):
        """
        Add another task in sequence.

        Args:
            other (Task):
        """
        pass


class WalkingTask(RLTask):

    def __init__(self, simulator, robot=None, policy=None):

        # define world
        world = World(simulator)
        world.set_gravity()

        # define reward
        rewards = [Reward()]

        # define env
        env = Env(simulator, world, policies, rewards)

        super(WalkingTask, self).__init__(env)
