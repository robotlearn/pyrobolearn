#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the curriculum learning task.

This type of tasks starts by training agents on simple tasks/environments first and then increase the difficulty
level of tasks progressively.

This can be achieved in three different ways; discretely, continuously, or a mixture of both.
For instance, in a locomotion task where a robot has to learn how to walk, we can progressively increase the
difficulty level by:
- having different discrete type of worlds; starting from a world with a flat floor, passing to smooth terrains with
ups and downs, to a world filled with inanimate obstacles, to finally a lively world with different moving agents.
- modifying in a continuous manner the reward/cost function landscape. This can be achieved by increasing / decreasing
the coefficient values of some rewards/costs as the number of episodes/iterations progresses.

References:
    [1] "Curriculum learning", Bengio et al., 2009
"""

import collections.abc

from pyrobolearn.worlds import World
from pyrobolearn.tasks.task import Task, Env

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CLTask(Task):
    r"""Curriculum Learning Task

    This type of tasks starts by training agents on simple tasks/environments first and then increase the difficulty
    level of tasks progressively.

    This can be achieved in three different ways; discretely, continuously, or a mixture of both.
    For instance, in a locomotion task where a robot has to learn how to walk, we can progressively increase the
    difficulty level by:
    - having different discrete type of worlds; starting from a world with a flat floor, passing to smooth terrains
    with ups and downs, to a world filled with inanimate obstacles, to finally a lively world with different moving
    agents.
    - modifying in a continuous manner the reward/cost function landscape. This can be achieved by increasing /
    decreasing the coefficient values of some rewards/costs as the number of episodes/iterations progresses.

    References:
        [1] "Curriculum learning", Bengio et al., 2009
    """

    def __init__(self, environments, policies):
        self.environments = environments
        first_environment = self.environments[0]
        super(CLTask, self).__init__(first_environment, policies)

    ##############
    # Properties #
    ##############

    @property
    def environments(self):
        """Return the list of environments ordered by complexities."""
        return self._environments

    @environments.setter
    def environments(self, environments):
        """Set the environments sorted by increasing order of difficulty level."""
        # TODO use an ordered dictionary
        if isinstance(environments, Env):
            environments = [environments]
        elif isinstance(environments, collections.abc.Iterable):
            if len(environments) < 1:
                raise ValueError("Expecting the list of environments to at least a length of one.")
            for i, env in enumerate(environments):
                if not isinstance(env, Env):
                    raise TypeError("Expecting 'environments' to be a list of environments, instead the {} item "
                                    "in the list has a type of {}".format(i, type(env)))
        else:
            raise TypeError("Expecting 'environments' to be a list of environments, instead got "
                            "{}".format(type(environments)))

        self._environments = environments

    @property
    def num_environments(self):
        """Return the number of environments."""
        return len(self.environments)

    ###########
    # Methods #
    ###########

    def add_environment(self, environment, index=-1):
        """Add an environment at the specified index."""
        if not isinstance(environment, Env):
            raise TypeError("Expecting the given 'environment' to be an instance of Environment, instead got "
                            "{}".format(type(environment)))
        self.environments.insert(index, environment)

    def remove_environment(self, index=-1):
        """Remove and return the specified environment from the ordered list of environments."""
        return self.environments.pop(index)
