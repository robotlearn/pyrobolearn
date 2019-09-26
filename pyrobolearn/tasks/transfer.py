#!/usr/bin/env python
"""Define the transfer learning task.
"""

from pyrobolearn.tasks.task import Task

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TLTask(object):
    r"""Transfer Learning task

    Transfer learning consists to transfer the knowledge acquired by the agent while solving a problem to another
    different but similar problem [1,2].

    References:
        - [1] "A Survey on Transfer Learning", Pan et al., 2010
        - [2]" Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self, domain_task, target_task):
        # super(TLTask, self).__init__(environment, policies)
        self.domain_task = domain_task
        self.target_task = target_task

    ##############
    # Properties #
    ##############

    @property
    def domain_task(self):
        """Return the domain task."""
        return self._domain_task

    @domain_task.setter
    def domain_task(self, task):
        """Set the domain task."""
        if not isinstance(task, Task):
            raise TypeError("Expecting the domain task to be an instance of Task, instead got {}".format(type(task)))
        self._domain_task = task

    @property
    def target_task(self):
        """Return the target task."""
        return self._target_task

    @target_task.setter
    def target_task(self, task):
        """Set the target task."""
        if not isinstance(task, Task):
            raise TypeError("Expecting the target task to be an instance of Task, instead got {}".format(type(task)))
        self._target_task = task

    @property
    def domain_environment(self):
        """Return the domain environment."""
        return self.domain_task.environment

    @property
    def target_environment(self):
        """Return the target environment."""
        return self.target_task.environment

    @property
    def domain_policies(self):
        """Return the domain policies."""
        return self.domain_task.policies

    @property
    def target_policies(self):
        """Return the target policies."""
        return self.target_task.policies

    ###########
    # Methods #
    ###########

    def train(self):
        pass

    def test(self):
        pass
