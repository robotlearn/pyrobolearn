#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the `Scheduler` class which deals on how to run multiple tasks in a sequential or parallel way.

A user can create its own task/scenario independently of the rest. Once a task is done, which is known when
the environment returns `done=True`, the next task is loaded into memory. When sequencing tasks, if a policy
is defined for Task1 and no policy is defined for Task2, it will use the same policy. If another policy is
defined it will sequence this one with the previous policy. The same rationale applies for the world, robots,
and so on.
"""

from pyrobolearn.tasks.task import Task


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Scheduler(object):
    r"""Scheduler class.

    The scheduler is responsible on how to run multiple tasks might it be in a sequential or parallel manner.
    Some tasks can only be run after certain conditions are met, thus in our framework, we represent the scheduler
    as a directed graph.

    If the user has only one task, this class is not useful. The scheduler can be dynamically built as the agent(s)
    progress(es) in the various tasks.
    """

    def __init__(self, tasks):
        self.tasks = tasks
        self.graph = {}

    def add_task(self, task, previous_tasks=None, next_tasks=None):
        pass


class NodeTask(object):

    def __init__(self, task):
        self.task = task

        # references to the parent/children nodes
        self.parents = []
        self.children = []

    def is_done(self):
        return self.task.is_done()
