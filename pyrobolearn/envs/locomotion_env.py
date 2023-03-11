#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the locomotion environment.

Define the environment to perform a locomotion task; it mainly defines the reward function.
"""

from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.envs.env import Env
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.states import State
from pyrobolearn.policies import Policy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LocomotionEnv(Env):  # TODO
    r"""Locomotion environment

    Define a simple environment for a locomotion task.
    """

    def __init__(self, simulator, states, world=None, terminal_condition='default'):
        r"""Initialize the locomotion environment

        Args:
            simulator (Simulator): simulator
            states (State, Policy): states that environment must return. If the policy is given, it takes the states
                that are given as input to the policy.
            world (World, None): the world for the locomotion task. If None, it creates a basic world.
            terminal_condition (str, default): the terminating condition criterion to stop if the policy failed
                the task.
        """

        # check parameters
        if not isinstance(simulator, Simulator):
            raise TypeError("Expecting the `simulator` parameter to be an instance of Simulator.")
        if not isinstance(states, (State, Policy)):
            raise TypeError("Expecting the `states` parameter to be an instance of State or Policy.")

        # define world
        if world is None:
            world = BasicWorld(simulator)

        # get states/actions if policy
        actions = None
        if isinstance(states, Policy):
            actions = states.actions
            states = states.states

        # define reward based on states/actions
        rewards = None

        # define terminating condition criterion
        if terminal_condition == 'default':
            terminal_condition = None
        terminal_condition = None

        super(LocomotionEnv, self).__init__(world, states, rewards=rewards,
                                            terminal_conditions=terminal_condition, extra_info=None)

