# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the abstract control environment from which all the other control environments inherit from.
"""

from pyrobolearn.envs.env import Env


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ControlEnv(Env):
    r"""Control Environment (abstract)

    This is the abstract control environment from which all control environments inherit from.
    """

    def __init__(self, world, states, rewards=None, terminal_conditions=None, initial_state_generators=None,
                 physics_randomizers=None, extra_info=None, actions=None):
        """
        Initialize the control environment.

        Args:
            world (World): world of the environment. The world contains all the objects (including robots), and has
                access to the simulator.
            states ((list of) State): states that are returned by the environment at each time step.
            rewards (None, Reward): The rewards can be None when for instance we are in an imitation learning setting,
                instead of a reinforcement learning one. If None, only the state is returned by the environment.
            terminal_conditions (None, callable, TerminalCondition, list of TerminalCondition): A callable function or
                object that check if the policy has failed or succeeded the task.
            initial_state_generators (None, StateGenerator, list of StateGenerator): state generators which are used
                when resetting the environment to generate the initial states.
            physics_randomizers (None, PhysicsRandomizer, list of PhysicsRandomizer): physics randomizers. This will be
                called each time you reset the environment.
            extra_info (None, callable): Extra info returned by the environment at each time step.
            actions ((list of) Action): actions that are given to the environment. Note that this is not used here in
                the current environment as it should be the policy that performs the action. This is useful when
                creating policies after the environment (that is, the policy can uses the environment's states and
                actions).
        """
        super(ControlEnv, self).__init__(world=world, states=states, rewards=rewards,
                                         terminal_conditions=terminal_conditions,
                                         initial_state_generators=initial_state_generators,
                                         physics_randomizers=physics_randomizers, extra_info=extra_info,
                                         actions=actions)
