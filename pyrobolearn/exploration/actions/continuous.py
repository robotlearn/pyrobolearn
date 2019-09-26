# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the continuous action exploration strategies.

Action exploration is used in reinforcement learning algorithms and describe how the policy explores in the
environment. Note that the policy is the only (probability) function that we have control over; we do not control the
dynamic transition (probability) function nor the reward function. In action exploration, a probability distribution
is put on the outputted action space :math:`a_t \sim \pi_{\theta}(.|s_t)`. There are mainly two categories:
exploration for discrete actions (which uses discrete probability distribution) and exploration for continuous action
(which uses continuous probability distribution).

Note that action exploration is a step-based exploration strategy where at each time step of an episode, an action is
sampled based on the specified distribution.

Action exploration might change a bit the structure of the policy while running.

References:
    [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 2018
"""

from abc import ABCMeta

from pyrobolearn.exploration.actions.action_exploration import ActionExploration

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ContinuousActionExploration(ActionExploration):
    r"""Continuous action exploration.

    Continuous action exploration strategies use continuous probability distributions on the (continuous) actions.
    """

    __metaclass__ = ABCMeta

    def __init__(self, policy, action):
        """
        Initialize the continuous action exploration strategy.

        Args:
            policy (Policy): policy to wrap.
            action (action): continuous action.
        """
        super(ContinuousActionExploration, self).__init__(policy, action=action)
        if not self.action.is_continuous():
            raise ValueError("Expecting the given action to be continuous.")
