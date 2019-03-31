#!/usr/bin/env python
r"""Provide the discrete epsilon-greedy action exploration.

The epsilon-greedy exploration strategy consists to explore in the discrete action space of policies. Specifically,
it selects the best action :math:`a*` with probability :math:`p = (1 - \epsilon)`, or another action
:math:`a \in A\{a*}` randomly (based on uniform distribution) with probability :math:`p = \frac{\epsilon}{|A|-1}`.
"""

import torch

from pyrobolearn.exploration.actions.discrete import DiscreteActionExploration


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class EpsilonGreedyActionExploration(DiscreteActionExploration):
    r"""Epsilon-greedy action exploration.

    The epsilon-greedy exploration strategy consists to explore in the discrete action space of policies. Specifically,
    it selects the best action :math:`a*` with probability :math:`p = (1 - \epsilon)`, or another action
    :math:`a \in A\{a*}` randomly (based on uniform distribution) with probability :math:`p = \frac{\epsilon}{|A|-1}`.
    """

    def __init__(self, policy, action):
        """
        Initialize the epsilon-greedy action exploration strategy.

        Args:
            policy (Policy): policy to wrap.
            action (Action): discrete actions.
        """
        super(EpsilonGreedyActionExploration, self).__init__(policy, action=action)
