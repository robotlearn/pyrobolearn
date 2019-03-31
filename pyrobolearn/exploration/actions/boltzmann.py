#!/usr/bin/env python
r"""Provide the discrete Boltzmann action exploration.

The Boltzmann exploration strategy consists to explore in the discrete action space of policies by using a
categorical distribution on action probabilities. The probabilities are often computed using a softmax function
which maps the values of the logits to correct probabilities (i.e. each probability is between 0 and 1, and the
sum of them sums to 1). That is, compared to epsilon-greedy which selects another action uniformly, Boltzmann
exploration selects an action based on its weight which is outputted by the policy.
"""


import torch

from pyrobolearn.distributions.modules import CategoricalModule, IdentityModule
from pyrobolearn.exploration.actions.discrete import DiscreteActionExploration


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BoltzmannActionExploration(DiscreteActionExploration):
    r"""Boltzmann action exploration.

    The Boltzmann exploration strategy consists to explore in the discrete action space of policies by using a
    categorical distribution on action probabilities. The probabilities are often computed using a softmax function
    which maps the values of the logits to correct probabilities (i.e. each probability is between 0 and 1, and the
    sum of them sums to 1). That is, compared to epsilon-greedy which selects another action uniformly, Boltzmann
    exploration selects an action based on its weight which is outputted by the policy.
    """

    def __init__(self, policy, action):
        """
        Initialize the Boltzmann action exploration strategy.

        Args:
            policy (Policy): policy to wrap.
            action (Action): discrete actions.
        """
        super(BoltzmannActionExploration, self).__init__(policy, action=action)

        # create Categorical module
        logits = IdentityModule()
        self._module = CategoricalModule(logits=logits)
