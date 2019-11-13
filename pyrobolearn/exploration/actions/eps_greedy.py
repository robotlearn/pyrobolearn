#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the discrete epsilon-greedy action exploration.

The epsilon-greedy exploration strategy consists to explore in the discrete action space of policies. Specifically,
it selects the best action :math:`a*` with probability :math:`p = (1 - \epsilon)`, or another action
:math:`a \in A\{a*}` randomly (based on uniform distribution) with probability :math:`p = \frac{\epsilon}{|A|-1}`.
"""

import torch

from pyrobolearn.distributions.categorical import Categorical
from pyrobolearn.exploration.actions.discrete import DiscreteActionExploration


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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

    def __init__(self, policy, action, epsilon=0.1):
        """
        Initialize the epsilon-greedy action exploration strategy.

        Args:
            policy (Policy): policy to wrap.
            action (Action): discrete actions.
            epsilon (float): epsilon probability.
        """
        super(EpsilonGreedyActionExploration, self).__init__(policy, action=action)
        self.epsilon = epsilon

    def explore(self, outputs):
        r"""
        Explore in the action space. Note that this does not run the policy; it is assumed that it has been called
        outside.

        Args:
            outputs (torch.Tensor): action outputs (=logits) returned by the model.

        Returns:
            torch.Tensor: action
            torch.distributions.Distribution: distribution on the action :math:`\pi_{\theta}(.|s)`
        """
        idx = torch.argmax(outputs, dim=-1)  # shape = (N,) or (1,)
        probs = self.epsilon/(outputs.size()[-1] - 1) * torch.ones_like(outputs)  # shape = (N, D) or (D,)
        if len(probs.shape) > 1:  # multiple data
            probs[range(len(outputs)), idx] = (1. - self.epsilon)
        else:
            probs[idx] = (1. - self.epsilon)
        distribution = Categorical(probs=probs)   # shape = (N, D)
        action = distribution.sample((1,))        # shape = (1, N)
        if len(action.shape) > 1:  # multiple data
            action = action.view(-1, 1)           # shape = (N, 1) otherwise shape = (1,)
        return action, distribution
