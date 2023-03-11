#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the neural network actor-critic models which combine a policy and value function.

Note that actor-critic methods can share their parameters.

Dependencies:
- `pyrobolearn.policies`
- `pyrobolearn.values`
"""

import itertools
import torch

from pyrobolearn.policies import MLPPolicy
from pyrobolearn.values import MLPValue
from pyrobolearn.actorcritics import ActorCritic, SharedActorCritic

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MLPActorCritic(ActorCritic):
    r"""Multi-Layer Perceptron Actor Critic
    """

    def __init__(self, states, actions, hidden_units=(), activation='linear', last_activation=None,
                 dropout_prob=None, rate=1, preprocessors=None, postprocessors=None):
        """Initialize MLP policy.

        Args:
            states (State): 1D-states that is feed to the policy (the input dimensions will be inferred from the
                            states)
            actions (Action): 1D-actions outputted by the policy and will be applied in the simulator (the output
                              dimensions will be inferred from the actions)
            hidden_units (list/tuple of int): number of hidden units in the corresponding layer
            activation (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the
            last_activation (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout_prob (None, float, or list/tuple of float/None): dropout probability.
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the policy's output
        """
        policy = MLPPolicy(states, actions, hidden_units=hidden_units, activation=activation,
                           last_activation=last_activation, dropout=dropout_prob, rate=rate,
                           preprocessors=preprocessors, postprocessors=postprocessors)
        value = MLPValue(states, hidden_units=hidden_units, activation=activation,
                         last_activation=last_activation, dropout=dropout_prob,
                         preprocessors=preprocessors)
        super(MLPActorCritic, self).__init__(policy, value)


class MLPSharedActorCritic(SharedActorCritic):
    r"""Multi-Layer Perceptron Shared Actor Critic
    """
    pass

