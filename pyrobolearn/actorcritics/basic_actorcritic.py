# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Defines basic actor-critic models (such as linear models) which combine a policy and value function.

Dependencies:
- `pyrobolearn.policies`
- `pyrobolearn.values`
"""

import itertools
import torch

from pyrobolearn.policies import LinearPolicy
from pyrobolearn.values import LinearValue
from pyrobolearn.actorcritics import ActorCritic, SharedActorCritic

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinearActorCritic(ActorCritic):
    r"""Linear Actor Critic
    """

    def __init__(self, states, actions, rate=1, preprocessors=None, postprocessors=None):
        """Initialize MLP policy.

        Args:
            states (State): 1D-states that is feed to the policy (the input dimensions will be inferred from the
                            states)
            actions (Action): 1D-actions outputted by the policy and will be applied in the simulator (the output
                              dimensions will be inferred from the actions)
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the policy's output
        """
        policy = LinearPolicy(states, actions, rate=rate, preprocessors=preprocessors, postprocessors=postprocessors)
        value = LinearValue(states, preprocessors=preprocessors)
        super(LinearActorCritic, self).__init__(policy, value)


class LinearSharedActorCritic(SharedActorCritic):
    r"""Linear Shared Actor Critic
    """
    pass

