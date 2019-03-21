#!/usr/bin/env python
"""Define various basic policies.

Define the various basic policies such as the random policy, linear policy, policies based on value functions, etc.
"""

import numpy as np

from pyrobolearn.policies.policy import Policy
from pyrobolearn.approximators import LinearApproximator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RandomPolicy(Policy):
    """Random policy
    """

    def __init__(self, states, actions, seed=None, *args, **kwargs):
        super(RandomPolicy, self).__init__(states, actions, *args, **kwargs)
        if seed is not None:
            np.random.seed(seed)

    def act(self, state=None, deterministic=False, to_numpy=True):
        # get the space of each action
        spaces = self.actions.space

        # sample from each space
        action_data = [space.sample() for space in spaces]

        # set the data for each action
        self.actions.data = action_data

        return self.actions

    def sample(self, state):
        return self.act(state)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)


class LinearPolicy(Policy):
    """Linear Policy
    """

    def __init__(self, states, actions, *args, **kwargs):
        model = LinearApproximator(states, actions)
        super(LinearPolicy, self).__init__(states, actions, model, *args, **kwargs)

    def act(self, state, deterministic=True, to_numpy=True):
        return self.model.predict(state, to_numpy=to_numpy)

    def sample(self, state):
        pass


class PolicyFromValue(Policy):
    r"""Policy From Value Function Approximator

    .. math::

        a = argmax_a Q^\pi(s,a)

    .. seealso::

        * `value.py`
    """

    def __init__(self, states, actions, *args, **kwargs):
        super(PolicyFromValue, self).__init__(states, actions, *args, **kwargs)

    def act(self, state=None, deterministic=True, to_numpy=True):
        pass

    def sample(self, state):
        pass
