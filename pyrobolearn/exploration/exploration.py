#!/usr/bin/env python
"""Provide the various exploration strategies.

Specifically, this file provides the main abstract `Exploration` class from which all the other exploration strategies
inherit from. Exploration is mainly useful in reinforcement learning, and can be carried out in the action or parameter
space.

References:
    [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 2018
"""

import torch

from pyrobolearn.policies import Policy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Exploration(object):
    r"""Exploration class

    Exploration strategy: it wraps the policy, and defines how the policy should explore in the environment.

    There are 2 main exploration strategies:
    1. exploration in the action space of the policy
    2. exploration in the parameter space of the policy

    Exploration can be, for instance, carried out based on the uncertainty of an action, a dynamic model (which
    predicts the next state given the current state and action), a value fct evaluated at the given state, and so on.
    This is also called "curiosity" and is linked to the notion of entropy (as the entropy is related to the notion
    of uncertainty).

    Exploration is crucial because it defines a stochastic policy, and thus a probability distribution. This in turn
    enables the use of probability concepts such as 'maximizing the likelihood or marginal likelihood'.
    Policy search algorithms currently only works with stochastic policies.
    """

    def __init__(self, policy):
        """
        Initialize the Exploration strategist.

        Args:
            policy (Policy): Policy to wrap.
        """
        self.policy = policy

    ##############
    # Properties #
    ##############

    @property
    def policy(self):
        """Return the policy instance"""
        return self._policy

    @policy.setter
    def policy(self, policy):
        """Set the policy instance."""
        if not isinstance(policy, Policy):
            raise TypeError("Expecting policy to be an instance of `Policy`, instead got: {}.".format(type(policy)))
        self._policy = policy

    ###########
    # Methods #
    ###########

    def reset(self):
        """Reset the exploration strategy, which can be useful at the beginning of an episode."""
        self.policy.reset()

    # def explore(self, *args, **kwargs):
    #     """Perform the exploratory action."""
    #     pass

    def act(self, state=None, deterministic=False, to_numpy=False, return_logits=False, apply_action=True):
        """Perform the action given the state.

        Args:
            state (State, list of np.array, list of torch.Tensor): current state.
            deterministic (bool): If True, it will return a deterministic action. If False, it will explore.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
            return_logits (bool): If True, in the case of discrete outputs, it will return the logits.
            apply_action (bool): If True, it will call and execute the action.

        Returns:
            (list of) np.array / torch.Tensor: action data
        """
        pass

    # def step(self, states):
    #     """Perform one step using the policy with the corresponding exploration strategy."""
    #     pass

    # def clear(self):
    #     """Called at the end of an episode to reset a policy or clear whatever has been done."""
    #     pass


# class ModelUncertaintyExploration(Exploration):
#     r"""Model Uncertainty Exploration
#
#     Exploration based on the uncertainty of a learned dynamic model.
#
#     References:
#     [1]
#     """
#
#     def __init__(self, policy):
#         super(ModelUncertaintyExploration, self).__init__(policy)
