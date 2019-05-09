#!/usr/bin/env python
"""Defines the various actor-critic models which combine a policy and value function.

Note that actor-critic methods can share their parameters.

Dependencies:
- `pyrobolearn.policies`
- `pyrobolearn.values`
"""

import copy
import itertools
import torch

from pyrobolearn.policies import Policy
from pyrobolearn.values import ValueApproximator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ActorCritic(object):
    r"""Actor-Critic Methods

    These methods are between value-based and policy-based reinforcement learning approaches.
    Both a policy (called the actor) and a (state) value function approximator (called the critic) are being optimized.
    """

    def __init__(self, policy, value):
        """
        Initialize the actor critic.

        Args:
            policy (Policy): policy approximator
            value (Value): value function approximator
        """
        self.actor = policy
        self.critic = value

    ##############
    # Properties #
    ##############

    @property
    def actor(self):
        """Return the actor."""
        return self._actor

    @actor.setter
    def actor(self, actor):
        """Set the actor."""
        if not isinstance(actor, (Policy, torch.nn.Module)):
            raise TypeError("Expecting the actor to be an instance of 'Policy' or 'torch.nn.Module', "
                            "instead got {}".format(type(actor)))
        self._actor = actor

    @property
    def critic(self):
        """Return the critic."""
        return self._critic

    @critic.setter
    def critic(self, critic):
        """Set the critic."""
        if not isinstance(critic, (ValueApproximator, torch.nn.Module)):
            raise TypeError("Expecting the critic to be an instance of 'ValueApproximator' or 'torch.nn.Module', "
                            "instead got {}".format(type(critic)))
        self._critic = critic

    @property
    def states(self):
        """Return the states."""
        return self.actor.states

    @property
    def actions(self):
        """Return the actions."""
        return self.actor.actions

    ###########
    # Methods #
    ###########

    def parameters(self):
        """Return the parameters of first the actor then the critic."""
        generator = itertools.chain(self.actor.parameters(), self.critic.parameters())
        return generator

    def value(self, x):
        """Compute the value function."""
        return self.evaluate(x)

    def action(self, x):
        """Compute the action."""
        return self.act(x)

    def act(self, state=None, deterministic=True, to_numpy=True, return_logits=False, apply_action=True):
        """Evaluate the given input states.

        Args:
            state (State): current state
            deterministic (bool): True by default. It can only be set to False, if the policy is stochastic.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
            return_logits (bool): If True, in the case of discrete outputs, it will return the logits.
            apply_action (bool): If True, it will call and execute the action.

        Returns:
            (list of) np.array / torch.Tensor: action data
        """
        return self.actor.act(state, deterministic=deterministic, to_numpy=to_numpy, return_logits=return_logits,
                              apply_action=apply_action)

    def evaluate(self, state=None, to_numpy=False):
        """Evaluate the given input state.

        Args:
            state (None, State, (list of) np.array, (list of) torch.Tensor): state input data. If None, it will get
                the data from the inputs that were given at the initialization.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
        """
        return self.critic.evaluate(state, to_numpy=to_numpy)

    def act_and_evaluate(self, state=None, deterministic=True, to_numpy=True, return_logits=False, apply_action=True):
        """Act and evaluate the given input states.

        Args:
            state (State): current state
            deterministic (bool): True by default. It can only be set to False, if the policy is stochastic.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
            return_logits (bool): If True, in the case of discrete outputs, it will return the logits.
            apply_action (bool): If True, it will call and execute the action.

        Returns:
            (list of) np.array / torch.Tensor: action data
        """
        action = self.act(state, deterministic=deterministic, to_numpy=to_numpy, return_logits=return_logits,
                          apply_action=apply_action)
        value = self.evaluate(state, to_numpy=to_numpy)
        return action, value

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a string describing the object."""
        return self.__class__.__name__

    def __copy__(self):
        """Return a shallow copy of the approximator. This can be overridden in the child class."""
        return self.__class__(policy=self.actor, value=self.critic)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        policy = copy.deepcopy(self.actor, memo) if isinstance(self.actor, Policy) else copy.deepcopy(self.actor)
        value = copy.deepcopy(self.critic, memo) if isinstance(self.critic, ValueApproximator) \
            else copy.deepcopy(self.critic)
        actor_critic = self.__class__(policy=policy, value=value)

        memo[self] = actor_critic
        return actor_critic


class SharedActorCritic(object):
    r"""Shared Actor Critic

    This class described the actor critic with shared parameters.
    From the policy :math:`\pi_{\theta}(a_t | s_t)`
    """

    def __init__(self, states, actions, model=None, rate=1, preprocessors=None, postprocessors=None):
        # super(SharedActorCritic, self).__init__()
        # add a linear output node to the policy for the critic value
        # TODO
        pass
