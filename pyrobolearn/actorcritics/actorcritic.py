#!/usr/bin/env python
"""Defines the various actor-critic models which combine a policy and value function.

Note that actor-critic methods can share their parameters.

Dependencies:
- `pyrobolearn.policies`
- `pyrobolearn.values`
"""

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

    def act(self, states=None, deterministic=True):
        """Evaluate the given input states."""
        return self.actor.act(states, deterministic=deterministic)

    def evaluate(self, states=None):
        """Evaluate the given input states."""
        return self.critic.compute(states)

    def act_and_evaluate(self, states=None):
        """Act and evaluate the given input states."""
        return self.act(states), self.evaluate(states)


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
