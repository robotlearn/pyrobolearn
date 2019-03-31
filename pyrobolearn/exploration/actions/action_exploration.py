#!/usr/bin/env python
r"""Provide the action exploration strategies.

Action exploration is used in reinforcement learning algorithms and describe how the policy explores in the
environment. Note that the policy is the only (probability) function that we have control over; we do not control the
dynamic transition (probability) function nor the reward function. In action exploration, a probability distribution
is put on the outputted action space :math:`a_t \sim \pi_{\theta}(\cdot|s_t)`. There are mainly two categories:
exploration for discrete actions (which uses discrete probability distribution) and exploration for continuous action
(which uses continuous probability distribution).

Note that action exploration is a step-based exploration strategy where at each time step of an episode, an action is
sampled based on the specified distribution.

Action exploration might change a bit the structure of the policy while running.

References:
    [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 2018
"""

import collections
import torch

from pyrobolearn.actions import Action
import pyrobolearn as prl
from pyrobolearn.exploration import Exploration


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ActionExploration(Exploration):
    r"""Action Exploration (aka Step-based RL)

    Explore in the action space of the policy. At each time step, the action is sampled from the policy based
    on the given distribution (hence the name 'step-based' RL).

    Assume a policy is denoted by :math:`\pi_{\theta}(a|s)` which maps states :math:`s` to action :math`a`, and
    is parametrized by :math:`\theta` which are the parameters that can be learned/optimized/trained. In action
    space exploration, the actions :math:`a` are sampled from a probability distribution, such as a Gaussian
    distribution such that :math:`a \sim \mathcal{N}(\pi_{\theta}(a|s), \Sigma)`.

    This way of exploring is notably used in:
    - several reinforcement learning algorithms (REINFORCE, TRPO, PPO, etc)
    """

    def __init__(self, policy, action=None, explorations=None):
        """
        Initialize the action exploration strategy.

        Args:
            policy (Policy): Policy to wrap.
            action (Action, None): action space to explore.
            explorations ((list of) ActionExploration, None): (list of) action exploration strategies. Each action
                exploration strategy describes how the policy explores in the specified (discrete or continuous)
                action space.
        """
        super(ActionExploration, self).__init__(policy)

        # check action
        if action is not None and not isinstance(action, Action):
            raise TypeError("Expecting the given action to be an instance of `Action`, instead got: "
                            "{}".format(type(action)))
        self._action = action

        # check exploration strategies

        # if no exploration strategies set
        if explorations is None:
            explorations = []

            # if no action has been defined
            if self.action is None:
                # go over each action of the policy, and add the corresponding exploration strategy based on the
                # action type
                for action in self.policy.actions:
                    if action.is_discrete():
                        prl.logger.debug('creating a Boltzmann action exploration with action of size: %d',
                                         action.space[0].n)
                        exploration = prl.exploration.actions.BoltzmannActionExploration(self.policy, action)
                    elif action.is_continuous():
                        exploration = prl.exploration.actions.GaussianActionExploration(self.policy, action)
                    else:
                        raise ValueError("Expecting an action to be discrete or continuous.")
                    explorations.append(exploration)

        else:
            # check if an action has already been set
            if self.action is not None:
                raise ValueError("Expecting to be given an action or a list of explorations, not both.")

            # transform the explorations to a list if not iterable
            if not isinstance(explorations, collections.Iterable):
                explorations = [explorations]

            # check the length of exploration strategies and the number of actions in the policy
            if len(explorations) != len(self.policy.actions):
                raise ValueError("Expecting the number of actions (={}) to be the same as the number of exploration "
                                 "strategy (={}).".format(len(self.policy.actions), len(explorations)))

            actions = set([exploration.action for exploration in explorations])

            # check that each exploration is set for each action
            for action in self.policy.actions:
                if action not in actions:
                    raise ValueError("Expecting for each action in the policy to have its corresponding exploration "
                                     "strategy. The following action was not found in the exploration strategies: "
                                     "{}".format(action))

        # set the exploration strategies
        self._explorations = explorations

    ##############
    # Properties #
    ##############

    @property
    def action(self):
        """Return the specific action to explore. This might return None."""
        return self._action

    @property
    def explorations(self):
        """Return the list of exploration strategies."""
        return self._explorations

    ###########
    # Methods #
    ###########

    def act(self,  state=None, deterministic=True, to_numpy=True, return_logits=False, apply_action=True):
        r"""
        Act/Explore in the environment given the states.

        Args:
            state (State): current state
            deterministic (bool): True by default. It can only be set to False, if the policy is stochastic.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
            return_logits (bool): If True, in the case of discrete outputs, it will return the logits.
            apply_action (bool): If True, it will call and execute the action.

        Returns:
            (list of) torch.Tensor: action(s)
            (list of) torch.distributions.Distribution: policy distribution(s) :math:`\pi_{\theta}(\cdot | s)`
        """
        # TODO: finish to clean
        print(state)
        actions = self.policy.act(state, to_numpy=False, return_logits=True)

        if deterministic:
            return actions, None

        # From deterministic output into stochastic outputs
        # print("Actions before dist: {}").format(actions.train_data)
        print("Exploration strategy - actions: {}".format(actions))
        self.dist = self.distribution(actions)
        actions = self.dist.sample()
        print("Exploration strategy - sampled action: {}".format(actions))
        if isinstance(actions, torch.Tensor):
            if actions.requires_grad:
                self.policy.actions.data = actions.detach().numpy()
            else:
                self.policy.actions.data = actions.numpy()
        else:
            self.policy.actions.data = actions

        return actions, self.dist

    def mode(self):
        """Return the mode of the distributions."""
        actions = self.dist.mode()
        return actions

    def sample(self):
        """Sample an action from the distribution."""
        actions = self.dist.sample()
        return actions

    def action_log_prob(self, actions):
        """Return the log probability evaluated at the given actions."""
        return self.dist.log_probs(actions)

    def action_prob(self, actions):
        """Return the probability evaluated at the given actions."""
        return torch.exp(self.dist.log_probs(actions))

    def entropy(self):
        """Return the entropy of the distribution."""
        return self.dist.entropy().mean()
