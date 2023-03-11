#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
__license__ = "Apache License 2.0"
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
        self.action_data = None
        self.action_distribution = None

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
        raise NotImplementedError

    def predict(self, state=None, deterministic=True, to_numpy=False, return_logits=True):
        """Predict the action given the state.

        This does not set the action data in the action instances, nor apply the actions in the simulator. Instead,
        it gets the state data, preprocess it, predict using the actions using the inner model, then post-process
        the actions, and return the resulting action data.

        Args:
            state (State): current state
            deterministic (bool): True by default. It can only be set to False, if the policy is stochastic.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
            return_logits (bool): If True, in the case of discrete outputs, it will return the logits.

        Returns:
            (list of) torch.Tensor: action data
        """
        # if deterministic outcome, i.e. we don't explore we just predict the actions using the policy
        if deterministic:
            action_data = self.policy.predict(state=state, deterministic=True, to_numpy=to_numpy,
                                              return_logits=return_logits)
            action_distribution = None

        # if we should explore
        else:
            # get the state data
            state_data = self.policy.get_state_data(state=state)

            # pre-process the state data
            state_data = self.policy.preprocess(state_data)

            # predict the actions using the inner model
            action_data = self.policy.inner_predict(state_data, deterministic=True, to_numpy=False,
                                                    return_logits=True, set_output_data=False)

            # exploration phase

            # if exploration is a combination of multiple exploration
            if self._explorations:
                # explore for each action
                actions = [explorer.explore(action_data) for explorer in self.explorations]
                action_data, action_distribution = [a[0] for a in actions], [a[1] for a in actions]

            else:  # there is only one action
                action_data, action_distribution = self.explore(action_data)

            # post-process the action data
            action_data = self.policy.postprocess(action_data)

        return action_data, action_distribution

    def act(self,  state=None, deterministic=False, to_numpy=False, return_logits=False, apply_action=True):
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
        # if deterministic outcome, i.e. we don't explore just run the policy
        if deterministic:
            self.action_data = self.policy.act(state, deterministic=True, to_numpy=to_numpy,
                                               return_logits=return_logits, apply_action=apply_action)

            self.action_distribution = None

        # if we should explore
        else:
            # if we should predict
            if (self.policy.cnt % self.policy.rate) == 0:
                # get the state data
                state_data = self.policy.get_state_data(state=state)

                # pre-process the state data
                state_data = self.policy.preprocess(state_data)

                # predict the actions using the inner model
                action_data = self.policy.inner_predict(state_data, deterministic=True, to_numpy=False,
                                                        return_logits=True, set_output_data=False)

                # exploration phase

                # if exploration is a combination of multiple exploration
                if self._explorations:
                    # explore for each action
                    actions = [explorer.explore(action_data) for explorer in self.explorations]
                    action_data, action_distribution = [a[0] for a in actions], [a[1] for a in actions]

                else:  # there is only one action
                    action_data, action_distribution = self.explore(action_data)

                # post-process the action data
                action_data = self.policy.postprocess(action_data)

                # set the action data
                self.action_data = self.policy.set_action_data(action_data, to_numpy=to_numpy,
                                                               return_logits=return_logits)
                self.action_distribution = action_distribution

            # apply action
            if apply_action:
                self.policy.actions()

            # increment policy's tick counter
            self.policy.cnt = self.policy.cnt + 1

        return self.action_data, self.action_distribution

    def mode(self):
        """Return the mode of the distributions."""
        if self.action_distribution is not None:
            if isinstance(self.action_distribution, list):
                return [dist.mode() for dist in self.action_distribution]
            return self.action_distribution.mode()
        return self.action_data

    def sample(self):
        """Sample an action from the distribution."""
        if self.action_distribution is None:
            raise NotImplementedError("The action distribution has not been set.")
        if isinstance(self.action_distribution, list):
            return [dist.sample() for dist in self.action_distribution]
        return self.action_distribution.sample()

    def action_log_prob(self, actions):
        """Return the log probability evaluated at the given actions."""
        return self.action_distribution.log_probs(actions)

    def action_prob(self, actions):
        """Return the probability evaluated at the given actions."""
        return torch.exp(self.action_distribution.log_probs(actions))

    def entropy(self):
        """Return the entropy of the distribution."""
        return self.action_distribution.entropy().mean()
