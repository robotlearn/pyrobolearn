#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Describes the various `Estimators` / `Returns` used in reinforcement learning.

- Estimators are evaluated on trajectories and directly saved in the rollout storage unit (i.e. RolloutStorage).
- Returns are estimated on batches of transition tuples (s, a, s', r, d) and saved in the `batch.current` attribute.

In both case, the computed data is saved in the storage / batch unit, which can then be later used by the loss for
instance.

Dependencies:
- `pyrobolearn.storages`
"""

import torch

from pyrobolearn.storages import RolloutStorage
from pyrobolearn.values import Value, QValue


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Estimator(object):
    r"""Estimator / Return

    Estimator / Return used for gradient based algorithms. The gradient is given by:

    .. math::

       g = \mathbb{E}_{s_{0:T}, a_{0:T}}[ \sum_{t=0}^{T} \psi_t \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) ]

    where :math:`\psi_t` is the associated return estimator, and the expectation is over the "states and actions
    sampled sequentially from the dynamics model :math:`P(s_{t+1} | s_t, a_t)` and policy :math:`\pi(a_t | s_t)`,
    respectively" [1].

    Note that a discount factor can be used for :math:`\psi_t`. If the discount factor :math:`\gamma` is smaller
    than 1, then it will reduces the variance but at the cost of introducing a bias.

    Reference:
        - [1] "High-Dimensional Continuous Control using Generalized Advantage Estimation", Schulman et al., 2016
    """

    def __init__(self, storage, gamma=1.):
        """
        Initialize the estimator / return function.

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
        """
        self.gamma = gamma
        self.storage = storage

    ##############
    # Properties #
    ##############

    @property
    def gamma(self):
        """Return the discount factor"""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """Set the discount factor"""
        if gamma > 1.:
            gamma = 1.
        elif gamma < 0.:
            gamma = 0.

        self._gamma = gamma

    @property
    def storage(self):
        """Return the storage instance"""
        return self._storage

    @storage.setter
    def storage(self, storage):
        """Set the storage instance"""
        if not isinstance(storage, RolloutStorage):
            raise TypeError("Expecting the given storage to be an instance of `RolloutStorage`, instead got: {} with "
                            "type {}".format(storage, type(storage)))
        self._storage = storage

    @property
    def returns(self):
        """Return the returns tensor from the rollout storage."""
        return self.storage[self]

    @property
    def states(self):
        """Return the states / observations from the rollout storage."""
        return self.storage['states']

    @property
    def rewards(self):
        """Return the rewards tensor from the rollout storage."""
        return self.storage['rewards']

    @property
    def actions(self):
        """Return the actions from the rollout storage."""
        return self.storage['actions']

    @property
    def masks(self):
        """Return the masks tensor from the rollout storage."""
        return self.storage['masks']

    # @property
    # def values(self):
    #     """Return the value tensor V(s) from the rollout storage."""
    #     return self.storage.values
    #
    # @property
    # def action_values(self):
    #     """Return the action value tensor Q(s,a) from the rollout storage."""
    #     return self.storage.action_values

    @property
    def num_steps(self):
        """Return the total number of steps in the storage."""
        return self.storage.num_steps

    ###########
    # Methods #
    ###########

    def _evaluate(self):
        """Evaluate the estimator on the given rollout storage. To be implemented in the child class.

        Returns:
            torch.Tensor: the computed returns.
        """
        raise NotImplementedError

    def evaluate(self, storage=None, store=True):
        """Evaluate the estimator on the given rollout storage.

        Args:
            storage (RolloutStorage): rollout storage.
            store (True): This is always True.

        Returns:
            torch.Tensor: the computed returns.
        """
        # set storage
        if storage is not None:
            self.storage = storage

        # create returns in the storage
        if self not in self.storage:
            self.storage.create_new_entry(key=self, shapes=1, num_steps=self.num_steps+1)

            # add alias reference (note that modifying storage['returns'] will modify storage[self])
            self.storage['returns'] = self.storage[self]

        return self._evaluate()

    #############
    # Operators #
    #############

    # def __repr__(self):
    #     """Return a representation string of the object."""
    #     return self.__class__.__name__

    def __str__(self):
        """Return a string describing the object."""
        return self.__class__.__name__ + "(gamma=" + str(self.gamma) + ")"

    def __call__(self, storage=None):
        """Evaluate the estimator on the rollout storage.

        Args:
            storage (RolloutStorage): rollout storage.

        Returns:
            torch.Tensor: the computed returns.
        """
        return self.evaluate(storage=storage)


class TotalRewardEstimator(Estimator):
    r"""Total reward Estimator (aka (finite-horizon) discounted return)

    Return the total reward of the trajectory given by:

    .. math::

        \psi_t = R(\tau) = \sum_{t'=0}^{T} \gamma^{t'} r_{t'}

    where :math:`\tau` represents a trajectory :math:`\tau = (s_0, a_0, s_1,..., a_{T-1}, s_T)`.
    """

    def __init__(self, storage, gamma=1.):
        """
        Initialize the total reward estimator (also known as finite-horizon undiscounted return)

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
        """
        super(TotalRewardEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self):
        """Evaluate the estimator / return.

        Returns:
            torch.Tensor: the computed returns.
        """
        # get reference to data in storage
        returns, rewards, masks = self.returns, self.rewards, self.masks

        # compute the returns
        returns[-1] = rewards[-1]
        for t in reversed(range(self.num_steps)):
            returns[t] = rewards[t] + self.gamma * masks[t+1] * returns[t + 1]
        returns[:] = returns[0]

        return returns


class ActionRewardEstimator(Estimator):
    r"""Action Reward Estimator

    Return the accumulated reward following action :math:`a_t`:

    .. math::

        \psi_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}

    This is based on the observation that future actions don't have any effects on previous rewards.
    """

    def __init__(self, storage, gamma=1.):
        """
        Initialize the action reward estimator.

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
        """
        super(ActionRewardEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self):
        """Evaluate the estimator / return.

        Returns:
            torch.Tensor: the computed returns.
        """
        # get reference to data in storage
        returns, rewards, masks = self.returns, self.rewards, self.masks

        # compute the returns
        returns[-1] = rewards[-1]
        for t in reversed(range(self.num_steps)):
            returns[t] = rewards[t] + self.gamma * masks[t+1] * returns[t + 1]

        return returns


class BaselineRewardEstimator(ActionRewardEstimator):  # TODO: check with masks
    r"""Baseline Reward Estimator

    Return the accumulated reward following action a_t minus a baseline depending on the state:

    .. math::

        \psi_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} - b(s_t)

    This baseline allows to reduce the variance, and does not introduce any biases.
    The baseline is often selected to be the state value function: :math:`b(s_t) = V^{\pi}(s_t)`.
    """

    def __init__(self, storage, baseline, gamma=1.):
        """
        Initialize the TD residual Estimator.

        Args:
            storage (RolloutStorage): rollout storage
            baseline (callable): the baseline function which predicts a scalar given the state.
            gamma (float): discount factor
        """
        super(BaselineRewardEstimator, self).__init__(storage=storage, gamma=gamma)
        if not callable(baseline):
            raise TypeError("Expecting the given baseline to be callable.")
        self.baseline = baseline

    def _evaluate(self):
        """Evaluate the estimator / return.

        Returns:
            torch.Tensor: the computed returns.
        """
        # get reference to data in storage
        returns, rewards, masks, states = self.returns, self.rewards, self.masks, self.states

        # compute the returns
        returns[-1] = rewards[-1]
        for t in reversed(range(self.num_steps)):
            state = [state[t] for state in states]
            returns[t] = rewards[t] + self.gamma * masks[t+1] * returns[t + 1] - self.baseline(state)
        return returns


class ValueEstimator(Estimator):  # TODO: check with masks
    r"""State Value Estimator

    Return the state value function:

    .. math::

        \psi_t = V^{\pi, \gamma}(s_t) = \mathbb{E}_{s_{t+1:T}, a_{t:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    If the discount factor :math:`\gamma` is smaller than 1, then it will reduces the variance but at the cost
    of introducing a bias.
    """

    def __init__(self, storage, value, gamma=1.):
        """
        Initialize the state value estimator.

        Args:
            storage (RolloutStorage): rollout storage
            value (Value): value function approximator.
            gamma (float): discount factor
        """
        super(ValueEstimator, self).__init__(storage=storage, gamma=gamma)
        # check value function
        if not isinstance(value, Value):
            raise TypeError("Expecting the given value to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

    def _evaluate(self):
        """Evaluate the estimator / return.

        Returns:
            torch.Tensor: the computed returns.
        """
        # get reference to data in storage if present. If not, evaluate the value based on the states.
        if self._value in self.storage:
            values = self.storage[self._value]
        else:
            values = self._value(self.states)

        # compute the returns
        self.returns[:] = values.clone()
        return self.returns


class QValueEstimator(Estimator):  # TODO: check with masks
    r"""State Action Value Estimator

    Return the state action value function:

    .. math::

        \psi_t = Q^{\pi, \gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1:T}, a_{t+1:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    If the discount factor :math:`\gamma` is smaller than 1, then it will reduces the variance but at the cost
    of introducing a bias.
    """

    def __init__(self, storage, q_value, gamma=1.):
        """
        Initialize the station-action value estimator.

        Args:
            storage (RolloutStorage): rollout storage
            q_value (QValue): Q-value function approximator.
            gamma (float): discount factor
        """
        super(QValueEstimator, self).__init__(storage=storage, gamma=gamma)

        # check q-value
        if not isinstance(q_value, QValue):
            raise TypeError("Expecting the given q_value to be an instance of `QValue`, instead got: "
                            "{}".format(type(q_value)))
        self._q_value = q_value

    def _evaluate(self):
        """Evaluate the estimator / return.

        Returns:
            torch.Tensor: the computed returns.
        """
        # get reference to data in storage if present. If not, evaluate the value based on the states and actions.
        if self._q_value in self.storage:
            q_values = self.storage[self._q_value]
        else:
            q_values = self._q_value(self.states, self.actions)

        # compute the returns
        self.returns[:] = q_values.clone()
        return self.returns


class AdvantageEstimator(Estimator):  # TODO: check with masks
    r"""Advantage Estimator

    Return the advantage function:

    .. math::

        \psi_t = A^{\pi,\gamma}(s_t, a_t) = Q^{\pi,\gamma}(s_t,a_t) - V^{\pi,\gamma}(s_t)

    where

    .. math::

        Q^{\pi,\gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1:T}, a_{t+1:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

        V^{\pi,\gamma}(s_t) = \mathbb{E}_{s_{t+1:T}, a_{t:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    The advantage function represents what is the 'advantage' of taking a certain action at a certain state.
    If the discount factor :math:`\gamma` is smaller than 1, then it will reduces the variance but at the cost
    of introducing a bias.

    Note that:

    .. math::

        A^{\pi,\gamma}(s_t, a_t) &= Q^{\pi,\gamma}(s_t,a_t) - V^{\pi,\gamma}(s_t) \\
                             &= \mathbb{E}_{s_{t+1}}[ Q^{\pi,\gamma}(s_t,a_t) - V^{\pi,\gamma}(s_t) ] \\
                             &= \mathbb{E}_{s_{t+1}}[ r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t) ] \\
                             &= \mathbb{E}_{s_{t+1}}[ \delta_t^{V^{\pi,\gamma}} ]
    """

    def __init__(self, storage, value, q_value, gamma=1., standardize=False):
        """
        Initialize the Advantage estimator.

        Args:
            storage (RolloutStorage): rollout storage
            value (Value): value function approximator.
            q_value (QValue, Estimator): Q-value function approximator, or estimator.
            gamma (float): discount factor
            standardize (bool): If True, it will standardize the advantage estimates.
        """
        super(AdvantageEstimator, self).__init__(storage=storage, gamma=gamma)

        # check value function
        if not isinstance(value, Value):
            raise TypeError("Expecting the given value to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

        # check q-value
        if not isinstance(q_value, (QValue, Estimator)):
            raise TypeError("Expecting the given q_value to be an instance of `QValue` or `Estimator`, instead got: "
                            "{}".format(type(q_value)))
        self._q_value = q_value

        # set if we should standardize or not
        self._standardize = bool(standardize)

    def _evaluate(self):
        """Evaluate the estimator / return.

        Returns:
            torch.Tensor: the computed returns.
        """

        # get values from storage if present. If not, evaluate the value based on the states.
        if self._value in self.storage:
            values = self.storage[self._value]
        else:
            values = self._value(self.states)

        # get Q-values/estimators from storage if present. If not, evaluate the Q-values based on the states and
        # actions.
        if self._q_value in self.storage:
            q_values = self.storage[self._q_value]
        else:
            if isinstance(self._q_value, Estimator):
                # evaluate the estimator
                self._q_value.evaluate()
                # get the returns
                q_values = self.storage[self._q_value]
            else:
                q_values = self._q_value(self.states, self.actions)

        # compute the advantage estimates
        returns = q_values - values

        # standardize the advantage estimates
        if self._standardize:
            returns = (returns - returns.mean()) / (returns.std() + 1.e-5)

        # set the returns
        self.returns[:] = returns
        return self.returns


class TDResidualEstimator(Estimator):  # TODO: check with masks
    r"""TD Residual Estimator

    Return the temporal difference (TD) residual, given by:

    .. math:: \psi_t = \delta_t^{V^{\pi,\gamma}} = (r_t + \gamma V^{\pi,\gamma}(s_{t+1})) - V^{\pi,\gamma}(s_t)

    where,

    .. math:: V^{\pi,\gamma}(s_t) = \mathbb{E}_{s_{t+1:T}, a_{t:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    Note that:

    .. math:: A^{\pi,\gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1}}[ \delta_t^{V^{\pi,\gamma}} ]
    """

    def __init__(self, storage, value, gamma=1.):
        """
        Initialize the TD residual Estimator.

        Args:
            storage (RolloutStorage): rollout storage
            value (Value): value function approximator.
            gamma (float): discount factor
        """
        super(TDResidualEstimator, self).__init__(storage=storage, gamma=gamma)
        # check value function
        if not isinstance(value, Value):
            raise TypeError("Expecting the given value to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

    def _evaluate(self):  # , next_value):
        """Evaluate the estimator / return.

        Returns:
            torch.Tensor: the computed returns.
        """
        # get reference to data in storage
        returns, rewards, masks = self.returns, self.rewards, self.masks

        # get values from storage if present. If not, evaluate the value based on the states.
        if self._value in self.storage:
            values = self.storage[self._value]
        else:
            values = self._value(self.states)

        # compute the returns
        # self.returns[-1] = next_value
        for t in reversed(range(self.num_steps)):
            returns[t] = rewards[t] + self.gamma * masks[t+1] * values[t + 1] - values[t]

        return returns


class GAE(Estimator):
    r"""Generalized Advantage Estimator

    Return the GAE, which is the exponentially-weighted average of the discounted sum of the TD/Bellman residuals:

    .. math:: \psi_t = A^{GAE(\gamma, \tau)}_t = \sum_{l=0}^{T} (\gamma \tau)^{l} \delta_{t+l}^{V}

    Notes:

    .. math::

        GAE(\gamma, 0) = \delta_t^{V} = r_t + \gamma V(s_{t+1}) - V(s_t)

        GAE(\gamma, 1) = \sum_{l=0}^{T} \gamma^{l} \delta_{t+l}^{V} = \sum_{l=0}^{T} \gamma^{l} r_{t+l} - V(s_t)

    where :math:`GAE(\gamma, 1)` has high variance, while :math:`GAE(\gamma, 0)` has usually lower variance.

    A compromise between bias and variance is made by controlling the open trace-decay parameter :math:`\tau`.
    Good values for GAE are obtained when :math:`\gamma` and :math:`\tau` are in :math:`[0.9,0.99]`.

    References:
        - [1] "High-Dimensional Continuous Control using Generalized Advantage Estimation", Schulman et al., 2016
    """

    def __init__(self, storage, value, gamma=0.98, tau=0.99):
        """
        Initialize the Generalized Advantage Estimator.

        Args:
            storage (RolloutStorage): rollout storage
            value (Value): value function approximator.
            gamma (float): discount factor
            tau (float): trace-decay parameter (which is a bias-variance tradeoff). If :math:`\tau=1`, this results
                in a Monte Carlo method, while :math:`\tau=0` results in a one-step TD methods.
        """
        super(GAE, self).__init__(storage=storage, gamma=gamma)
        self.tau = tau
        # check value function
        if not isinstance(value, Value):
            raise TypeError("Expecting the given value to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

    def _evaluate(self):
        """Evaluate the estimator / return.

        Returns:
            torch.Tensor: the computed returns.
        """
        # get reference to data in storage
        returns, rewards, masks = self.returns, self.rewards, self.masks

        # get values from storage if present. If not, evaluate the value based on the states.
        if self._value in self.storage:
            values = self.storage[self._value]
        else:
            values = self._value(self.states)

        # compute the returns
        # self.values[-1] = next_value
        gae = 0
        for t in reversed(range(self.num_steps)):
            delta = rewards[t] + self.gamma * values[t + 1] * masks[t + 1] - values[t]
            gae = delta + self.gamma * self.tau * masks[t + 1] * gae
            returns[t] = gae + values[t]

        return returns

    def __str__(self):
        """Return a string describing the object."""
        return self.__class__.__name__ + "(gamma=" + str(self.gamma) + ", tau=" + str(self.tau) + ")"
