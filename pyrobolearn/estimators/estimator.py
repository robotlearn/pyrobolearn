#!/usr/bin/env python
"""Describes the various `Estimators` / `Returns` used in reinforcement learning.

Dependencies:
- `pyrobolearn.storages`
"""

from abc import ABCMeta
import collections

import torch

from pyrobolearn.storages import RolloutStorage, Batch
from pyrobolearn.values import Value, QValue


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BaseReturn(object):
    r"""Base Return / Estimator

    Base return / estimator computed in RL algorithms.
    """

    __metaclass__ = ABCMeta

    def __init__(self, gamma=1.):
        """
        Initialize the base return / estimator.

        Args:
            gamma (float): discount factor
        """
        self.gamma = gamma

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


class Estimator(BaseReturn):
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
        [1] "High-Dimensional Continuous Control using Generalized Advantage Estimation", Schulman et al., 2016
    """

    def __init__(self, storage, gamma=1.):
        """
        Initialize the estimator / return function.

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
        """
        super(Estimator, self).__init__(gamma)
        self.storage = storage

    ##############
    # Properties #
    ##############

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
        return self.storage.returns

    @property
    def rewards(self):
        """Return the rewards tensor from the rollout storage."""
        return self.storage.rewards

    @property
    def masks(self):
        """Return the masks tensor from the rollout storage."""
        return self.storage.masks

    @property
    def values(self):
        """Return the value tensor V(s) from the rollout storage."""
        return self.storage.values

    @property
    def action_values(self):
        """Return the action value tensor Q(s,a) from the rollout storage."""
        return self.storage.action_values

    @property
    def num_steps(self):
        """Return the total number of steps in the storage."""
        return self.storage.num_steps

    @property
    def states(self):
        """Return the states / observations from the rollout storage."""
        return self.storage.observations

    ###########
    # Methods #
    ###########

    def _evaluate(self):
        """Evaluate the estimator.
        To be implemented in the child class.
        """
        raise NotImplementedError

    def evaluate(self, storage=None):
        """Evaluate the estimator"""
        if storage is not None:
            self.storage = storage
        return self._evaluate()

    #############
    # Operators #
    #############

    def __call__(self):
        """Evaluate the estimator."""
        self.evaluate()


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
        """Evaluate the estimator / return."""
        self.returns[-1] = self.rewards[-1]
        for t in reversed(range(self.num_steps)):
            self.returns[t] = self.rewards[t] + self.gamma * self.returns[t + 1]
        self.returns[:] = self.returns[0]


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
        """Evaluate the estimator / return."""
        self.returns[-1] = self.rewards[-1]
        for t in reversed(range(self.num_steps)):
            self.returns[t] = self.rewards[t] + self.gamma * self.returns[t + 1]


class BaselineRewardEstimator(ActionRewardEstimator):
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
        """Evaluate the estimator / return."""
        self.returns[-1] = self.rewards[-1]
        for t in reversed(range(self.num_steps)):
            self.returns[t] = self.rewards[t] + self.gamma * self.returns[t + 1] - self.baseline(self.states[t])


class ValueEstimator(Estimator):
    r"""State Value Estimator

    Return the state value function:

    .. math::

        \psi_t = V^{\pi, \gamma}(s_t) = \mathbb{E}_{s_{t+1:T}, a_{t:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    If the discount factor :math:`\gamma` is smaller than 1, then it will reduces the variance but at the cost
    of introducing a bias.
    """

    def __init__(self, storage, gamma=1.):
        """
        Initialize the state value estimator.

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
        """
        super(ValueEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self):
        """Evaluate the estimator / return."""
        self.returns[:] = self.values.clone()


class QValueEstimator(Estimator):
    r"""State Action Value Estimator

    Return the state action value function:

    .. math::

        \psi_t = Q^{\pi, \gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1:T}, a_{t+1:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    If the discount factor :math:`\gamma` is smaller than 1, then it will reduces the variance but at the cost
    of introducing a bias.
    """

    def __init__(self, storage, gamma=1.):
        """
        Initialize the station-action value estimator.

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
        """
        super(QValueEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self):
        """Evaluate the estimator / return."""
        self.returns[:] = self.action_values.clone()


class AdvantageEstimator(Estimator):
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

    def __init__(self, storage, gamma=1.):
        """
        Initialize the Advantage estimator.

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
        """
        super(AdvantageEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self):
        """Evaluate the estimator / return."""
        self.returns[:] = self.action_values - self.values


class TDResidualEstimator(Estimator):
    r"""TD Residual Estimator

    Return the temporal difference (TD) residual, given by:

    .. math:: \psi_t = \delta_t^{V^{\pi,\gamma}} = (r_t + \gamma V^{\pi,\gamma}(s_{t+1})) - V^{\pi,\gamma}(s_t)

    where,

    .. math:: V^{\pi,\gamma}(s_t) = \mathbb{E}_{s_{t+1:T}, a_{t:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    Note that:

    .. math:: A^{\pi,\gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1}}[ \delta_t^{V^{\pi,\gamma}} ]
    """

    def __init__(self, storage, gamma=1.):
        """
        Initialize the TD residual Estimator.

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
        """
        super(TDResidualEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self):  # , next_value):
        """Evaluate the estimator / return."""
        # self.returns[-1] = next_value
        for t in reversed(range(self.num_steps)):
            self.returns[t] = self.rewards[t] + self.gamma * self.values[t + 1] - self.values[t]


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
        [1] "High-Dimensional Continuous Control using Generalized Advantage Estimation", Schulman et al., 2016
    """

    def __init__(self, storage, gamma=0.98, tau=0.99):
        """
        Initialize the Generalized Advantage Estimator.

        Args:
            storage (RolloutStorage): rollout storage
            gamma (float): discount factor
            tau (float): trace-decay parameter (which is a bias-variance tradeoff). If :math:`\tau=1`, this results
                in a Monte Carlo method, while :math:`\tau=0` results in a one-step TD methods.
        """
        super(GAE, self).__init__(storage=storage, gamma=gamma)
        self.tau = tau

    def _evaluate(self):  # , next_value):
        """Evaluate the estimator / return."""
        # self.values[-1] = next_value
        gae = 0
        for t in reversed(range(self.num_steps)):
            delta = self.rewards[t] + self.gamma * self.values[t + 1] * self.masks[t + 1] - self.values[t]
            gae = delta + self.gamma * self.tau * self.masks[t + 1] * gae
            self.returns[t] = gae + self.values[t]


class Return(BaseReturn):
    r"""Return

    Compared to the estimator that used the whole trajectory, it only uses transition tuples
    :math:`(s_t, a_t, s_{t+1}, r_t, d)`, where is :math:`s_t` is the state at time :math:`t`, :math:`a_t` is the action
    outputted by the policy in response to the state :math:`s_t`, :math:`s_{t+1}` is the next state returned by the
    environment due to the policy's action :math:`a_t` and the current state :math:`s_t`, :math:`r_t` is the reward
    signal returned by the environment, and :math:`d` is a boolean value that specifies if the task is over or not
    (i.e. if it has failed or succeeded).

    That is, estimators are estimated on the Monte-Carlo trajectories, while returns are estimated on temporal
    difference errors [1].

    References:
        [1] "Reinforcement Learning: an Introduction" (chap 8.13), Sutton and Barto, 2018
    """

    def __init__(self, gamma=1.):
        """
        Initialize the return / estimator.

        Args:
            gamma (float): discount factor
        """
        super(Return, self).__init__(gamma)


class TDReturn(Return):
    r"""TD Return

    Return based on the one-step temporal difference TD(0).
    """
    pass


class Target(BaseReturn):

    def compute(self, batch):
        pass

    def __call__(self, batch):
        return self.compute(batch)


class ValueTarget(Target):
    r"""Value target.

    Compute the value target given by :math:`(r + \gamma (1-d) \min_i V_{\phi_i}(s'))`, where the index `i` is in
    the case there are multiple value function approximators given to this class.
    """

    def __init__(self, values, gamma=1.):
        """
        Initialize the state value target.

        Args:
            values (Value, list of Value): state value function(s).
            gamma (float): discount factor
        """
        super(ValueTarget, self).__init__(gamma)
        if not isinstance(values, collections.Iterable):
            values = [values]
        for i, value in enumerate(values):
            if not isinstance(value, Value):
                raise TypeError('The {}th value is not an instance of `Value`, instead got: {}'.format(i, type(value)))
        self._values = values

    def compute(self, batch):
        r"""
        Compute the value target :math:`(r + \gamma (1-d) \min_i V_{\phi_i}(s'))`

        Args:
            batch (Batch): batch containing the transitions.
        """
        value = torch.min(torch.cat([value(batch['states']) for value in self._values], dim=1), dim=1)[0]
        batch[self] = batch['rewards'] + self.gamma * (1 - batch['masks']) * value
        return batch


class QValueTarget(Target):
    r"""Q-Value target.

    Compute the Q-value target given by :math:`(r + \gamma (1-d) \min_i Q_{\phi_i}(s',a'))`, where the index `i` is in
    the case there are multiple Q-value function approximators given to this class.
    """

    def __init__(self, q_values, gamma=1.):
        """
        Initialize the state value target.

        Args:
            q_values (QValue, list of QValue): state-action value function(s).
            gamma (float): discount factor
        """
        super(QValueTarget, self).__init__(gamma)
        if not isinstance(q_values, collections.Iterable):
            q_values = [q_values]
        for i, value in enumerate(q_values):
            if not isinstance(value, QValue):
                raise TypeError('The {}th value is not an instance of `QValue`, instead got: {}'.format(i, type(value)))
        self._q_values = q_values

    def compute(self, batch):
        r"""
        Compute the value target :math:`(r + \gamma (1-d) \min_i Q_{\phi_i}(s',a'))`

        Args:
            batch (Batch): batch containing the transitions.
        """
        value = torch.min(torch.cat([value(batch['states']) for value in self._q_values], dim=1), dim=1)[0]
        batch[self] = batch['rewards'] + self.gamma * (1 - batch['masks']) * value
        return batch


class QLearningTarget(Target):
    r"""Q-Learning target.

    Compute the Q-value target given by :math:`(r + \gamma (1-d) \min_i \max_{a'} Q_{\phi_i}(s',a'))`, where the
    index `i` is in the case there are multiple Q-value function approximators given to this class.
    """

    def __init__(self, q_values, gamma=1.):
        """
        Initialize the state value target.

        Args:
            q_values (QValue, list of QValue): state-action value function(s).
            gamma (float): discount factor
        """
        super(QLearningTarget, self).__init__(gamma)
        if not isinstance(q_values, collections.Iterable):
            q_values = [q_values]
        for i, value in enumerate(q_values):
            if not isinstance(value, QValue):
                raise TypeError('The {}th value is not an instance of `QValue`, instead got: {}'.format(i, type(value)))
        self._q_values = q_values

    def compute(self, batch):
        r"""
        Compute the value target :math:`(r + \gamma (1-d) \min_i \max_{a'} Q_{\phi_i}(s',a'))`.

        Args:
            batch (Batch): batch containing the transitions.
        """
        q_max = [torch.max(value(batch['states']), dim=1, keepdim=True)[0] for value in self._q_values]
        value = torch.min(torch.cat(q_max, dim=1), dim=1)[0]
        batch[self] = batch['rewards'] + self.gamma * (1 - batch['masks']) * value
        return batch


class TDValueReturn(TDReturn):
    r"""TD State Value Return

    Compute the state value one-step temporal difference TD(0), given by:
    .. math:: \delta_t^{V} = (r + \gamma (1-d) V_{\phi_{target}}(s')) - V_{\phi}(s)

    where :math:`V_{\phi_{target}}` is the target value function.
    """

    def __init__(self, value, target_value=None, gamma=1.):
        """
        Initialize the TD state value return.

        Args:
            value (ValueApproximator): state value function.
            target_value (ValueApproximator): target state value function.
            gamma (float): discount factor
        """
        super(TDValueReturn, self).__init__(gamma)
        self.value = value
        if target_value is None:
            self.target_value = self.value

    def evaluate(self, batch):
        """Evaluate the TD return on the given batch.

        Args:
            batch (Batch): batch containing transitions.

        Returns:
            Batch: batch
        """
        target = batch['rewards'] + self.gamma * (1 - batch['masks']) * self.target_value(batch['states'])
        batch[self] = target - self.value(batch['states'])
        return batch


class TDQValueReturn(TDReturn):
    r"""TD Action Value Return

    Compute the action value one-step temporal difference TD(0), given by:
    .. math:: (r + \gamma (1-d) Q_{\phi_{target}}(s',a')) - Q_{\phi}(s,a)
    where :math:`a'` is the action performed by the policy given the state :math:`s'`.

    This is also known as the Sarsa (an on-policy TD control) algorithm.
    """

    def __init__(self, q_value, policy, target_qvalue=None, gamma=1.):
        """
        Initialize the TD state-action value return.

        Args:
            q_value (QValueApproximator): Q-value function.
            policy (Policy): policy to compute the action a'.
            target_qvalue (QValueApproximator, None): target Q-value function. If None, it will use the given
                :attr:`q_value`.
            gamma (float): discount factor
        """
        super(TDQValueReturn, self).__init__(gamma)
        self.q_value = q_value
        self.policy = policy
        if target_qvalue is None:
            self.target_qvalue = self.q_value

    def evaluate(self, batch):
        """Evaluate the TD return on the given batch.

        Args:
            batch (Batch): batch containing transitions.

        Returns:
            Batch: batch
        """
        action = self.policy.predict(batch['states'])
        target = batch['rewards'] + self.gamma * (1 - batch['masks']) * self.target_qvalue(action)
        batch[self] = target - self.q_value(batch['states'])
        return batch


class TDQLearningReturn(TDReturn):
    r"""TD Q-Learning Value Return

    Compute the one-step Q-Learning, given by:
    .. math:: (r + \gamma (1-d) \max_{a'} Q_{\phi_{target}}(s',a')) - Q_{\phi}(s,a)

    where if the actions :math:`a` are discrete, then :math:`a'` is selected such that it maximizes the Q-value, while
    if :math:`a` are continuous, :math:`a'`, with the assumption that the policy is fully differentiable, is selected
    such that it maximizes locally the Q-value. That is, in the latter case, the action :math:`a'` is first computed
    using the policy :math:`\pi(a'|s')`, then by taking the gradient of the Q-value with respect to the action, we
    increment the initial action by :math:`a' = a' + \alpha \grad_{a} Q_{\phi_{target}}(s',a)`. This increment step
    can be computed for few iterations such that it locally maximizes the :math:` Q_{\phi_{target}}`.

    This is known as the Q-Learning (an off-policy TD control) algorithm.
    """

    def __init__(self, q_value, target_qvalue=None, gamma=1.):
        """
        Initialize the TD Q-Learning value return.

        Args:
            q_value (QValueApproximator): Q-value function.
            target_qvalue (QValueApproximator): target Q-value function. If None, it will use the given
                :attr:`q_value`.
            gamma (float): discount factor
        """
        super(TDQLearningReturn, self).__init__(gamma)
        self.q_value = q_value
        if target_qvalue is None:
            self.target_qvalue = self.q_value

    def evaluate(self, batch):
        """Evaluate the TD return on the given batch.

        Args:
            batch (Batch): batch containing transitions.

        Returns:
            Batch: batch
        """
        q_max = torch.max(self.target_qvalue(batch['states']), dim=1, keepdim=True)[0]
        target = batch['rewards'] + self.gamma * (1 - batch['masks']) * q_max
        batch[self] = target - self.q_value(batch['states'])
        return batch
