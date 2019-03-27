#!/usr/bin/env python
"""Describes the various `Estimators` used in reinforcement learning.

Dependencies:
- `pyrobolearn.storages`
"""

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Estimator(object):
    r"""Estimator / Returns

    Estimator used for gradient based algorithms. The gradient is given by:

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
        Initialize the storage.

        Args:
            storage: storage
            gamma (float): discount factor
        """
        self.storage = storage
        self.gamma = gamma

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
        self._storage = storage

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

    ###########
    # Methods #
    ###########

    def _evaluate(self, storage):
        """Evaluate the estimator.
        To be implemented in the child class.
        """
        raise NotImplementedError

    def evaluate(self, storage):
        """Evaluate the estimator"""
        if storage is None:
            storage = self.storage
        return self.evaluate(storage)

    #############
    # Operators #
    #############

    def __call__(self):
        self.evaluate()


class TotalRewardEstimator(Estimator):
    r"""Total reward Estimator

    Return the total reward of the trajectory given by:

    .. math::

        \psi_t = R(\tau) = \sum_{t'=0}^{T} \gamma^{t'} r_{t'}

    where :math:`\tau` represents a trajectory :math:`\tau = (s_0, a_0, s_1,..., a_{T-1}, s_T)`.
    """

    def __init__(self, storage, gamma=1.):
        super(TotalRewardEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self, storage):
        storage.returns[-1] = storage.rewards[-1]
        for t in reversed(range(storage.num_steps)):
            storage.returns[t] = storage.rewards[t] + self.gamma * storage.returns[t + 1]
        storage.returns[:] = storage.returns[0]


class ActionRewardEstimator(Estimator):
    r"""Action Reward Estimator

    Return the accumulated reward following action :math:`a_t`:

    .. math::

        \psi_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}

    This is based on the observation that future actions don't have any effects on previous rewards.
    """

    def __init__(self, storage, gamma=1.):
        super(ActionRewardEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self, storage):
        storage.returns[-1] = storage.rewards[-1]
        for t in reversed(range(storage.num_steps)):
            storage.returns[t] = storage.rewards[t] + self.gamma * storage.returns[t + 1]


class BaselineRewardEstimator(ActionRewardEstimator):
    r"""Baseline Reward Estimator

    Return the accumulated reward following action a_t minus a baseline depending on the state:

    .. math::

        \psi_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} - b(s_t)

    This baseline allows to reduce the variance, and does not introduce any biases.
    The baseline is often selected to be the state value function: :math:`b(s_t) = V^{\pi}(s_t)`.
    """

    def __init__(self, storage, baseline, gamma=1.):
        super(BaselineRewardEstimator, self).__init__(storage=storage, gamma=gamma)
        self.baseline = baseline

    def _evaluate(self, storage):
        storage.returns[-1] = storage.rewards[-1]
        for t in reversed(range(storage.num_steps)):
            storage.returns[t] = storage.rewards[t] + self.gamma * storage.returns[t + 1] \
                                - self.baseline(storage.states[t])


class StateValueEstimator(Estimator):
    r"""State Value Estimator

    Return the state value function:

    .. math::

        \psi_t = V^{\pi, \gamma}(s_t) = \mathbb{E}_{s_{t+1:T}, a_{t:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    If the discount factor :math:`\gamma` is smaller than 1, then it will reduces the variance but at the cost
    of introducing a bias.
    """

    def __init__(self, storage, gamma=1.):
        super(StateValueEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self, storage):
        storage.returns = storage.values.clone()


class StateActionValueEstimator(Estimator):
    r"""State Action Value Estimator

    Return the state action value function:

    .. math::

        \psi_t = Q^{\pi, \gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1:T}, a_{t+1:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    If the discount factor :math:`\gamma` is smaller than 1, then it will reduces the variance but at the cost
    of introducing a bias.
    """

    def __init__(self, storage, gamma=1.):
        super(StateActionValueEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self, storage):
        storage.returns = storage.action_values.clone()


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
        super(AdvantageEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self, storage):
        storage.returns = storage.action_values - storage.values


class TDResidualEstimator(Estimator):
    r"""TD Residual Estimator

    Return the TD residual, given by:

    .. math:: \psi_t = \delta_t^{V^{\pi,\gamma}} = r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t)

    where,

    .. math:: V^{\pi,\gamma}(s_t) = \mathbb{E}_{s_{t+1:T}, a_{t:T}}[ \sum_{l=0}^{T} \gamma^{l} r_{t+l} ]

    Note that:

    .. math:: A^{\pi,\gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1}}[ \delta_t^{V^{\pi,\gamma}} ]
    """

    def __init__(self, storage, gamma=1.):
        super(TDResidualEstimator, self).__init__(storage=storage, gamma=gamma)

    def _evaluate(self, storage=None):  # , next_value):
        # storage.returns[-1] = next_value
        for t in reversed(range(storage.num_steps)):
            storage.returns[t] = storage.rewards[t] + self.gamma * storage.values[t + 1] - storage.values[t]


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
            storage:
            gamma (float): discount factor
            tau (float): trace-decay parameter (which is a bias-variance tradeoff). If :math:`\tau=1`, this results
                in a Monte Carlo method, while :math:`\tau=0` results in a one-step TD methods.
        """
        super(GAE, self).__init__(storage=storage, gamma=gamma)
        self.tau = tau

    def _evaluate(self, storage):  # , next_value):
        # storage.values[-1] = next_value
        gae = 0
        for t in reversed(range(storage.num_steps)):
            delta = storage.rewards[t] + self.gamma * storage.values[t + 1] * storage.masks[t + 1] - storage.values[t]
            gae = delta + self.gamma * self.tau * storage.masks[t + 1] * gae
            storage.returns[t] = gae + storage.values[t]
