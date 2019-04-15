#!/usr/bin/env python
"""Computes the various returns evaluated on batches of transitions (used in RL).

The targets that are evaluated are placed inside the given batch, which can then be accessed by other classes.

Dependencies:
- `pyrobolearn.storages`
- `pyrobolearn.values`
"""

import torch

from pyrobolearn.storages import Batch
from pyrobolearn.values import Value, QValue
from pyrobolearn.policies import Policy
from pyrobolearn.returns.estimator import BaseReturn


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Return(BaseReturn):
    r"""Return

    Compared to the estimator that used the whole trajectory, it only uses transition tuples
    :math:`(s_t, a_t, s_{t+1}, r_t, d)`, where is :math:`s_t` is the state at time :math:`t`, :math:`a_t` is the action
    outputted by the policy in response to the state :math:`s_t`, :math:`s_{t+1}` is the next state returned by the
    environment due to the policy's action :math:`a_t` and the current state :math:`s_t`, :math:`r_t` is the reward
    signal returned by the environment, and :math:`d` is a boolean value that specifies if the task is over or not
    (i.e. if it has failed or succeeded).

    That is, returns are estimated on the Monte-Carlo trajectories, while returns are estimated on temporal
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
        action = self.policy.predict(batch['next_states'])
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
