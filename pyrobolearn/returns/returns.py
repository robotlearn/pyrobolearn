#!/usr/bin/env python
"""Computes the various returns evaluated on batches of transitions (used in RL).

The targets that are evaluated are placed inside the given batch, which can then be accessed by other classes.

Dependencies:
- `pyrobolearn.storages`
- `pyrobolearn.values`
"""

from abc import ABCMeta

import torch

from pyrobolearn.storages import Batch
from pyrobolearn.values import Value, QValue, QValueOutput
from pyrobolearn.policies import Policy
from pyrobolearn.returns.targets import ValueTarget, QValueTarget, QLearningTarget


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Return(object):
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

    def _evaluate(self, batch):
        """
        Evaluate the return on the given batch.

        Args:
            batch (Batch): batch containing the transitions.

        Returns:
            torch.Tensor: evaluated return.
        """
        raise NotImplementedError

    def evaluate(self, batch, store=True):
        """
        Evaluate the return on the given batch.

        Args:
            batch (Batch): batch containing the transitions.
            store (bool): If True, it will save the evaluation of the target in the given batch.

        Returns:
            torch.Tensor: evaluated return.
        """
        output = self._evaluate(batch)
        if store:  # store the target in the batch if specified
            batch[self] = output
        return output

    def __call__(self, batch, store=True):
        """
        Evaluate the return on the given batch.

        Args:
            batch (Batch): batch containing the transitions.
            store (bool): If True, it will save the evaluation of the target in the given batch.

        Returns:
            torch.Tensor: evaluated return.
        """
        return self.evaluate(batch, store=store)


class TDReturn(Return):
    r"""TD Return

    Return based on the one-step temporal difference TD(0).
    """

    __metaclass__ = ABCMeta

    def __init__(self, gamma=1.):
        """
        Initialize the base return / estimator.

        Args:
            gamma (float): discount factor
        """
        super(TDReturn, self).__init__()
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
            value (Value): state value function.
            target_value (Value, list of Value, None): target state value function(s). If None, it will be set to be
                the same as the given :attr:`value`. Note however that this can lead to unstable behaviors.
            gamma (float): discount factor
        """
        super(TDValueReturn, self).__init__(gamma)

        # check value
        if not isinstance(value, Value):
            raise TypeError("Expecting the given value to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

        # check target value
        if target_value is None:
            # Warning: using the same value function for the target can lead to unstable behaviors.
            target_value = value
        self._target = ValueTarget(values=target_value, gamma=gamma)

    def _evaluate(self, batch):
        """Evaluate the TD return on the given batch.

        Args:
            batch (Batch): batch containing transitions.

        Returns:
            torch.Tensor: evaluated TD-return.
        """
        target = self._target(batch, store=False)
        return target - self._value(batch['states'])


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
            q_value (QValue): Q-value function.
            policy (Policy): policy to compute the action a'.
            target_qvalue (QValue, list of QValue, None): target Q-value function(s). If None, it will use the given
                :attr:`q_value`. Note however that this can lead to unstable behaviors.
            gamma (float): discount factor
        """
        super(TDQValueReturn, self).__init__(gamma)

        # check Q-value
        if not isinstance(q_value, QValue):
            raise TypeError("Expecting the given q_value to be an instance of `QValue`, instead got: "
                            "{}".format(type(q_value)))
        self._q_value = q_value

        # check target Q-value
        if target_qvalue is None:
            # Warning: using the same value function for the target can lead to unstable behaviors.
            target_qvalue = q_value
        self._target = QValueTarget(q_values=target_qvalue, policy=policy, gamma=gamma)

    def _evaluate(self, batch):
        """Evaluate the TD return on the given batch.

        Args:
            batch (Batch): batch containing transitions.

        Returns:
            torch.Tensor: evaluated TD-return.
        """
        target = self._target(batch, store=False)
        return target - self._q_value(batch['states'], actions)


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
            q_value (QValue): Q-value function.
            target_qvalue (QValue): target Q-value function. If None, it will use the given :attr:`q_value`.
            gamma (float): discount factor
        """
        super(TDQLearningReturn, self).__init__(gamma)

        # check Q-value
        if not isinstance(q_value, QValueOutput):
            raise TypeError("Expecting the given q_value to be an instance of `QValueOutput`, instead got: "
                            "{}".format(type(q_value)))
        self._q_value = q_value

        # check target Q-value
        if target_qvalue is None:
            # Warning: using the same value function for the target can lead to unstable behaviors.
            target_qvalue = q_value
        self._target = QLearningTarget(q_values=target_qvalue, gamma=gamma)

    def _evaluate(self, batch):
        """Evaluate the TD return on the given batch.

        Args:
            batch (Batch): batch containing transitions.

        Returns:
            torch.Tensor: evaluated TD-return.
        """
        target = self._target(batch, store=False)
        return target - self._q_value(batch['states'])
