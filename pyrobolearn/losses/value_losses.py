# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Defines the value losses in RL.
"""

import torch

from pyrobolearn.losses import BatchLoss
from pyrobolearn.policies import Policy
from pyrobolearn.values import QValue, Value
from pyrobolearn.returns import TDReturn, Estimator, Return


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ValueL2Loss(BatchLoss):
    r"""L2 loss for values
    """

    def __init__(self, returns, value):
        """
        Initialize the L2 loss between the returns and values.

        Args:
            returns (Estimator, Return): returns.
            value (Value): value function approximator.
        """
        super(ValueL2Loss, self).__init__()

        # check the given returns or estimators
        if not isinstance(returns, (Estimator, Return)):
            raise TypeError("Expecting the given 'returns' to be an instance of `Estimator` or `Return`, instead got: "
                            "{}".format(type(returns)))
        self._returns = returns

        # check the given value approximator
        if not isinstance(value, Value):
            raise TypeError("Expecting the given 'value' to be an instance of `Value`, instead got: "
                            "{}".format(type(value)))
        self._value = value

    def _compute(self, batch):
        """
        Compute the loss on the given batch.

        Args:
            batch (Batch): batch that contains the 'states'.

        Returns:
            torch.tensor: loss scalar value
        """
        if self._returns in batch.current:
            returns = batch.current[self._returns]
        elif self._returns in batch:
            returns = batch[self._returns]
        else:
            returns = self._returns.evaluate(batch, store=False)

        if self._value in batch.current:
            values = batch.current[self._value]
        elif self._value in batch:
            values = batch[self._value]
        else:
            values = self._value(state=batch['states'])
        return 0.5 * (returns - values).pow(2).mean()


class QLoss(BatchLoss):
    r"""QLoss

    This computes :math:`\frac{1}{|B|} \sum_{s \in B} Q_{s, \mu_{\theta}(s)}}`, where :math:`\mu_\theta` is the
    policy, and this quantity is being maximized.
    """

    def __init__(self, q_value, policy):
        """
        Initialize the Q-loss.

        Args:
            q_value (QValue): Q-value function approximator.
            policy (Policy): policy.
        """
        super(QLoss, self).__init__()

        # check the given q_value
        if not isinstance(q_value, QValue):
            raise TypeError("Expecting the given q_value to be an instance of `QValue`, instead got: "
                            "{}".format(type(q_value)))
        self._q_value = q_value

        # check the policy
        if not isinstance(policy, Policy):
            raise TypeError("Expecting the given policy to be an instance of `Policy`, instead got: "
                            "{}".format(type(policy)))
        self._policy = policy

    def _compute(self, batch):
        """
        Compute the loss on the given batch.

        Args:
            batch (Batch): batch that contains the 'states'.

        Returns:
            torch.tensor: loss scalar value
        """
        actions = self._policy.predict(batch['states'])
        q_values = self._q_value(batch['states'], actions)
        return -q_values.mean()


class MSBELoss(BatchLoss):
    r"""Mean-squared Bellman error

    The mean-squared Bellman error (MSBE) computes the Bellman error, also known as the one-step temporal difference
    (TD).

    This is given by, in the case of the off-policy Q-learning TD algorithm:
    .. math:: (r + \gamma (1-d) max_{a'} Q_{\phi}(s',a')) - Q_{\phi}(s,a),

    or in the case of the on-policy Sarsa TD algorithm:
    .. math:: (r + \gamma (1-d) Q_{\phi}(s',a')) - Q_{\phi}(s,a)

    or in the case of the TD(0):
    .. math:: (r + \gamma (1-d) V_{\phi}(s')) - V_{\phi}(s)

    where (r + \gamma (1-d) f(s,a)) is called (one-step return) the target. The target could also be, instead of the
    one step return, the n-step return or the lambda-return value. [2]
    These losses roughly tell us how closely the value functions satisfy the Bellman equation. Thus, by trying to
    minimize them, we try to enforce the Bellman equations.

    References:
        - [1] https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        - [2] "Reinforcement Learning: An Introduction", Sutton and Barto, 2018
    """

    def __init__(self, td_return):
        """
        Initialize the mean-squared Bellman error (MSBE).

        Args:
            td_return (TDReturn): Temporal difference return.
        """
        super(MSBELoss, self).__init__()
        if not isinstance(td_return, TDReturn):
            raise TypeError("Expecting the given 'td_return' to be an instance of `TDReturn`, instead got: "
                            "{}".format(type(td_return)))
        self._td = td_return

    def _compute(self, batch):
        """
        Compute the mean-squared TD return.

        Args:
            batch (Batch): batch that contains the td returns.

        Returns:
            torch.Tensor: loss value.
        """
        if self._td in batch.current:
            returns = batch.current[self._td]
        elif self._td in batch:
            returns = batch[self._td]
        else:
            returns = self._td.evaluate(batch, store=False)
        return returns.pow(2).mean()
