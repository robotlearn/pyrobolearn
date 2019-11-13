#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Computes the various targets based on value function on batches of trajectories/transitions (used in RL).

The targets that are evaluated are placed inside the given batch, which can then be accessed by other classes.

Dependencies:
- `pyrobolearn.storages`
- `pyrobolearn.values`
"""

from abc import ABCMeta
import collections

import torch

from pyrobolearn.storages import Batch
from pyrobolearn.values import Value, QValue
from pyrobolearn.policies import Policy
from pyrobolearn.exploration import Exploration


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Target(object):
    r"""Target

    Targets are evaluated on a batch of transitions or trajectories, and inserted back in the given batch.
    """

    def __init__(self, targets=None):
        """
        Initialize the target.

        Args:
            targets (None, list of Target): inner targets.
        """
        # check targets
        if targets is not None:
            if not isinstance(targets, collections.Iterable):
                targets = [targets]
            for i, target in enumerate(targets):
                if not isinstance(target, Target):
                    raise TypeError("The {}th given target is not an instance of `Target`, instead got: "
                                    "{}".format(i, type(target)))
        else:
            targets = []
        self._targets = targets

    def _evaluate(self, batch):
        """Compute/evaluate the target on the given batch, and return the result.

        Args:
            batch (Batch): batch containing the transitions / trajectories.

        Returns:
            torch.Tensor: evaluated targets.
        """
        raise NotImplementedError

    def evaluate(self, batch, store=True):
        """Compute/evaluate the target on the given batch and insert the result in the given batch.

        Args:
            batch (Batch): batch containing the transitions / trajectories.
            store (bool): If True, it will save the evaluation of the target in the given batch.

        Returns:
            torch.Tensor: evaluated targets.
        """
        # check batch type
        if not isinstance(batch, Batch):
            raise TypeError("Expecting the given 'batch' to be an instance of `Batch`, instead got: "
                            "{}".format(type(batch)))

        output = self._evaluate(batch)
        if store:  # store the target in the current batch if specified
            batch.current[self] = output
        return output

    def __repr__(self):
        """Return a representation string of the object."""
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the object."""
        return self.__class__.__name__

    def __call__(self, batch, store=True):
        """Evaluate the target on the given batch."""
        return self.evaluate(batch)


class GammaTarget(Target):

    __metaclass__ = ABCMeta

    def __init__(self, gamma=1.):
        """
        Initialize the gamma target.

        Args:
            gamma (float): discount factor
        """
        super(GammaTarget, self).__init__()
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

    def __str__(self):
        """Return a string describing the object."""
        return self.__class__.__name__ + "(gamma=" + str(self.gamma) + ")"


class VTarget(Target):
    r"""Value Target.

    Evaluate the value function given by :math:`V_{\phi}(s)` or :math:`V_{\phi}(s')`, depending on the flag.
    """

    def __init__(self, value, flag=0):
        r"""
        Initialize the V-target.

        Args:
            value (Value): state value function
            flag (int): If flag=0, the value function is evaluated on the 'states' :math:`s`, while if flag=1, it is
                evaluated on the 'next_states' :math:`s'`.
        """
        super(VTarget, self).__init__()
        if not isinstance(value, Value):
            raise TypeError("Expecting the given value to be an instance of 'Value', instead got: "
                            "{}".format(type(value)))
        self._value = value
        self._flag = flag % 2

    def _evaluate(self, batch):
        r"""
        Compute :math:`V(s)`.

        Args:
            batch (Batch): batch containing the transitions.

        Returns:
            torch.Tensor: evaluated targets.
        """
        if self._flag == 0:
            return self._value(batch['states'])
        return self._value(batch['next_states'])


class QTarget(Target):
    r"""Value Target.

    Evaluate the value function given by :math:`Q_{\phi}(s, a)` or :math:`Q_{\phi}(s', \pi(s'))`, depending if the
    policy is given or not.
    """

    def __init__(self, q_value, policy=None):
        r"""
        Initialize the Q-target.

        Args:
            q_value (Value): state value function
            policy (Policy): policy.
        """
        super(QTarget, self).__init__()

        # check given value function
        if not isinstance(q_value, Value):
            raise TypeError("Expecting the given value to be an instance of 'Value', instead got: "
                            "{}".format(type(q_value)))
        self._qvalue = q_value

        # check given policy
        if policy is not None and not isinstance(policy, Policy):
            raise TypeError("Expecting the given policy to be None, or an instance of 'Policy', instead got: "
                            "{}".format(type(policy)))
        self._policy = policy

    def _evaluate(self, batch):
        r"""
        Compute :math:`Q(s, a)` or :math:`Q(s', \pi(s'))`.

        Args:
            batch (Batch): batch containing the transitions.

        Returns:
            torch.Tensor: evaluated targets.
        """
        if self._policy is None:
            return self._qvalue(batch['states'], batch['actions'])
        actions = self._policy.predict(batch['states'])
        return self._qvalue(batch['next_states'], actions)


class PolicyTarget(Target):  # TODO
    r"""Policy target.

    Compute the log-likelihood on the action :math:`a` returned by the policy :math:`\pi(s)` or :math:`\pi(s')`
    depending on the flag. That is, it computes :math:`\log \pi(a|s)` or :math:`\log \pi(`
    """

    def __init__(self, policy, flag=0):
        """
        Initialize the policy target.

        Args:
            policy (Exploration): wrapped policy with an exploration strategy.
            flag (int): If flag=0, the policy is evaluated on the 'states' :math:`s`, while if flag=1, it is
                evaluated on the 'next_states' :math:`s'`.
        """
        super(PolicyTarget, self).__init__()

        # check given policy
        if not isinstance(policy, Exploration):
            raise TypeError("Expecting the given policy to be None, or an instance of 'Exploration', instead got: "
                            "{}".format(type(policy)))
        self._policy = policy

        self._flag = flag % 2

    def _evaluate(self, batch):
        """
        Compute the log-likelihood on the action :math:`a` returned by the policy.

        Args:
            batch (Batch): batch containing the transitions.

        Returns:
            torch.Tensor: evaluated targets.
        """
        if self._flag == 0:
            return self._policy.predict(batch['states'])
        return self._policy.predict(batch['next_states'])


class ValueTarget(GammaTarget):
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

    def _evaluate(self, batch):
        r"""
        Compute the value target :math:`(r + \gamma (1-d) \min_i V_{\phi_i}(s'))`

        Args:
            batch (Batch): batch containing the transitions.

        Returns:
            torch.Tensor: evaluated targets.
        """
        value = torch.min(torch.cat([value(batch['next_states']) for value in self._values], dim=1), dim=1)[0]
        return batch['rewards'] + self.gamma * batch['masks'] * value


class QValueTarget(GammaTarget):
    r"""Q-Value target.

    Compute the Q-value target given by :math:`(r + \gamma (1-d) \min_i Q_{\phi_i}(s',a'))`, where the index `i` is in
    the case there are multiple Q-value function approximators given to this class.
    """

    def __init__(self, q_values, policy, gamma=1.):
        """
        Initialize the state value target.

        Args:
            q_values (QValue, list of QValue): state-action value function(s).
            policy (Policy): policy.
            gamma (float): discount factor
        """
        super(QValueTarget, self).__init__(gamma)

        # check Q-values
        if not isinstance(q_values, collections.Iterable):
            q_values = [q_values]
        for i, value in enumerate(q_values):
            if not isinstance(value, QValue):
                raise TypeError('The {}th value is not an instance of `QValue`, instead got: {}'.format(i, type(value)))
        self._q_values = q_values

        # check policy
        if not isinstance(policy, Policy):
            raise TypeError("Expecting the policy to be an instance of `Policy`, instead got: {}".format(type(policy)))
        self._policy = policy

    def _evaluate(self, batch):
        r"""
        Compute the value target :math:`(r + \gamma (1-d) \min_i Q_{\phi_i}(s',a'))`.

        Args:
            batch (Batch): batch containing the transitions.

        Returns:
            torch.Tensor: evaluated targets.
        """
        next_states = batch['next_states']
        actions = self._policy.predict(next_states)
        self.actions = actions
        value = torch.min(torch.cat([value(next_states, actions) for value in self._q_values], dim=1), dim=1)[0]
        return batch['rewards'] + self.gamma * batch['masks'] * value


class QLearningTarget(GammaTarget):
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

    def _evaluate(self, batch):
        r"""
        Compute the value target :math:`(r + \gamma (1-d) \min_i \max_{a'} Q_{\phi_i}(s',a'))`.

        Args:
            batch (Batch): batch containing the transitions.

        Returns:
            torch.Tensor: evaluated targets.
        """
        next_states = batch['next_states']
        q_max = [torch.max(q_value(next_states), dim=1, keepdim=True)[0] for q_value in self._q_values]
        value = torch.min(torch.cat(q_max, dim=1), dim=1)[0]
        return batch['rewards'] + self.gamma * batch['masks'] * value


class EntropyValueTarget(Target):
    r"""Entropy regularized value target.

    Compute the entropy regularized Q-value target given by
    :math:`min_i Q_{\phi_i}(s, \tilde{a}) - \alpha \log \pi_\theta(\tilde{a}|s)`, where
    :math:`\tilde{a} \sim \pi_\theta(.|s)`.
    """

    def __init__(self, q_values, policy, alpha=0.2):
        """
        Initialize the entropy value target.

        Args:
            q_values (QValue, list of QValue): state-action value funtion(s).
            policy (Exploration): wrapped policy with an exploration strategy.
            alpha (float): entropy regularization coefficient which controls the tradeoff between exploration and
                exploitation. Higher :attr:`alpha` means more exploration, and lower :attr:`alpha` corresponds to more
                exploitation.
        """
        super(EntropyValueTarget, self).__init__()

        # check Q-values
        if not isinstance(q_values, collections.Iterable):
            q_values = [q_values]
        for i, value in enumerate(q_values):
            if not isinstance(value, QValue):
                raise TypeError('The {}th value is not an instance of `QValue`, instead got: {}'.format(i, type(value)))

        # check policy
        if not isinstance(policy, Exploration):
            raise TypeError(
                "Expecting the policy to be an instance of `Exploration`, instead got: {}".format(type(policy)))
        self._policy = policy

        self._alpha = float(alpha)

    def _evaluate(self, batch):
        r"""
        Compute the entropy regularized Q-value target
        :math:`min_i Q_{\phi_i}(s, \tilde{a}) - \alpha \log \pi_\theta(\tilde{a}|s)`, where
        :math:`\tilde{a} \sim \pi_\theta(.|s)`.

        Args:
            batch (Batch): batch containing the transitions.

        Returns:
            torch.Tensor: evaluated targets.
        """
        actions, distribution = self._policy.predict(batch['states'])
        self.actions, self.distribution = actions, distribution
        value = torch.min(torch.cat([value(batch['states'], actions) for value in self._q_values], dim=1), dim=1)[0]
        return value - self._alpha * distribution.log_prob(actions)
