#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the common loss functions that are used by the learning algorithm / optimizer.

Losses are evaluated on model parameters, data batches / storages, or transitions tuples.
"""

import torch

from pyrobolearn.losses.loss import Loss
from pyrobolearn.storages import Batch


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BatchLoss(Loss):
    r"""Loss evaluated on a batch.
    """

    def __init__(self):
        super(BatchLoss, self).__init__()

        # cache the last value that was computed by the loss
        self.value = None

    def _compute(self, batch):
        """Compute the loss on the given batch. This method has to be implemented in the child classes."""
        raise NotImplementedError

    def compute(self, batch):
        """
        Compute the loss on the given batch.

        Args:
            batch (Batch): batch to evaluate the loss on.

        Returns:
            torch.Tensor: scalar loss value.
        """
        # check that we are given a batch
        if not isinstance(batch, Batch):
            raise TypeError("Expecting the given 'batch' to be an instance of `Batch`, instead got: "
                            "{}".format(type(batch)))
        self.value = self._compute(batch)
        return self.value


class FixedLoss(BatchLoss):
    r"""Fixed Loss

    This is a dummy loss that returned always the same values given initially.
    """

    def __init__(self, value):
        """
        Initialize the fixed loss.

        Args:
            value (torch.Tensor, float, int, np.array, list): fixed initial values that will be returned at each call.
        """
        super(FixedLoss, self).__init__()
        self.value = torch.tensor(value)

    def _compute(self, batch):
        """
        Compute the fixed loss.

        Args:
            batch (Batch): batch containing the states, actions, rewards, etc.

        Returns:
            torch.Tensor: loss scalar value
        """
        return self.value


class L2Loss(BatchLoss):
    r"""L2 Loss

    Compute the L2 loss given by: :math:`1/2 * (y_{target} - y_{predict})^2`
    """

    def __init__(self, target, predictor):
        """
        Initialize the L2 loss.

        Args:
            target (callable): callable target that accepts a Batch instance as input. If it is not in the given batch,
                it will give the batch to it.
            predictor (callable): callable predictor that accepts a Batch instance as input. If it is not in the given
                batch, it will give the batch to it.
        """
        super(L2Loss, self).__init__()
        self._target = target
        self._predictor = predictor

    def _compute(self, batch):
        r"""
        Compute the L2 loss: :math:`1/2 * (y_{target} - y_{predict})^2`.

        Args:
            batch (Batch): batch containing the states, actions, rewards, etc.

        Returns:
            torch.Tensor: loss scalar value
        """
        # get target data
        if self._target in batch.current:
            target = batch.current[self._target]
        elif self._target in batch:
            target = batch[self._target]
        else:
            target = self._target(batch)

        # get predicted data
        if self._predictor in batch.current:
            output = batch.current[self._predictor]
        elif self._predictor in batch:
            output = batch[self._predictor]
        else:
            output = self._predictor(batch)

        # compute L2 loss
        return 0.5 * (target - output).pow(2).mean()


class HuberLoss(BatchLoss):
    r"""Huber Loss

    "In statistics, the Huber loss is a loss function used in robust regression, that is less sensitive to outliers
    in data than the squared error loss." [1]

    This loss is given by [1]:

    .. math:: {\mathcal L}_{\delta}(a) = \left\{ \begin{array}{lc} 1/2 a^2 & for |a| \leq \delta, \\
    \delta (|a| - 1/2 \delta), & \mbox{otherwise} \left \end{array}

    "This function is quadratic for small values of :math:`a`, and linear for large values, with equal values and
    slopes of the different sections at the two points where :math:`|a| = \delta`" [1].

    In [2], this loss is used for DQN, where :math:`\delta=1`, and :math:`a` is the temporal difference error, that is,
    :math:`a = Q(s,a) - (r + \gamma \max_a Q(s',a))` where :math:`(r + \gamma \max_a Q(s',a))` is the target function.

    References:
        - [1] Huber Loss (on Wikipedia): https://en.wikipedia.org/wiki/Huber_loss
        - [2] "Reinforcement Learning (DQN) Tutorial":
          https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, loss, delta=1.):
        """
        Initialize the Huber loss.

        Args:
            loss (Loss): initial loss to smooth.
            delta (float): coefficient
        """
        super(HuberLoss, self).__init__()
        self.loss = loss
        self.delta = delta

    def _compute(self, batch):
        r"""
        Compute the Huber loss.

        Args:
            batch (Batch): batch containing the states, actions, rewards, etc.

        Returns:
            torch.Tensor: loss scalar value
        """
        a = self.loss(batch)
        if abs(a) <= self.delta:
            return 0.5 * torch.pow(a, 2)
        return self.delta * (torch.abs(a) - 0.5 * self.delta)


class PseudoHuberLoss(BatchLoss):
    r"""Pseudo-Huber Loss

    "The Pseudo-Huber loss function can be used as a smooth approximation of the Huber loss function. It combines the
    best properties of L2 squared loss and L1 absolute loss by being strongly convex when close to the target/minimum
    and less steep for extreme values. This steepness can be controlled by the :math:`\delta` value. The Pseudo-Huber
    loss function ensures that derivatives are continuous for all degrees. It is defined as:

    .. math:: {\mathcal L}_{\delta}(a) = \delta^2 \left( \sqrt{1 + (a/\delta)^2} - 1 \right)

    As such, this function approximates :math:`a^2/2` for small values of :math:`a`, and approximates a straight line
    with slope :math:`\delta` for large values of :math:`a`.

    While the above is the most common form, other smooth approximations of the Huber loss function also exist." [1]

    References:
        - [1] Huber Loss (on Wikipedia): https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
    """

    def __init__(self, loss, delta=1.):
        """
        Initialize the Pseudo-Huber loss.

        Args:
            loss (Loss): initial loss to smooth.
            delta (float): steepness coefficient
        """
        super(PseudoHuberLoss, self).__init__()
        self.loss = loss
        self.delta = delta

    def _compute(self, batch):
        r"""
        Compute the pseudo Huber loss.

        Args:
            batch (Batch): batch containing the states, actions, rewards, etc.

        Returns:
            torch.Tensor: loss scalar value
        """
        a = self.loss(batch)
        return self.delta**2 * (torch.sqrt(1 + (a/self.delta)**2) - 1)


class KLLoss(BatchLoss):
    r"""KL Penalty Loss

    KL Penalty to minimize:

    .. math:: L^{KL}(\theta) = \mathbb{E}[ KL( p || q) ]

    where :math:`KL(.||.)` is the KL-divergence between two probability distributions.
    """

    def __init__(self, p, q):
        """
        Initialize the KL Penalty loss.

        Args:
            p (torch.distributions.Distribution): 1st distribution
            q (torch.distributions.Distribution): 2nd distribution
        """
        super(KLLoss, self).__init__()
        self.p = p
        self.q = q

    def _compute(self, batch):
        """
        Compute :math:`KL(p||q)`.

        Args:
            batch (Batch): batch containing the states, actions, rewards, etc.

        Returns:
            torch.Tensor: loss scalar value
        """
        # TODO use the batch
        return torch.distributions.kl.kl_divergence(self.p, self.q)

    def latex(self):
        """Return a latex formula of the loss."""
        return r"\mathbb{E}[ KL( p || q ) ]"


# class ForwardKLPenaltyLoss(KLPenaltyLoss):
#     r"""Forward KL Penalty Loss"""
#     pass
#
#
# class ReverseKLPenaltyLoss(KLPenaltyLoss):
#     r"""Reverse KL Penalty Loss"""
#     pass


class HLoss(BatchLoss):
    r"""Entropy Loss

    Entropy loss of a distribution:

    .. math:: L^{Entropy}(\theta) = H[ p ]

    where :math:`H[.]` is the Shannon entropy of the given probability distribution.
    """

    def __init__(self, distribution):
        """
        Initialize the entropy loss.

        Args:
            distribution (torch.distributions.Distribution): probability distribution.
        """
        super(HLoss, self).__init__()
        if not isinstance(distribution, torch.distributions.Distribution):
            raise TypeError("Expecting the given distribution to be an instance of `torch.distributions.Distribution`, "
                            "instead got: {}".format(type(distribution)))
        self.p = distribution

    def _compute(self, batch):
        """
        Compute the entropy loss.

        Args:
            batch (Batch): batch containing the states, actions, rewards, etc.

        Returns:
            torch.Tensor: loss scalar value
        """
        entropy = self.p.entropy().mean()
        return entropy


# Tests
if __name__ == '__main__':
    # compute the losses
    loss = - FixedLoss(3)  # + FixedLoss(2)
    print(loss(2))
