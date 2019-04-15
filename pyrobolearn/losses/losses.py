#!/usr/bin/env python
"""Defines the common loss functions that are used by the learning algorithm / optimizer.

Losses are evaluated on model parameters, data batches / storages, or transitions tuples.
"""

import torch

from pyrobolearn.losses.loss import Loss

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FixedLoss(Loss):
    r"""Fixed Loss

    """

    def __init__(self, value):
        super(FixedLoss, self).__init__()
        self.value = torch.tensor(value)

    def compute(self, batch):
        return self.value


class L2Loss(Loss):
    r"""L2 Loss

    Compute the L2 loss given by: :math:`1/2 * (y_{target} - y_{predict})^2`
    """

    def __init__(self, target, predictor):
        super(L2Loss, self).__init__()
        self._target = target
        self._predictor = predictor

    def compute(self, batch):
        if self._target in batch:
            target = batch[self._target]
        else:
            target = self._target(batch)
        if self._predictor in batch:
            output = batch[self._predictor]
        else:
            output = self._predictor(batch)
        return 0.5 * (target - output).pow(2).mean()


class HuberLoss(Loss):
    r"""Huber Loss

    "In statistics, the Huber loss is a loss function used in robust regression, that is less sensitive to outliers
    in data than the squared error loss." [1]

    This loss is given by [1]:

    .. math:: {\mathcal L}(\delta) = \left\{ \begin{array}{lc} 1/2 a^2 & for |a| \leq \delta, \\
    \delta (|a| - 1/2 \delta), & \mbox{otherwise} \left \end{array}

    "This function is quadratic for small values of :math:`a`, and linear for large values, with equal values and
    slopes of the different sections at the two points where :math:`|a| = \delta`" [1].

    In [2], this loss is used for DQN, where :math:`\delta=1`, and :math:`a` is the temporal difference error, that is,
    :math:`a = Q(s,a) - (r + \gamma \max_a Q(s',a))` where :math:`(r + \gamma \max_a Q(s',a))` is the target function.

    References:
        [1] Huber Loss (on Wikipedia): https://en.wikipedia.org/wiki/Huber_loss
        [2] "Reinforcement Learning (DQN) Tutorial":
            https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    def __init__(self, loss, delta=1.):
        super(HuberLoss, self).__init__()
        self.loss = loss
        self.delta = delta

    def compute(self, batch):
        a = self.loss(batch)
        if abs(a) <= self.delta:
            return 0.5 * torch.pow(a, 2)
        return self.delta * (torch.abs(a) - 0.5 * self.delta)


class KLLoss(Loss):
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

    def compute(self, batch):
        """
        Compute :math:`KL(p||q)`.
        """
        # TODO use the batch
        return torch.distributions.kl.kl_divergence(self.p, self.q)

    def latex(self):
        return r"\mathbb{E}[ KL( p || q ) ]"


# class ForwardKLPenaltyLoss(KLPenaltyLoss):
#     r"""Forward KL Penalty Loss"""
#     pass
#
#
# class ReverseKLPenaltyLoss(KLPenaltyLoss):
#     r"""Reverse KL Penalty Loss"""
#     pass


class HLoss(Loss):
    r"""Entropy Loss

    Entropy loss of a distribution:

    .. math:: L^{Entropy}(\theta) = H[ p ]

    where :math:`H[.]` is the Shannon entropy of the given probability distribution.
    """

    def __init__(self, distribution):
        super(HLoss, self).__init__()
        self.p = distribution

    def compute(self, batch):
        entropy = self.p.entropy().mean()
        return entropy


# Tests
if __name__ == '__main__':
    # compute the losses
    loss = - FixedLoss(3)  # + FixedLoss(2)
    print(loss(2))
