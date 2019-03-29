#!/usr/bin/env python
"""Defines the common loss functions that are used by the learning algorithm / optimizer.

Losses are evaluated on model parameters, data batches, and / or storages.

TODO: loss vs cost vs reward
costs and rewards are defined possibly for each time steps.
"""

from abc import ABCMeta, abstractmethod
import operator
import copy
import collections

import torch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Loss(object):
    r"""Loss abstract class.

    Each loss is minimized.
    """
    __metaclass__ = ABCMeta

    def __init__(self, losses=None):
        self.losses = losses

    ##############
    # Properties #
    ##############

    @property
    def losses(self):
        """Return the inner losses."""
        return self._losses

    @losses.setter
    def losses(self, losses):
        """Set the inner losses."""
        if losses is None:
            losses = []
        elif isinstance(losses, collections.Iterable):
            for loss in losses:
                if not isinstance(loss, Loss):
                    raise TypeError("Expecting a Loss instance for each item in the iterator.")
        else:
            if not isinstance(losses, Loss):
                raise TypeError("Expecting losses to be an instance of Loss.")
            losses = [losses]
        self._losses = losses

    ###########
    # Methods #
    ###########

    def has_losses(self):
        """Check if it has losses."""
        return len(self._losses) > 0

    def compute(self, *args, **kwargs):
        """Compute the loss and return the scalar value."""
        pass

    def latex(self):
        """Return a latex formula of the loss."""
        pass

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string."""
        if not self.losses or self.losses is None:
            return self.__class__.__name__
        else:
            lst = [loss.__repr__() for loss in self.losses]
            return ' + '.join(lst)

    def __call__(self, *args, **kwargs):
        """Compute the loss."""
        return self.compute(*args, **kwargs)

    def __build_loss(self, other):
        # built the internal list of losses
        losses = self._losses if self.has_losses() else [self]
        if isinstance(other, Loss):
            if other.has_losses():
                losses.extend(other._losses)
            else:
                losses.append(other)
        return Loss(losses=losses)

    def __get_operation(self, other, op):
        if isinstance(other, Loss):  # callable
            def compute(*args, **kwargs):
                return op(self(*args, **kwargs), other(*args, **kwargs))
        else:
            def compute(*args, **kwargs):
                return op(self(*args, **kwargs), other)
        return compute

    def __add__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__add__)
        return loss

    def __div__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__div__)
        return loss

    def __floordiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__floordiv__)
        return loss

    def __iadd__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__iadd__)
        return loss

    def __idiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__idiv__)
        return loss

    def __ifloordiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__ifloordiv__)
        return loss

    def __imod__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__imod__)
        return loss

    def __imul__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__imul__)
        return loss

    def __ipow__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__ipow__)
        return loss

    def __isub__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__isub__)
        return loss

    def __itruediv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__itruediv__)
        return loss

    def __mod__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__mod__)
        return loss

    def __mul__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__mul__)
        return loss

    def __pow__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__pow__)
        return loss

    def __radd__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__add__)
        return loss

    def __rdiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__div__)
        return loss

    def __rfloordiv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__floordiv__)
        return loss

    def __rmod__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__mod__)
        return loss

    def __rmul__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__mul__)
        return loss

    def __rpow__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__pow__)
        return loss

    def __rsub__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__sub__)
        return loss

    def __rtruediv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__truediv__)
        return loss

    def __sub__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__sub__)
        return loss

    def __truediv__(self, other):
        loss = self.__build_loss(other)
        loss.compute = self.__get_operation(other, operator.__truediv__)
        return loss

    # binary comparison operators
    def __eq__(self, other):
        compute = self.__get_operation(other, operator.__eq__)
        return compute()

    def __ge__(self, other):
        compute = self.__get_operation(other, operator.__ge__)
        return compute()

    def __gt__(self, other):
        compute = self.__get_operation(other, operator.__gt__)
        return compute()

    def __le__(self, other):
        compute = self.__get_operation(other, operator.__le__)
        return compute()

    def __lt__(self, other):
        compute = self.__get_operation(other, operator.__lt__)
        return compute()

    def __ne__(self, other):
        compute = self.__get_operation(other, operator.__ne__)
        return compute()

    # unary operators
    def __abs__(self):
        loss = copy.copy(self)  # shallow copy

        def compute(*args, **kwargs):
            return operator.__abs__(self(*args, **kwargs))

        loss.compute = compute
        return loss

    def __neg__(self):
        loss = copy.copy(self)  # shallow copy

        def compute(*args, **kwargs):
            return operator.__neg__(self(*args, **kwargs))

        loss.compute = compute
        return loss

    def __pos__(self):
        loss = copy.copy(self)  # shallow copy

        def compute(*args, **kwargs):
            return operator.__pos__(self(*args, **kwargs))

        loss.compute = compute
        return loss


class FixedLoss(Loss):
    r"""Fixed Loss

    """
    def __init__(self, value):
        super(FixedLoss, self).__init__()
        self.value = value

    def compute(self, batch):
        return self.value * batch


class L2Loss(Loss):
    r"""L2 Loss

    Compute the L2 loss given by: :math:`1/2 * (y_{target} - y_{predict})^2`
    """
    def __init__(self, target, approximator):
        super(L2Loss, self).__init__()
        self.target = target
        self.approximator = approximator

    def compute(self, batch):
        # based on approximator check what we need
        return 0.5 * (self.target(arg) - self.approximator(arg)).pow(2).mean()


class ValueLoss(Loss):
    r"""L2 loss for values
    """
    def __init__(self):
        super(ValueLoss, self).__init__()

    def compute(self, batch):
        returns = batch['returns']
        values = batch.current['values']
        return 0.5 * (returns - values).pow(2).mean()


class PGLoss(Loss):
    r"""Policy Gradient Loss

    Compute the policy gradient loss which is maximized and given by:

    .. math:: L^{PG} = \mathbb{E}[ \log \pi_{\theta}(a_t | s_t) \psi_t ]

     where :math:`\psi_t` is the associated return estimator, which can be for instance, the total reward estimator
     :math:`\psi_t = R(\tau)` (where :math:`\tau` represents the whole trajectory), the state action value estimator
     :math:`\psi_t = Q(s_t, a_t)`, or the advantage estimator :math:`\psi_t = A_t = Q(s_t, a_t) - V(s_t)`. Other
     estimators are also possible.

    The gradient with respect to the parameters :math:`\theta` is then given by:

    .. math:: g = \mathbb{E}[ \nabla_\theta \log \pi_{\theta}(a_t | s_t)

    References:
        [1] "Proximal Policy Optimization Algorithms", Schulman et al., 2017
        [2] "High-Dimensional Continuous Control using Generalized Advantage Estimation", Schulman et al., 2016
    """

    def __init__(self):
        super(PGLoss, self).__init__()

    def compute(self, batch):
        log_curr_pi = batch.current['action_distributions']
        log_curr_pi = log_curr_pi.log_probs(batch.current['actions'])
        estimator = batch['estimator']
        loss = torch.exp(log_curr_pi) * estimator
        return -loss.mean()

    def latex(self):
        return "\\mathbb{E}[ r_t(\\theta) A_t ]"


class CPILoss(Loss):
    r"""CPI Loss

    Conservative Policy Iteration objective which is maximized and defined in [1]:

    .. math:: L^{CPI}(\theta) = \mathbb{E}[ r_t(\theta) A_t ]

    where the expectation is taken over a finite batch of samples, :math:`A_t` is an estimator of the advantage fct at
    time step :math:`t`, :math:`r_t(\theta)` is the probability ratio given by
    :math:`r_t(\theta) = \frac{ \pi_{\theta}(a_t|s_t) }{ \pi_{\theta_{old}}(a_t|s_t) }`.

    References:
        [1] "Approximately optimal approximate reinforcement learning", Kakade et al., 2002
        [2] "Proximal Policy Optimization Algorithms", Schulman et al., 2017
    """

    def __init__(self):
        """
        Initialize the CPI Loss.
        """
        super(CPILoss, self).__init__()

    def compute(self, batch):  # policy_distribution, old_policy_distribution, estimator):
        # ratio = policy_distribution / old_policy_distribution
        log_curr_pi = batch.current['action_distributions']
        log_curr_pi = log_curr_pi.log_probs(batch.current['actions'])
        log_prev_pi = batch['action_distributions']
        log_prev_pi = log_prev_pi.log_probs(batch['actions'])

        ratio = torch.exp(log_curr_pi - log_prev_pi)
        estimator = batch['estimator']

        loss = ratio * estimator
        return -loss.mean()

    def latex(self):
        return "\\mathbb{E}[ r_t(\\theta) A_t ]"


class CLIPLoss(Loss):
    r"""CLIP Loss

    Loss defined in [1] which is maximized and given by:

    .. math:: L^{CLIP}(\theta) = \mathbb{E}[ \min(r_t(\theta) A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) ]

    where the expectation is taken over a finite batch of samples, :math:`A_t` is an estimator of the advantage fct at
    time step :math:`t`, :math:`r_t(\theta)` is the probability ratio given by
    :math:`r_t(\theta) = \frac{ \pi_{\theta}(a_t|s_t) }{ \pi_{\theta_{old}}(a_t|s_t) }`.

    References:
        [1] "Proximal Policy Optimization Algorithms", Schulman et al., 2017
    """

    def __init__(self, clip=0.2):
        """
        Initialize the loss.

        Args:
            epsilon (float): clip parameter
        """
        super(CLIPLoss, self).__init__()
        self.eps = clip

    def compute(self, batch):  # , policy_distribution, old_policy_distribution, estimator):
        log_curr_pi = batch.current['action_distributions']
        log_curr_pi = log_curr_pi.log_probs(batch.current['actions'])
        log_prev_pi = batch['action_distributions']
        log_prev_pi = log_prev_pi.log_probs(batch['actions'])

        ratio = torch.exp(log_curr_pi - log_prev_pi)
        estimator = batch['estimator']

        loss = torch.min(ratio * estimator, torch.clamp(ratio, 1.0-self.eps, 1.0+self.eps) * estimator)
        return -loss.mean()

    def latex(self):
        return "\\mathbb{E}[ \\min(r_t(\\theta) A_t, clip(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) A_t) ]"


class KLPenaltyLoss(Loss):
    r"""KL Penalty Loss

    KL Penalty to minimize:

    .. math:: L^{KL}(\theta) = \mathbb{E}[ KL( \pi_{\theta_{old}}(a_t | s_t) || \pi_{\theta}(a_t | s_t) ) ]

    where :math:`KL(.||.)` is the KL-divergence between two probability distributions.
    """

    def __init__(self):  # p, q):
        """
        Initialize the KL Penalty loss.

        Args:
            p (torch.distributions.Distribution): 1st distribution
            q (torch.distributions.Distribution): 2nd distribution
        """
        super(KLPenaltyLoss, self).__init__()
        # self.p = p
        # self.q = q

    def compute(self, batch):
        """
        Compute :math:`KL(p||q)`.
        """
        curr_pi = batch.current['action_distributions']
        prev_pi = batch['action_distributions']

        return torch.distributions.kl.kl_divergence(prev_pi, curr_pi)

    def latex(self):
        return "\\mathbb{E}[ KL( \\pi_{\\theta_{old}}(a_t | s_t) || \\pi_{\\theta}(a_t | s_t) ) ]"


# class ForwardKLPenaltyLoss(KLPenaltyLoss):
#     r"""Forward KL Penalty Loss"""
#     pass
#
#
# class ReverseKLPenaltyLoss(KLPenaltyLoss):
#     r"""Reverse KL Penalty Loss"""
#     pass


class EntropyLoss(Loss):
    r"""Entropy Loss

    Entropy loss, which is used to ensure sufficient exploration when maximized [1,2,3]:

    .. math:: L^{Entropy}(\theta) = H[ \pi_{\theta} ]

    where :math:`H[.]` is the Shannon entropy of the given probability distribution.

    References:
        [1] "Simple Statistical Gradient-following Algorithms for Connectionist Reinforcement Learning", Williams, 1992
        [2] "Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016
        [3] "Proximal Policy Optimization Algorithms", Schulman et al., 2017
    """

    def __init__(self):  # approximator):
        super(EntropyLoss, self).__init__()

    def compute(self, batch):
        distribution = batch.current['action_distributions']
        entropy = distribution.entropy().mean()
        return entropy


# TODO: the functions defined here should be the same for `reward.py` and `loss.py`. Thus, we need to check numpy
# TODO: or torch inside the functions.
def ceil(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.ceil(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.ceil(x)


def cos(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.cos(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.cos(x)


def cosh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.cosh(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.cosh(x)


# def degrees(x):
#     if callable(x):
#         y = copy.copy(x)  # shallow copy
#         y.compute = lambda: torch.degrees(x())
#         return y
#     else:
#         return torch.degrees(x)


def exp(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.exp(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.exp(x)


def expm1(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.expm1(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.expm1(x)


def floor(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.floor(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.floor(x)


# def frexp(x):
#     if callable(x):
#         y = copy.copy(x)  # shallow copy
#         y.compute = lambda: torch.frexp(x())
#         return y
#     else:
#         return torch.frexp(x)


def isinf(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.isinf(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.isinf(x)


def isnan(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.isnan(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.isnan(x)


def log(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.log(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.log(x)


def log10(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.log10(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.log10(x)


def log1p(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.log1p(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.log1p(x)


# def modf(x):
#     if callable(x):
#         y = copy.copy(x)  # shallow copy
#         y.compute = lambda: torch.modf(x())
#         return y
#     else:
#         return torch.modf(x)


# def radians(x):
#     if callable(x):
#         y = copy.copy(x)  # shallow copy
#         y.compute = lambda: torch.radians(x())
#         return y
#     else:
#         return torch.radians(x)


def sin(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.sin(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.sin(x)


def sinh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.sinh(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.sinh(x)


def sqrt(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.sqrt(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.sqrt(x)


def tan(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.tan(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.tan(x)


def tanh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.tanh(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.tanh(x)


def trunc(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy

        def compute(*args, **kwargs):
            return torch.trunc(x(*args, **kwargs))

        y.compute = compute
        return y
    else:
        return torch.trunc(x)


# Tests
if __name__ == '__main__':
    # compute the losses
    loss = - FixedLoss(3)  # + FixedLoss(2)
    print(loss(2))
