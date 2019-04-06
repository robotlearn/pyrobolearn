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
        return 0.5 * (self.target(batch) - self.approximator(batch)).pow(2).mean()


class ValueLoss(Loss):
    r"""L2 loss for values
    """
    def __init__(self):
        super(ValueLoss, self).__init__()

    def compute(self, batch):
        returns = batch['returns']
        values = batch.current['values']
        return 0.5 * (returns - values).pow(2).mean()


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


class MSBELoss(Loss):
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
        [1] https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        [2] "Reinforcement Learning: An Introduction", Sutton and Barto, 2018
    """

    def __init__(self):
        super(MSBELoss, self).__init__()

    def compute(self, batch):
        pass


# Tests
if __name__ == '__main__':
    # compute the losses
    loss = - FixedLoss(3)  # + FixedLoss(2)
    print(loss(2))
