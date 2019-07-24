#!/usr/bin/env python
"""Defines the metrics used in reinforcement learning (RL).
"""

import numpy as np
import matplotlib.pyplot as plt

from pyrobolearn.tasks import RLTask
from pyrobolearn.algos import RLAlgo
from pyrobolearn.metrics import Metric
from pyrobolearn.losses import BatchLoss

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RLMetric(Metric):
    r"""Reinforcement Learning (abstract) Metric

    Metrics used in reinforcement learning.

    References:
        [1] "Deep Reinforcement Learning that Matters", Henderson et al., 2018
    """

    def __init__(self):
        """
        Initialize the RL metric.
        """
        super(RLMetric, self).__init__()


class AverageReturnMetric(RLMetric):
    r"""Average / Expected return metric.

    This computes the average / expected RL return given by:

    .. math:: J(\pi_{\theta}) = \mathcal{E}_{\tau \sim \pi_{\theta}}[ R(\tau) ]

    where :math:`R(\tau) = \sum_{t=0}^T \gamma^t r_t` is the discounted return.
    """

    def __init__(self, task, gamma=1., num_episodes=10, num_steps=100):
        """
        Initialize the average return metric.

        Args:
            gamma (float): discount factor
        """
        super(AverageReturnMetric, self).__init__()
        self.gamma = gamma
        self.task = task
        self.returns = []
        self._num_steps = num_steps
        self._num_episodes = num_episodes

    ##############
    # Properties #
    ##############

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

    @property
    def task(self):
        """Return the RL task."""
        return self._task

    @task.setter
    def task(self, task):
        """Set the RL task."""
        if not isinstance(task, RLTask):
            raise TypeError("Expecting the given 'task' to be an instance of `RLTask`, instead got: "
                            "{}".format(type(task)))
        self._task = task

    ###########
    # Methods #
    ###########

    def _episode_update(self, episode_idx=None):
        """Update the metric."""
        rewards = []
        for ep in range(self._num_episodes):
            reward = self.task.run(num_steps=self._num_steps)
            rewards.append(reward)

        rewards = np.asarray(rewards).mean()
        self.returns.append(rewards)

    def _plot(self, ax):
        """
        Plot the average return metric in the given axis.
        """
        ax.set_title('Average Return per iteration')  # per epoch, per iteration=epoch*batch
        ax.set_xlabel('iterations')
        ax.set_ylabel('Average return')
        ax.plot(self.returns)


class LossMetric(RLMetric):
    r"""Loss Metric
    """

    def __init__(self, loss):
        """
        Initialize the loss metric.

        Args:
            loss (BatchLoss): batch loss.
        """
        super(LossMetric, self).__init__()
        self.loss = loss
        self.losses = []

    ##############
    # Properties #
    ##############

    @property
    def loss(self):
        """Return the loss instance."""
        return self._loss

    @loss.setter
    def loss(self, loss):
        """Set the loss instance."""
        if not isinstance(loss, BatchLoss):
            raise TypeError("Expecting the given 'loss' to be an instance of `BatchLoss`, but got instead: "
                            "{}".format(type(loss)))
        self._loss = loss

    ###########
    # Methods #
    ###########

    def update(self):
        pass

    def _plot(self, ax):
        """
        Plot the loss in the given axis.
        """
        ax.set_title(self.loss.__class__.__name__ + ' per iteration')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
