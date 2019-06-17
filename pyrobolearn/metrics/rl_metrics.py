#!/usr/bin/env python
"""Defines the metrics used in reinforcement learning (RL).
"""

import matplotlib.pyplot as plt

from pyrobolearn.tasks import RLTask
from pyrobolearn.algos import RLAlgo
from pyrobolearn.metrics import Metric

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

    def __init__(self, task, gamma=1.):
        """
        Initialize the average return metric.

        Args:
            gamma (float): discount factor
        """
        super(AverageReturnMetric, self).__init__()
        self.gamma = gamma
        self.task = task
        self.returns = []

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

    def update(self):
        pass

    def _plot(self, ax=None, filename=None):
        """
        Plot the average return metric.

        Args:
            ax (plt.Axes): axis to plot the figure.
            filename (str, None): if a string is given, it will save the plot in the given filename.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.set_title('Average Return per iteration')  # per epoch, per iteration=epoch*batch
        ax.set_xlabel('iterations')
        ax.set_ylabel('Average return')
        ax.plot(self.returns)

        return ax


class LossMetric(RLMetric):
    r"""Loss Metric
    """

    def __init__(self, loss):
        super(LossMetric, self).__init__()
        self.loss = loss

    def update(self):
        pass
