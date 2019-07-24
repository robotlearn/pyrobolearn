#!/usr/bin/env python
"""Defines the metrics used in transfer learning (TL).

References:
    - [1] "A Survey on Transfer Learning", Pan et al., 2010
    - [2] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
"""

import matplotlib.pyplot as plt

from pyrobolearn.tasks import TLTask
from pyrobolearn.metrics import Metric

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TLMetric(Metric):
    r"""Transfer Learning Metric

    Metrics used in transfer learning.

    References:
        - [1] "A Survey on Transfer Learning", Pan et al., 2010
        - [2] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self):
        """
        Initialize the transfer learning metric.
        """
        super(TLMetric, self).__init__()


class JumpstartMetric(TLMetric):
    r"""Jumpstart metric

    The jumpstart metric measures how much the initial performance of an agent in a target task may be improved by
    transferring knowledge from a source task. [1]

    References:
        - [1] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self):
        """
        Initialize the jumpstart metric.
        """
        super(JumpstartMetric, self).__init__()


class AsymptoticPerformance(TLMetric):
    r"""Asymptotic performance metric

    The asymptotic performance metric measures how much the final learned performance of an agent in the target task
    has improved via transfer. [1]

    References:
        - [1] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self):
        """
        Initialize the asymptotic performance metric.
        """
        super(AsymptoticPerformance, self).__init__()


class TotalRewardMetric(TLMetric):
    r"""Total reward metric

    The total reward metric measures the total reward accumulated by a learning agent. This one may be improved if
    knowledge transfer was used. [1]

    References:
        - [1] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self):
        """
        Initialize the total reward metric.
        """
        super(TotalRewardMetric, self).__init__()


class TransferRatioMetric(TLMetric):
    r"""Transfer ratio metric

    The transfer ratio metric measures "the ratio of the total reward accumulated by the transfer learner and the total
    reward accumulated by the non-transfer learner". [1]

    References:
        - [1] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self):
        """
        Initialize the transfer ratio metric.
        """
        super(TransferRatioMetric, self).__init__()


class TimeToThresholdMetric(TLMetric):
    r"""Time to threshold metric

    The time to threshold metric measures how much the learning time needed by the agent to perform a pre-specified
    performance level (i.e. the threshold) is reduced via knowledge transfer. [1]

    References:
        - [1] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self, threshold):
        """
        Initialize the time to threshold metric.

        Args:
            threshold (float): threshold performance level.
        """
        super(TimeToThresholdMetric, self).__init__()
        self._threshold = threshold
