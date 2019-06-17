#!/usr/bin/env python
"""Defines the metrics used in transfer learning (TL).
"""

import matplotlib.pyplot as plt

from pyrobolearn.tasks import TLTask
from pyrobolearn.metrics import Metric

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
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
        [1] "A Survey on Transfer Learning", Pan et al., 2010
        [2] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self):
        super(TLMetric, self).__init__()
