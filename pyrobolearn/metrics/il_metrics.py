# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Defines the metrics used in imitation learning (IL).
"""

import matplotlib.pyplot as plt

from pyrobolearn.tasks import ILTask
from pyrobolearn.metrics import Metric

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ILMetric(Metric):
    r"""Imitation Learning Metric

    Metrics used in imitation learning.

    References:
        [1] "Learning from Humans", Billard et al., 2016
    """

    def __init__(self):
        super(ILMetric, self).__init__()
