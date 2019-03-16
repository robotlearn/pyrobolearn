#!/usr/bin/env python
"""Defines the various metrics used in different learning paradigms.

Dependencies:
- `pyrobolearn.tasks`
"""

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Metric(object):
    r"""Metric (abstract) class

    The metric class contains the various metrics used to evaluate a certain learning paradigm (e.g. imitation
    learning, reinforcement learning, transfer learning, active learning, and so on).

    It notably contains the functionalities to evaluate a certain task using the metric, and different to plot them.
    """

    def __init__(self):
        pass


class ILMetric(Metric):
    r"""Imitation Learning Metric

    Metrics used in imitation learning.

    References:
        [1] "Learning from Humans", Billard et al., 2016
    """

    def __init__(self):
        super(ILMetric, self).__init__()


class RLMetric(Metric):
    r"""Reinforcement Learning Metric

    Metrics used in reinforcement learning.

    References:
        [1] "Deep Reinforcement Learning that Matters", Henderson et al., 2018
    """

    def __init__(self):
        super(RLMetric, self).__init__()


class TLMetric(Metric):
    r"""Transfer Learning Metric

    Metrics used in transfer learning.

    References:
        [1] "A Survey on Transfer Learning", Pan et al., 2010
        [2] "Transfer Learning for Reinforcement Learning Domains: A Survey", Taylor et al., 2009
    """

    def __init__(self):
        super(TLMetric, self).__init__()
