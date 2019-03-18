#!/usr/bin/env python
"""Provide a wrapper around the Covariance Adaptation Matrix (CMA) which uses an evolution strategy to optimize
the parameters.

References:
    [1] https://github.com/CMA-ES/pycma
"""

import numpy as np

# CMA-ES
try:
    import cma
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install CMA-ES or `pycma` directly via 'pip install cma'.")

from optimizer import Optimizer

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: check the `pyrobolearn/algos/cmaes` algo
class CMAES(object):
    pass
