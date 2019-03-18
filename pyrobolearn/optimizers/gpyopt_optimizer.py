#!/usr/bin/env python
"""Provide a wrapper around the GPyOpt optimizers for Bayesian Optimization (BO).

References:
    [1] https://sheffieldml.github.io/GPyOpt/
"""

import numpy as np

# Bayesian optimization
try:
    import GPy
    import GPyOpt
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install GPy/GPyOpt directly via 'pip install GPy' and "
                                    "'pip install GPyOpt'.")

from optimizer import Optimizer

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: check the `pyrobolearn/algos/bo` algo
class BayesianOptimizer(object):
    pass
