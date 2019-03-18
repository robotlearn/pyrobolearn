#!/usr/bin/env python
"""Provide the abstract optimizer class.

Optimizers allows to optimize (i.e. minimize or maximize) a utility function (also known as objective function,
fitness function, loss, etc.) with or without constraints and bounds. Notably, it assumes the functions have some
parameters that the optimizer can update.

Mathematically, this is described as:

.. math:: \min_{x \in R^n} f(x)

subject to

.. math::

    g_i(x) \geq 0, \quad i = 1,...,m
    h_j(x)  =   0, \quad j = 1,...,p
    x_l \leq x \leq x_u

For trajectory optimization, check "An Introduction to Trajectory Optimization: How to do your own Direct Collocation".
"""

# TODO: trajectory optimization

from abc import ABCMeta, abstractmethod

# Numpy with autograd
# import autograd.numpy as np  # Thinly-wrapped numpy
# from autograd import grad    # The only autograd function you may ever need
import numpy as np


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Optimizer(object):
    r"""Optimizer abstract class

    This is an abstract class from which all optimizers inherit from. Most of the child optimizer classes are wrappers
    around the original optimizer. This is to provide the same common interface to all the optimizers, and convert
    seamlessly to the correct data types.

    Optimizers are often given as a parameter to the learning algorithms, but can also be used of out of the box
    directly on models. Several original optimizers can be given to the learning algorithm which will automatically
    wrap the optimizer with the corresponding wrapper to provide a common interface.

    In their most natural form, optimizers are used to ... optimization process which can be described mathematically
    by:

    .. math::

        \min_{\theta} J(\theta) \mbox{ subj. to constraints}

        \max_{\theta} J(\theta)

    Several optimizers provides

    The list of optimizers available are from the following libraries:
    * nlopt
    * ipopt
    * torch.optim
    * GPy / GPyOpt
    * scipy.optimize
    * qpsolvers (which includes cvxpy)
    * cmaes
    * pso

    Each one of them expect a certain kind of type of the parameters.

    Optimizers can be divided into:
    * global vs local
    * derivative-free vs gradient-based
    * without constraints vs with (equality and/or inequality) constraints

    Many implemented optimizers that can be found online are specific to a certain type of learning model.
    If necessary, a conversion or wrapping process is carried out to make the optimizer work with the given learning
    model.
    """
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):  # model, losses, hyperparameters):
        """
        Initialize the optimizer.
        """
        self.best_parameters = None
        self.best_result = None
        self.is_maximizing = True

    ##############
    # Properties #
    ##############

    @property
    def best_parameters(self):
        return self._best_parameters

    @best_parameters.setter
    def best_parameters(self, params):
        self._best_parameters = params

    @property
    def best_result(self):
        return self._best_result

    @best_result.setter
    def best_result(self, result):
        self._best_result = result

    @property
    def is_maximizing(self):
        return self._is_maximizing

    @is_maximizing.setter
    def is_maximizing(self, boolean):
        self._is_maximizing = bool(boolean)

    @property
    def is_minimizing(self):
        return not self.is_maximizing

    ###########
    # Methods #
    ###########

    def optimize(self, *args, **kwargs):
        pass

    #############
    # Operators #
    #############

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.optimize(*args, **kwargs)
