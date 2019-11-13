#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the abstract optimizer class.

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
__license__ = "GNU GPLv3"
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
        self.optimizer = None
        self.is_minimizing = True
        self.best_parameters = None
        self.best_result = None

    ##############
    # Properties #
    ##############

    @property
    def best_parameters(self):
        """Return the best parameters."""
        return self._best_parameters

    @best_parameters.setter
    def best_parameters(self, params):
        """Set the best parameters."""
        self._best_parameters = params

    @property
    def best_result(self):
        """Return the best value."""
        return self._best_result

    @best_result.setter
    def best_result(self, result):
        """Set the optimal value."""
        self._best_result = result

    @property
    def is_minimizing(self):
        """Return if the optimizer is used to minimize an objective / loss function."""
        return self._is_minimizing

    @is_minimizing.setter
    def is_minimizing(self, boolean):
        """Set if the optimizer is used to minimize an objective / loss function."""
        self._is_minimizing = boolean

    @property
    def is_maximizing(self):
        """Return if we are maximizing. If False, we are minimizing."""
        return not self.is_minimizing

    @is_maximizing.setter
    def is_maximizing(self, boolean):
        """Setting if the optimizer is used to maximize an objective function."""
        self.is_minimizing = not bool(boolean)

    ###########
    # Methods #
    ###########

    def convert_from(self, parameters):
        """
        Convert the parameters to the desired form for the optimizer. This should be implemented in the child classes.

        Args:
            parameters: initial parameters.

        Returns:
            object: parameters in the desired form.
        """
        return parameters

    def convert_to(self, parameters):
        """
        Convert back the optimized parameters to the initial form. This should be implemented in the child classes.

        Args:
            parameters: optimized parameters.

        Returns:
            object: parameters in the initial form.
        """
        return parameters

    def optimize(self, parameters, loss, max_iters=1, verbose=False, *args, **kwargs):
        """
        Optimize the given objective function using the optimizer. This should be implemented in the child classes.

        Args:
            parameters: parameters.
            loss: callable objective / loss function to minimize.
            max_iters (int): number of maximum iterations.
            verbose (bool): if True, it will display information during the optimization process.
            *args: list of arguments to give to the loss function if callable.
            **kwargs: dictionary of arguments to give to the loss function if callable.

        Returns:
            float, torch.Tensor, np.array: loss scalar value.
            object: best parameters
        """
        pass

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string of the object."""
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the object."""
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        """Optimize the given objective function using the optimizer."""
        return self.optimize(*args, **kwargs)
