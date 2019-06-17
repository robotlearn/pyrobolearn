#!/usr/bin/env python
"""Provide a wrapper around the Covariance Adaptation Matrix (CMA) which uses an evolution strategy to optimize
the parameters.

References:
    [1] https://github.com/CMA-ES/pycma
"""

import numpy as np
import torch

# CMA-ES
try:
    import cma
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install CMA-ES or `pycma` directly via 'pip install cma'.")

from pyrobolearn.optimizers import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CMAES(Optimizer):
    r"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

    Type: population-based (genetic), stochastic and derivative-free, exploration in parameter space, optimization
    for non-linear and non-convex functions, episode-based.

    'The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a stochastic derivative-free numerical
    optimization algorithm for difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization
    problems in continuous search spaces.' [3]

    References:
        [1] "Completely Derandomized Self-Adaptation in Evolution Strategies", Hansen et al., 2001
        [2] "The CMA Evolution Strategy: A Tutorial", Hansen, 2016
        [3] "Python implementation of CMA-ES", Hansen et al., 2019: https://github.com/CMA-ES/pycma
        [4] pycma API documentation: cma.gforge.inria.fr/apidocs-pycma

    Python Implementations:
    - pycma: https://github.com/CMA-ES/pycma
    """

    def __init__(self, population_size=20, sigma=0.5, *args, **kwargs):
        """
        Initialize the CMA-ES optimizer.
        """
        super(CMAES, self).__init__(*args, **kwargs)
        self.population_size = population_size
        self.sigma = sigma
        self.shape = None
        self.dtype = None

    ##############
    # Properties #
    ##############

    @property
    def population_size(self):
        """Return the population size."""
        return self._population_size

    @population_size.setter
    def population_size(self, size):
        """Set the population size."""
        # check argument
        if not isinstance(size, int):
            raise TypeError("Expecting the population size to be an integer.")
        if size < 1:
            raise ValueError("Expecting the population size to be an integer bigger than 0.")

        # set population size
        self._population_size = size

    ###########
    # Methods #
    ###########

    def convert_from(self, parameters):
        if isinstance(parameters, np.ndarray):
            self.shape = parameters.shape
            self.dtype = np.ndarray
            return parameters.reshape(-1)
        if isinstance(parameters, torch.Tensor):
            self.shape = tuple(parameters.shape)
            self.dtype = torch.Tensor
            if parameters.requires_grad:
                return parameters.detach().numpy()
            return parameters.numpy()

    def convert_to(self, parameters):
        if isinstance(parameters, np.ndarray):
            if self.dtype == np.ndarray:
                return parameters.reshape(self.shape)
            if self.dtype == torch.Tensor:
                return torch.from_numpy(parameters.reshape(self.shape))

    def optimize(self, parameters, loss, bounds=None, max_iters=1, seed=None, options={}, verbose=False,
                 *args, **kwargs):
        """
        Optimize the given loss function with respect to the given parameters.

        Args:
            parameters (np.array): parameters to optimize.
            loss (callable): callable objective / loss function to minimize.
            bounds (tuple, list, np.array): parameter bounds. E.g. bounds=[0, np.inf]
            max_iters (int): number of maximum iterations.
            verbose (bool): if True, it will display information during the optimization process.
            *args: list of arguments to give to the loss function if callable.
            **kwargs: dictionary of arguments to give to the loss function if callable.

        Returns:
            float, torch.Tensor, np.array: loss scalar value.
            object: best parameters
        """
        # check loss function
        if not callable(loss):
            raise TypeError("Expecting the given loss function to be callable.")

        # check optimizer options
        opts = {'popsize': self.population_size}
        if bounds is not None:  # set the parameter bounds
            opts['bounds'] = bounds
        if seed is not None:  # set the seed
            opts['seed'] = seed
        if max_iters is not None:  # set the maximum number of iterations
            opts['maxiter'] = max_iters

        # update the rest of options
        opts.update(options)

        # create CMA-ES
        self.optimizer = cma.CMAEvolutionStrategy(parameters, sigma0=self.sigma, inopts=opts)

        # optimize
        parameters = self.optimizer.ask()
        values = [loss(params, *args, **kwargs) for params in parameters]
        self.optimizer.tell(parameters, values)

        # save the best result and parameters
        self.best_parameters = self.optimizer.result[0]
        self.best_result = self.optimizer.result[1]

        return self.best_result, self.best_parameters


