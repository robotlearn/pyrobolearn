#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide a wrapper around the GPyOpt optimizers for Bayesian Optimization (BO).

References:
    [1] https://sheffieldml.github.io/GPyOpt/
"""

import time
import numpy as np

# Bayesian optimization
try:
    import GPy
    import GPyOpt
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install GPy/GPyOpt directly via 'pip install GPy' and "
                                    "'pip install GPyOpt'.")

from pyrobolearn.optimizers import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BayesianOptimizer(Optimizer):
    r"""Bayesian Optimization

    Bayesian Optimization is a global (gradient-free), probabilistic, non-parametric, model-based, optimization of
    black-box functions.

    Bayesian optimization can be formulated as an optimization problem:

    .. math:: \theta^* = arg\,max_{\theta} f(\theta)

    where :math:`\theta` are the parameters of the model we are trying to optimize, and :math:`f` is the unknown
    objective function which is modeled using a probabilistic model such as a Gaussian Process (GP). By samp

    Popular acquisition functions which specify which parameters to test next by making a trade-off between
    exploitation and exploration, include:
    * Probability of Improvement (PI) [7]:
    * Expected Improvement (EI) [8]:
    * Upper Confidence Bound (UCB) [9]:

    Pseudo-Algo (from [3]):
        D <-- if available: {\theta, f(\theta)}
        Prior <-- if available: prior of the response surface
        while optimize:
            train a response surface from D

    References:
        [1] "Bayesian Approach to Global Optimization: Theory and Applications", Mockus, 1989
        [2] "A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling
            and Hierarchical Reinforcement Learning", Brochu et al., 2010
        [3] "Taking the Human Out of the Loop: a Review of Bayesian Optimization", Shahriari et al., 2016
        [4] "Bayesian Optimization for Learning Gaits under Uncertainty: An Experimental Comparison on a Dynamic
            Bipedal Walker", Calandra et al., 2015
        [5] "Gaussian Processes for Machine Learning", Rasmussen and Williams, 2006
        [6] "GPyOpt: A Bayesian Optimization framework in python" (2016), https://github.com/SheffieldML/GPyOpt
        [7] "A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise",
            Kushner, 1964
        [8] "The Application of Bayesian Methods for Seeking the Extremum", Mockus et al., 1978
        [9] "A Statistical Method for Global Optimization", Cox et al., 1997
    """

    def __init__(self, num_workers=1, *args, **kwargs):
        """
        Initialize the Bayesian optimizer.
        """
        super(BayesianOptimizer, self).__init__(*args, **kwargs)
        self.num_workers = num_workers

    ###########
    # Methods #
    ###########

    def optimize(self, parameters, loss, max_iters=1, verbose=False, *args, **kwargs):
        """
        Optimize the given loss function with respect to the given parameters.

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
        # define domain
        # domain = [{'name': 'params', 'type': 'continuous', 'domain': self.domain, 'dimensionality': len(parameters)}]

        # Solve the optimization
        self.optimizer = GPyOpt.methods.BayesianOptimization(f=loss,
                                                             # domain=domain,
                                                             # constraints=constraints,
                                                             model_type='GP',  # 'sparseGP'
                                                             acquisition_type='EI',  # 'UCB'/'LCB', 'EI', 'MPI'
                                                             acquisition_optimizer_type='lbfgs',  # 'DIRECT', 'CMA'
                                                             num_cores=self.num_workers,
                                                             verbosity=verbose,
                                                             maximize=self.is_maximizing,
                                                             verbosity_model=False,  # True
                                                             kernel=GPy.kern.RBF(input_dim=1))

        # print(opt.model.kernel.name)

        # Run the optimization
        max_iter = max_iters if max_iters < 5 else max_iters - 5  # evaluation budget (min=5)
        # max_time = max_time  # time budget
        eps = 1.e-6  # Minimum allows distance between the last two observations

        if verbose:
            print('Optimizing...')

        # optimize
        start = time.time()
        self.optimizer.run_optimization(max_iter, max_time, eps)
        end = time.time()

        if verbose:
            print('Done with total time: {}'.format(end - start))

        # save best parameters and reward
        self.best_parameters = self.optimizer.x_opt
        self.best_result = self.optimizer.fx_opt

        # print best reward
        if verbose:
            print("\nBest loss value found: {}".format(self.best_result))

        return self.best_result, self.best_parameters
