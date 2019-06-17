#!/usr/bin/env python
"""Provide the Bayesian Optimization algorithm.
"""

import numpy as np
import torch
import time
import GPy
import GPyOpt

from pyrobolearn.envs import Env
from pyrobolearn.tasks import RLTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BO(object):
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

    def __init__(self, task=None, policy=None, domain=(-3., 3.), num_workers=1):
        """
        Initialize the Bayesian Optimization algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            policy (Policy): specify the policy (model) to optimize
            num_workers (int): number of workers/jobs to run in parallel
        """
        if isinstance(task, Env):
            task = RLTask(task, policy)
        self.task = task
        self.policy = policy
        self.num_workers = num_workers

        self.best_reward = -np.infty
        self.best_parameters = None

        self.num_steps = 1000
        self.num_rollouts = 1
        self.verbose = False
        self.episode = 0
        self.render = False

        self.domain = domain

        self.rewards = []

    # def get_vectorized_parameters(self, to_numpy=True):
    #     parameters = self.policy.parameters
    #
    #     vector = []
    #     from_numpy = False
    #     for parameter in parameters:
    #         if isinstance(parameter, np.ndarray):
    #             from_numpy = True
    #         vector.append(parameter.reshape(-1))
    #
    #     if from_numpy:
    #         print(vector)
    #         vector = np.concatenate(vector)
    #         if not to_numpy:
    #             return torch.from_numpy(vector)
    #     else:
    #         vector = torch.cat(vector)
    #         if to_numpy:
    #             return vector.detach().numpy()
    #     return vector
    #
    # def set_vectorized_parameters(self, vector):
    #     # convert the vector to torch array
    #     if isinstance(vector, np.ndarray):
    #         vector = torch.from_numpy(vector).float()
    #
    #     # set the parameters from the vectorized one
    #     idx = 0
    #     for parameter in self.policy.parameters:
    #         size = parameter.nelement()
    #         parameter.data = vector[idx:idx+size].reshape(parameter.shape)
    #         idx += size

    def explore_and_evaluate(self, params):
        # set policy parameters
        # self.set_vectorized_parameters(params[0])
        self.policy.set_vectorized_parameters(params[0])

        # run a number of rollouts
        reward = []
        for rollout in range(self.num_rollouts):
            rew = self.task.run(num_steps=self.num_steps, dt=self.dt, use_terminating_condition=True,
                                render=self.render)
            reward.append(rew)
        reward = np.mean(reward)

        self.rewards.append(reward)

        # print info
        self.episode += 1
        if self.verbose:
            print('Episode {} - reward: {}'.format(self.episode, reward))

        return reward

    def train(self, num_steps=1000, num_rollouts=1, num_episodes=1, verbose=False, render=False, seed=None,
              max_time=3600, dt=0):
        """
        Train the policy.

        Args:
            num_steps (int): number of steps per rollout / episode. In one episode, how many steps does the environment
                proceeds.
            num_rollouts (int): number of rollouts per episode to average the results.
            num_episodes (int): number of episodes.
            verbose (bool): If True, it will print information about the training process.
            seed (int): random seed.

        Returns:
            list of float: average rewards per episode.
            list of float: maximum reward obtained per episode.
        """
        # set few variables
        self.num_steps = num_steps
        self.num_rollouts = num_rollouts
        self.episode = 0
        self.verbose = verbose
        self.rewards = []
        self.dt = dt
        self.render = render

        # set seed if specified
        if seed is not None:
            np.random.seed(seed)

        # init
        # parameters = self.get_vectorized_parameters(to_numpy=True)
        parameters = self.policy.get_vectorized_parameters(to_numpy=True)

        # define domain
        domain = [{'name': 'params', 'type': 'continuous', 'domain': self.domain, 'dimensionality': len(parameters)}]

        # Solve the optimization
        opt = GPyOpt.methods.BayesianOptimization(f=self.explore_and_evaluate,
                                                  domain=domain,
                                                  model_type='GP',  # 'sparseGP'
                                                  acquisition_type='EI',  # 'UCB'/'LCB', 'EI', 'MPI'
                                                  acquisition_optimizer_type='lbfgs',  # 'DIRECT', 'CMA'
                                                  num_cores=self.num_workers,
                                                  verbosity=verbose,
                                                  maximize=True,  # True
                                                  verbosity_model=False,  # True
                                                  kernel=GPy.kern.RBF(input_dim=1))

        # print(opt.model.kernel.name)

        # Run the optimization
        max_iter = num_episodes if num_episodes < 5 else num_episodes - 5  # evaluation budget (min=5)
        max_time = max_time  # time budget
        eps = 10e-6  # Minimum allows distance between the last two observations

        if verbose:
            print('Optimizing...')

        start = time.time()
        opt.run_optimization(max_iter, max_time, eps)
        end = time.time()

        if verbose:
            print('Done with total time: {}'.format(end - start))

        # save best parameters and reward
        self.best_parameters = opt.x_opt
        self.best_reward = -opt.fx_opt

        # print best reward
        if verbose:
            print("\nBest reward found: {}".format(self.best_reward))

        # set the best parameters
        self.policy.set_vectorized_parameters(self.best_parameters)

        return self.rewards

    def test(self, num_steps=1000, dt=0, use_terminating_condition=False, render=True):
        """
        Test the policy in the environment.

        Args:
            num_steps (int): number of steps to run the episode.
            dt (float): time to sleep before the next step.
            use_terminating_condition (bool): If True, it will use the terminal condition to end the environment.
            render (bool): If True, it will render the environment.

        Returns:
            float: obtained reward
        """
        return self.task.run(num_steps=num_steps, dt=dt, use_terminating_condition=use_terminating_condition,
                             render=render)

    # def optimize(self, fct, num_iter=6, max_time=3600, seed=None, verbose=False):
    #     """
    #     Optimize the given model based
    #
    #     Args:
    #         num_iter (int): number of iteration
    #         max_time (float):
    #         seed (int): random seed
    #         verbose (bool): True if we should print information during the optimization process
    #
    #     Returns:
    #
    #     """
    #     # set seed if specified
    #     if seed is not None:
    #         np.random.seed(seed)
    #
    #     # define domain
    #     domain = [{'name': 'params', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1}]
    #
    #     # Solve the optimization
    #     opt = GPyOpt.methods.BayesianOptimization(f=fct,
    #                                               domain=domain,
    #                                               model_type='GP',  # 'sparseGP'
    #                                               acquisition_type='EI',  # 'UCB'/'LCB', 'EI', 'MPI'
    #                                               acquisition_optimizer_type='lbfgs',  # 'DIRECT', 'CMA'
    #                                               num_cores=1,
    #                                               verbosity=verbose,
    #                                               maximize=True,  # True
    #                                               verbosity_model=False,  # True
    #                                               kernel=GPy.kern.RBF(input_dim=1))
    #
    #     # print(opt.model.kernel.name)
    #
    #     # Run the optimization
    #     max_iter = num_iter  # evaluation budget (min=4), nb_eval = 4 + max_iter
    #     max_time = max_time  # time budget
    #     eps = 10e-6  # Minimum allows distance between the last two observations
    #
    #     if verbose:
    #         print('Optimizing...')
    #
    #     start = time.time()
    #     opt.run_optimization(max_iter, max_time, eps)
    #     end = time.time()
    #
    #     if verbose:
    #         print('Done with total time: {}'.format(end - start))
    #
    #     # save best parameters and reward
    #     self.best_parameters = opt.x_opt
    #     self.best_reward = opt.fx_opt
    #
    #     return opt


# Tests
if __name__ == '__main__':
    pass
