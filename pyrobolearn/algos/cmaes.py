#!/usr/bin/env python
"""Provide the Covariance Matrix Adaptation Evolution Strategy algorithm.

'The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a stochastic derivative-free numerical optimization
algorithm for difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization problems in
continuous search spaces.' [1]

References:
    [1] "Python implementation of CMA-ES", Hansen et al., 2019 (https://github.com/CMA-ES/pycma)
"""

import numpy as np

try:
    import cma
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install CMA-ES or `pycma` directly via 'pip install cma'.")

from pyrobolearn.envs import Env
from pyrobolearn.tasks import RLTask

# from pyrobolearn.algos.rl_algos import *

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CMAES(object):  # Algo):
    r"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

    Type: population-based (genetic), stochastic and derivative-free, exploration in parameter space, optimization
    for non-linear and non-convex functions, episode-based.

    'The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a stochastic derivative-free numerical
    optimization algorithm for difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization
    problems in continuous search spaces.' [3]

    Complexity:

    References:
        [1] "Completely Derandomized Self-Adaptation in Evolution Strategies", Hansen et al., 2001
        [2] "The CMA Evolution Strategy: A Tutorial", Hansen, 2016
        [3] "Python implementation of CMA-ES", Hansen et al., 2019: https://github.com/CMA-ES/pycma
        [4] pycma API documentation: cma.gforge.inria.fr/apidocs-pycma

    Python Implementations:
    - pycma: https://github.com/CMA-ES/pycma
    - rllab: https://github.com/rll/rllab
    - DEAP: https://github.com/DEAP/deap   and   http://deap.readthedocs.io/en/master/examples/cmaes.html
    """

    def __init__(self, task, policy, population_size=20, sigma=0.5, num_workers=1):
        """
        Initialize the CMA-ES algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            policy (Policy): specify the policy (model) to optimize
            population_size (int): size of the population
            sigma (float): initial standard deviation for CMA-ES
            num_workers (int): number of workers/jobs to run in parallel
        """
        # create explorer
        # create evaluator
        # create updater
        # super(CEM, self).__init__(self, explorer, evaluator, updater, num_workers=1)

        if isinstance(task, Env):
            task = RLTask(task, policy)
        self.task = task
        self.policy = policy
        self.population_size = population_size
        self.sigma = sigma
        self.num_workers = num_workers
        self.es = None

        self.best_reward = -np.infty
        self.best_parameters = None

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

    def explore_and_evaluate(self, params, num_steps, num_rollouts):
        # set policy parameters
        self.policy.set_vectorized_parameters(params)

        # run a number of rollouts
        reward = []
        for rollout in range(num_rollouts):
            rew = self.task.run(num_steps=num_steps, use_terminating_condition=True, render=False)
            reward.append(rew)
        reward = np.mean(reward)

        # return cost to minimize
        return -reward

    def train(self, num_steps=1000, num_rollouts=1, num_episodes=1, verbose=False, seed=None):
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
        # create CMA-ES
        self.es = cma.CMAEvolutionStrategy(self.policy.get_vectorized_parameters(), sigma0=self.sigma,
                                           inopts={'popsize': self.population_size})  # {'bounds': [-np.inf, np.inf]}

        # set seed
        if seed is not None:
            self.es.opts.set({'seed': seed})

        # set number of iterations
        self.es.opts.set({'maxiter': num_episodes})

        # create recorders
        max_rewards, avg_rewards = [], []

        # optimize
        # self.es.optimize(self.explore_and_evaluate, iterations=num_episodes, args=(num_steps, num_rollouts),
        #                  verb_disp=int(verbose))
        # evaluate
        # (solutions, costs) = self.es.ask_and_eval(self.explore_and_evaluate)

        # for each episode/generation
        # while not self.es.stop():
        for episode in range(num_episodes):
            if self.es.stop():
                break

            # optimize
            parameters = self.es.ask()
            costs = [self.explore_and_evaluate(params, num_steps, num_rollouts) for params in parameters]
            self.es.tell(parameters, costs)

            # get rewards
            max_rewards.append(-np.min(costs))
            avg_rewards.append(-np.mean(costs))

            # print info
            if verbose:
                # self.es.disp(1)
                print("Episode {} mean reward: {} max reward: {}".format(episode + 1, avg_rewards[-1], max_rewards[-1]))

        # if verbose:
        #     self.es.result_pretty()

        # print info
        if verbose:
            # self.es.disp(1)
            print('Termination by {}'.format(self.es.stop()))
            print('Best reward found = {}'.format(-self.es.result[1]))
            # print('solution = {}'.format(self.es.result[0]))

        # get results (check documentation of ` _CMAEvolutionStrategyResult`)
        self.best_parameters = self.es.result[0]
        self.best_reward = -self.es.result[1]

        # set the best parameters
        self.policy.set_vectorized_parameters(self.best_parameters)

        return avg_rewards, max_rewards

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
