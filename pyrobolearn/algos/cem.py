#!/usr/bin/env python
"""Provide the Cross-Entropy Method algorithm.

This CEM is an evolutionary algorithm that explores in the parameter space of the policy in an episodic way.
"""

import numpy as np
import torch
# from pathos.multiprocessing import Pool

from pyrobolearn.envs import Env
from pyrobolearn.tasks import RLTask

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CEM(object):  # RLAlgo
    r"""Cross-Entropy Method

    Type: population-based and model-free, exploration in parameter space, episode-based.

    The Cross-Entropy Method (CEM) is an evolutionary algorithm that explores in the parameter space of the policy
    in an episodic way. That is, everytime the parameters are updated, the policy is run for a whole episode before
    being evaluated.

    This algorithm works by first assuming that the parameters are generated from a multivariate normal distribution
    with an initial mean and a fixed covariance matrix. Few samples are then drawn from this distribution to form
    the initial population of parameter vectors. Each parameter vector is then set on the policy and evaluated on
    the whole episode. The best parameter vectors which constitutes a fraction of the population and called
    the elites are then selected. From them, a new mean and standard covariance matrix are computed and used to form
    the new multivariate normal distribution from which the next population is generated. This process is carried
    out for several generation. At the end, the best parameter (the elite) is returned.

    References:
        [1] "The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation
            and Machine Learning", Rubinstein et al., 2004
        [2] "A Tutorial on the Cross-Entropy Method", de Boer, 2003
        [3] "The Cross Entropy Method for Fast Policy Search", Mannor et al., 2003 (ICML)

    Interesting codes (the code in this file was inspired from the first reference):
        - Schulman's presentation (2016)
        - modular_rl: https://github.com/joschu/modular_rl
        - pytorch-rl: https://github.com/khushhallchandra/pytorch-rl
        - rllab: https://github.com/rll/rllab
    """

    def __init__(self, task, policy, population_size=20, elite_fraction=0.2, num_workers=1):
        """
        Initialize the CEM algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            policy (Policy): specify the policy (model) to optimize
            population_size (int): size of the population
            elite_fraction (float): fraction of elites to use to compute the new mean and covariance matrix of the
                multivariate normal distribution
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
        self.elite_fraction = elite_fraction
        self.num_workers = num_workers

        self.best_reward = -np.infty
        self.best_parameters = None

    def get_vectorized_parameters(self, to_numpy=True):
        parameters = self.policy.parameters
        vector = torch.cat([parameter.reshape(-1) for parameter in parameters])  # np.concatenate = torch.cat
        if to_numpy:
            return vector.detach().numpy()
        return vector

    def set_vectorized_parameters(self, vector):
        # convert the vector to torch array
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()

        # set the parameters from the vectorized one
        idx = 0
        for parameter in self.policy.parameters:
            size = parameter.nelement()
            parameter.data = vector[idx:idx+size].reshape(parameter.shape)
            idx += size

    def train(self, num_steps=1000, num_rollouts=1, num_episodes=1, verbose=False, seed=None):
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # create recorders
        max_rewards, avg_rewards = [], []

        # init
        theta_mean = self.get_vectorized_parameters(to_numpy=True)
        theta_std = np.ones(len(theta_mean))
        # pool = Pool(self.num_workers)

        # for each episode/generation
        for episode in range(num_episodes):
            if verbose:
                print('\nEpisode {}'.format(episode+1))

            # 1. Explore
            # sample parameter vectors
            thetas = np.random.multivariate_normal(theta_mean, np.diag(theta_std), self.population_size)

            # perform one episode for each parameter
            # jobs = [pool.apipe(self.task.run, num_steps, use_terminating_condition=False) for theta in thetas]
            rewards = []
            for i, theta in enumerate(thetas):
                # set policy parameters
                self.set_vectorized_parameters(theta)

                # run a number of rollouts
                reward = []
                for rollout in range(num_rollouts):
                    rew = self.task.run(num_steps=num_steps, use_terminating_condition=True, render=False)
                    reward.append(rew)
                reward = np.mean(reward)
                rewards.append(reward)

                # print info
                if verbose:
                    print(' -- individual {} with avg reward of {}'.format(i+1, reward))

            # 2. Evaluate (compute loss)

            # 3. Update

            # get elite parameters
            num_elites = int(self.population_size * self.elite_fraction)
            elite_ids = np.argsort(rewards)[-num_elites:]
            elite_thetas = np.array([thetas[i] for i in elite_ids])

            # update theta_mean and theta_std
            theta_mean = elite_thetas.mean(axis=0)
            theta_std = np.sqrt(np.mean((elite_thetas - theta_mean) ** 2, axis=0))

            # 4. Save best reward and associated parameter
            max_reward, avg_reward = np.max(rewards), np.mean(rewards)
            if max_reward > self.best_reward:
                self.best_reward = max_reward
                self.best_parameters = thetas[elite_ids[-1]]

            # print info
            if verbose:
                print("Episode {} mean reward: {} max reward: {}".format(episode+1, avg_reward, max_reward))

            # Save the evolution of the algo
            avg_rewards.append(avg_reward)
            max_rewards.append(max_reward)

        # print best reward
        if verbose:
            print("\nBest reward found: {}".format(self.best_reward))

        # set the best parameters
        self.set_vectorized_parameters(self.best_parameters)

        return avg_rewards, max_rewards

    def test(self, num_steps=1000, dt=0, use_terminating_condition=False, render=True):
        return self.task.run(num_steps=num_steps, dt=dt, use_terminating_condition=use_terminating_condition,
                             render=render)


# Tests
if __name__ == '__main__':
    pass
