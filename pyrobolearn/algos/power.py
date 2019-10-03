# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the PoWER reinforcement learning algorithm.

This Policy learning by Weighting Exploration with the Returns (PoWER) algorithm is an model-free, on-policy, and
Expectation-Maximization (EM) algorithm. The exploration is carried out in the parameter space.
"""

import numpy as np
import sys

from pyrobolearn.envs import Env
from pyrobolearn.tasks import RLTask
# from pyrobolearn.algos.rl_algo import EMRLAlgo

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PoWER(object):  # EMRLAlgo):
    r"""PoWER: Policy learning by Weighting Exploration with the Returns

    Type:: this is a model-free, on-policy, episode-based, EM (Expectation-Maximization) algorithm.

    .. math:: TODO

    Properties:
        * The learning model is linear with respect to the parameters (in order to compute the close form solution)
        * Exploration is performed in the parameter space with a gaussian distribution over the parameters
    Pros:
        * Pros of using an EM algorithm (improvement on the lower bound)
    Cons:
        * Initialization of the EM algorithm is crucial; it works better when providing first few demonstrations
            of the task to complete
        * the rewards have to be strictly positives
        * doesn't work with nonlinear models (in terms of the parameters)

    .. note::

        * the reward has to be strictly positive in order to be a proper probability distribution (see [1])
        * this algorithm is popular and works well with Dynamic Movement Primitives

    The code is based on [2].

    Pseudocode:
        1. Exploration:
        2. Evaluation:
        3. Update:

    Examples:
        TODO

    References:
        [1] "Policy Search for Motor Primitives in Robotics", Kober et al., 2010
        [2] Original Matlab code (Kober): http://www.ausy.tu-darmstadt.de/uploads/Member/JensKober/matlab_PoWER.zip
    """

    def __init__(self, task, policy, std_params=1., num_best_rollouts=10, num_workers=1):
        """
        Initialize the PoWER algorithm.

        Args:
            task (RLTask, Env): RL task/env to run.
            policy (Policy): specify the policy (model) to optimize.
            std_params (float): standard deviation of the parameters.
        """
        # create explorer
        # create evaluator
        # create updater
        # super(PoWER, self).__init__(task, exploration_strategy, memory, hyperparameters)

        # set task
        if isinstance(task, Env):
            task = RLTask(task, policy)
        if not isinstance(task, RLTask):
            raise TypeError("Expecting task to be an instance of RLTask.")
        self.task = task

        # set policy
        self.policy = policy
        if not self.policy.is_parametric():
            raise ValueError("The policy should be parametric")
        if not self.policy.is_linear():
            raise ValueError("The policy should be linear with respect to the parameters")

        # set standard deviation of the parameters
        self.std_params = std_params

        # set num best rollouts for memory
        self.num_best_rollouts = num_best_rollouts

        # remember best parameters
        self.best_reward = -np.infty
        self.best_parameters = None

    ##############
    # Properties #
    ##############

    @property
    def std_params(self):
        """Return the standard deviation of the parameters."""
        return self._std_params

    @std_params.setter
    def std_params(self, std_params):
        """Set the standard deviation of the parameters."""
        if std_params < 0.:
            std_params = 1.
        self._std_params = std_params

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
            if rew <= 0:
                raise ValueError("With PoWER, the reward must be strictly positive.")
            reward.append(rew)
        reward = np.mean(reward)

        return reward

    # def train(self, num_episodes=1, num_steps=100, std_params=1., task=None, seed=None, debug=False):
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
        """
        # check parameters
        if num_episodes < 1:
            num_episodes = 1
        if num_steps < 1:
            num_steps = 1
        if num_rollouts < 1:
            num_rollouts = 1

        # set random seed
        if seed is not None:
            np.random.seed(seed)

        # Initial policy parameters
        num_params = self.policy.num_parameters
        w = self.policy.get_vectorized_parameters()

        # define exploration parameters
        mean = np.zeros(num_params)
        C_init = np.ones(num_params) * np.sqrt(self.std_params)
        C = C_init

        # Evaluate reward with current parameter
        reward = self.explore_and_evaluate(w, num_steps=num_steps, num_rollouts=num_rollouts)

        # remember best set of weights and best reward
        if self.best_reward < reward:
            self.best_reward, self.best_parameters = reward, np.copy(w)

        # print info
        if verbose:
            print('Episode {}/{} with current and best reward: {}, {}'.format(0, num_episodes, reward,
                                                                              self.best_reward))

        # for each episode
        rewards, memory = [reward], []
        memory = []
        for episode in range(num_episodes):

            # 1. Explore
            # explore and sample
            eps = np.random.multivariate_normal(mean, np.diag(C))
            weps = w + eps
            reward = self.explore_and_evaluate(weps, num_steps=num_steps, num_rollouts=num_rollouts)
            rewards.append(reward)

            # remember best set of weights and best reward
            if self.best_reward < reward:
                self.best_reward, self.best_parameters = reward, weps

            # print info
            if verbose:
                print('Episode {}/{} with current and best reward: {}, {}'.format(episode+1, num_episodes, reward,
                                                                                  self.best_reward))
            # record in memory
            memory.append((reward, weps, C))

            # 2. Evaluate
            # Reweight: importance sampling
            list.sort(memory, key=lambda x: x[0])  # in-place operation
            memory = memory[-self.num_best_rollouts:]  # just keep the best rollouts

            # 3. Update
            # Update weight parameters
            num, den = 0, 0
            for (rew, weps, Ceps) in memory[-self.num_best_rollouts:]:
                # prec = np.linalg.inv(Ceps)
                prec = np.linalg.inv(np.diag(Ceps))
                eps = weps - w
                num += prec.dot(eps) * rew
                den += prec * rew
            w = w + np.linalg.inv(den + 1e-8).dot(num)

            # Update Covariance matrix
            num, den = 0, 0
            for (rew, weps, Ceps) in memory[-self.num_best_rollouts:]:
                eps = weps - w
                # num += eps.dot(eps.T)*rew
                num += (eps ** 2) * rew
                den += rew
            C = num / (den + 1e-10)

            # Apply an upper and lower limit to the exploration (so we still get a kind of exploration)
            C = np.minimum(np.maximum(C, 0.1 * C_init), 10. * C_init)

        # print best reward
        if verbose:
            print("\nBest reward found: {}".format(self.best_reward))

        # set the best parameters
        self.policy.set_vectorized_parameters(self.best_parameters)

        return rewards

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
