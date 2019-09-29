# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the Finite-Difference (FD) method algorithm.

This FD method is a policy gradient algorithm that explores in the parameter space of the policy in an episodic way.
"""

import numpy as np
import torch

from pyrobolearn.envs import Env
from pyrobolearn.tasks import RLTask

# from pyrobolearn.algos.rl_algo import GradientRLAlgo

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FD(object):  # GradientRLAlgo):
    r"""Finite-Difference Policy Gradient Method.

    Type:: policy gradient based (on-policy by definition) with exploration in the parameter space


    Description
    -----------

    The goal of RL is to maximize the expected return:

    .. math:: J(\theta) = \int p(\tau) R(\tau) d\tau

    The Finite-Difference (FD) algorithm perturbs the parameter space of the policy and evaluate for each perturbation
    the expected return J(\theta_i + \Delta_{\theta_i})

    The gradient :math:`g_{FD} \approx \nabla_\theta J`is then given by:

    .. math:: `g_{FD} = (\Delta\Theta^\top \Delta\Theta)^{-1} \Delta\Theta^\top \Delta J`

    which is used to to perform a gradient ascent step: :math:`\theta_{i+1} = \theta_{i} + \eta g_{FD}`, where
    :math:`\eta` is the learning rate coefficient.


    Properties
    ----------

    Properties:
        * Exploration is performed in the parameter space of the policy
    Pros:
        * Easy to implement and test
        * work with deterministic and stochastic policies
        * highly efficient in simulation
    Cons:
        * the perturbation of the parameters is hard (especially with systems that can go unstable)
        * O(M^3) for the time complexity (because of the matrix inversion), where M is the number of parameters


    Pseudo-algo
    -----------

    Pseudo-algorithm (taken from [1] with some modification, and reproduce here for completeness)::
        1. Input: initial policy parameters :math:`\theta_0`
        2. for k=0,1,...,num_episodes do
        3.     Exploration: generate policy variation :math:`\Delta \theta_k`, and collect set of trajectories
                :math:`D_k=\{\tau_i\}` by running policy :math:`\pi_{\theta_k + \Delta \theta_k}` and
                :math:`\pi_{\theta_k - \Delta \theta_k}` in the environment.
        4.     Evaluation: compute total rewards
                :math:`J_{k+} = \mathbb{E}_{\theta_k + \Delta \theta_k}[\sum_{t=0}^T \gamma^t r_t]`,
                :math:`J_{k-} = \mathbb{E}_{\theta_k - \Delta \theta_k}[\sum_{t=0}^T \gamma^t r_t]`, and difference
                gradient estimator :math:`\Delta J = J_{k+} - J_{k-}`
        5.     Update: compute gradient :math:`g_{FD} = (\Delta \Theta ^\trsp \Delta \Theta)^{-1} \Delta\Theta
                \Delta\hat{J}` and update policy parameters using :math:`\theta_{k+1} = \theta_k + \alpha_k g_{FD}`

    References::
        [1] "Policy Gradient Methods" (http://www.scholarpedia.org/article/Policy_gradient_methods), Peters, 2010
    """

    def __init__(self, task, policy, num_variations=None, std_dev=0.01, difference_type='central', learning_rate=0.001,
                 normalize_grad=False, num_workers=1):
        # hyperparameters
        """
        Initialize the FD algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            policy (Policy): specify the policy (model) to optimize
            num_variations (None, int): number of times we vary the parameters by a small different increment.
                If None, it will be twice the number of parameters as according to [1], it yields very accurate
                gradient estimates.
            std_dev (float): the small increments are generated from a Normal distribution center at 0 and
            difference_type (str): there are two difference type of estimators: 'forward' or 'central'.
                The forward-difference estimator computes the gradient using
                :math:`J(\theta + \Delta\theta) - J(\theta)`, while the central-difference estimator computes the
                gradient using :math:`J(\theta + \Delta\theta) - J(\theta - \Delta\theta)`
            learning_rate (float): learning rate (=coefficient) for the gradient ascent step
            normalize_grad (bool): specify if we should normalize the gradients
            num_workers (int): number of workers/jobs to run in parallel
        """
        # create explorer
        # create evaluator
        # create updater
        # super(FD, self).__init__(self, explorer, evaluator, updater, num_workers=1)

        if isinstance(task, Env):
            task = RLTask(task, policy)
        self.task = task
        self.policy = policy
        self.num_workers = num_workers

        # set the number of variations (small increments to vary the parameters)
        # From [1]: "Empirically it can be observed that taking the number of variations as twice the number
        # of parameters yields very accurate gradient estimates"
        if num_variations is None:
            self.num_variations = 2 * self.policy.num_parameters

        # set standard deviation
        self.stddev = np.abs(std_dev)

        # set difference type
        if difference_type != 'forward' and difference_type != 'central':
            raise ValueError("Expecting the 'difference_type' argument to be 'forward' or 'central'. Instead got "
                             "'{}'".format(difference_type))
        self.difference_type = difference_type

        # set other parameters
        self.lr = learning_rate
        self.normalize_grad = bool(normalize_grad)

        # remember best parameters
        self.best_reward = -np.infty
        self.best_parameters = None

    def explore_and_evaluate(self, params, num_steps, num_rollouts):
        # set policy parameters
        self.policy.set_vectorized_parameters(params)

        # run a number of rollouts
        reward = []
        for rollout in range(num_rollouts):
            rew = self.task.run(num_steps=num_steps, use_terminating_condition=True, render=False)
            reward.append(rew)
        reward = np.mean(reward)

        return reward

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
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # for each episode
        rewards = []
        for episode in range(num_episodes):

            # get parameters
            params = self.policy.get_vectorized_parameters()
            J_plus, J_minus = np.zeros(self.num_variations), np.zeros(self.num_variations)
            Delta_Params = np.zeros((self.num_variations, len(params)))

            # evaluate with the current parameters
            J = self.explore_and_evaluate(params, num_steps, num_rollouts)
            rewards.append(J)

            # Save best reward and associated parameter
            if J > self.best_reward:
                self.best_reward = J
                self.best_parameters = params

            # print info
            if verbose:
                print('\nEpisode {} - expected return: {}'.format(episode + 1, J))

            # 1. Explore
            for i in range(self.num_variations):
                # sample parameter increment step vector
                delta_params = np.random.normal(loc=0.0, scale=self.stddev, size=len(params))
                Delta_Params[i] = delta_params

                # estimate J(\theta + \delta)
                new_params = params + delta_params
                J_plus[i] = self.explore_and_evaluate(new_params, num_steps, num_rollouts)

                # estimate J(\theta - \delta)
                if self.difference_type == 'forward':
                    J_minus[i] = J
                elif self.difference_type == 'central':
                    new_params = params - delta_params
                    J_minus[i] = self.explore_and_evaluate(new_params, num_steps, num_rollouts)
                else:
                    raise ValueError("Expecting the 'difference_type' argument to be 'forward' or 'central'. "
                                     "Instead got '{}'".format(self.difference_type))
            # 2. Evaluate

            # 3. Update
            delta_J = J_plus - J_minus
            grad = np.linalg.pinv(Delta_Params).dot(delta_J)
            if self.normalize_grad:
                grad /= np.linalg.norm(grad)
            params = params + self.lr * grad  # TODO: allows the user to choose the optimizer
            # self.optimizer.optimize(self.policy.list_parameters(), grad)
            self.policy.set_vectorized_parameters(params)

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
