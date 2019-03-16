#!/usr/bin/env python
"""Define the NEAT algorithm.

This uses the Neuro-Evolution through Augmenting topologies (NEAT) framework. It allows the evolution of not only the
parameters/weights but also the topological structure of neural networks. Note that the following algorithm only
works with neural networks and is thus tightly coupled with its associated policy/model.
"""

import numpy as np

from pyrobolearn.envs import Env
from pyrobolearn.tasks import RLTask
from pyrobolearn.policies import NEATPolicy
# from rl_algo import RLAlgo

try:
    import neat
    # from neat import nn, population, config, statistics
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install NEAT directly via 'pip install neat-python'.")


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NEAT(object):  # RLAlgo):
    r"""NEAT: Neuro-Evolution through Augmenting Topologies

    Exploration is carried out in the parameter and hyperparameter spaces.

    Warnings: This algorithm is a little bit special and currently only works with the associated policy/learning model.

    References:
        [1] "Evolving Neural Networks through Augmenting Topologies", Stanley et al., 2002
        [2] NEAT-Python
            - documentation: https://neat-python.readthedocs.io/en/latest/index.html
            - github repo: https://github.com/CodeReclaimers/neat-python
        [3] PyTorch NEAT (built upon NEAT-Python): https://github.com/uber-research/PyTorch-NEAT
    """

    def __init__(self, task, policy, population_size=20, species_elitism=2, elitism=2, min_species_size=2,
                 survival_threshold=0.2, max_stagnation=15, compatibility_threshold=3, num_workers=1):
        r"""
        Initialize the NEAT algorithm.

        Args:
            task (RLTask, Env): RL task/env to run
            policy (Policy): specify the policy (model) to optimize
            population_size (int): size of the population
            elitism (int): The number of most-fit individuals in each species that will be preserved as-is from one
                generation to the next.
            num_workers (int): number of workers/jobs to run in parallel
        """
        # create explorer
        # create evaluator
        # create updater
        # super(NEAT, self).__init__(self, explorer, evaluator, updater, num_workers=1)

        # set task
        if isinstance(task, Env):
            task = RLTask(task, policy)
        self.task = task

        # set policy
        if policy is None:
            policy = self.task.policies[0]  # TODO: currently assume only 1 policy
        if not isinstance(policy, NEATPolicy):
            raise TypeError("Expecting the policy to be an instance of 'NEATPolicy'.")
        self.policy = policy

        # set config file
        # more info about genome's config file: https://neat-python.readthedocs.io/en/latest/config_file.html
        # more info about activation fct: https://neat-python.readthedocs.io/en/latest/activation.html
        config_dict = {'[NEAT]': {'fitness_criterion': 'max',
                                  'fitness_threshold': 100,
                                  'no_fitness_termination': True,
                                  'pop_size': population_size,
                                  'reset_on_extinction': True},
                       '[DefaultSpeciesSet]': {'compatibility_threshold': compatibility_threshold},
                       '[DefaultStagnation]': {'species_fitness_func': 'max',
                                               'max_stagnation': max_stagnation,
                                               'species_elitism': species_elitism},
                       '[DefaultReproduction]': {'elitism': elitism,
                                                 'survival_threshold': survival_threshold,
                                                 'min_species_size': min_species_size}}

        # update config file of policy
        self.policy.update_config(config_dict)

        # get population
        self.population = self.policy.population

        # create useful variables
        self.num_steps = 1000
        self.num_rollouts = 1
        self.verbose = False
        self.episode = 0
        self.avg_rewards, self.max_rewards = [], []

        self.best_reward = -np.infty
        self.best_parameters = None

    def explore_and_evaluate(self, genomes, config):
        # print info
        self.episode += 1
        if self.verbose:
            print('\nEpisode {}'.format(self.episode))

        # for each individual in the population, evaluate it on the task
        rewards = []
        for genome_id, genome in genomes:
            # set genome
            self.policy.set_network(genome, config)

            # run a number of rollouts
            reward = []
            for rollout in range(self.num_rollouts):
                # run the task
                rew = self.task.run(num_steps=self.num_steps, use_terminating_condition=True, render=False)
                reward.append(rew)

            # run in parallel
            # jobs = [self.pool.apipe(evaluate, NEAT_Agent, self.env, self.cfg['num_steps'],
            #                         genome, typeNN=self.type) for genome in genomes]
            # for job, genome in zip(jobs, genomes):
            #     _, traj = job.get()
            #     genome.fitness = traj['tot_reward']

            # set fitness value
            reward = np.mean(reward)
            genome.fitness = reward
            rewards.append(reward)

            # save best reward and associated parameter/genome
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_parameters = genome

            # print info
            if self.verbose:
                print('  -- Genome id {} - reward: {}'.format(genome_id, reward))

        # append rewards
        self.avg_rewards.append(np.mean(rewards))
        self.max_rewards.append(np.max(rewards))

        # print info
        if self.verbose:
            print('Episode {}: average reward = {} and best reward = {}'.format(self.episode, self.avg_rewards[-1],
                                                                                self.max_rewards[-1]))

    def optimize(self):
        pass

    def train(self, num_steps=1000, num_rollouts=1, num_episodes=1, verbose=False):
        # set few variables
        self.num_steps = num_steps
        self.num_rollouts = num_rollouts
        self.episode = 0
        self.verbose = verbose
        self.avg_rewards, self.max_rewards = [], []

        # run the algo for the specified number of generations / episodes
        winner = self.population.run(self.explore_and_evaluate, num_episodes)

        # print best reward
        if verbose:
            print("\nBest reward found: {}".format(self.best_reward))

        # set the best genome
        self.policy.genome = winner

        # return the average rewards and max rewards per generation
        return self.avg_rewards, self.max_rewards

    def test(self, num_steps=1000, dt=0, use_terminating_condition=False, render=True):
        return self.task.run(num_steps=num_steps, dt=dt, use_terminating_condition=use_terminating_condition,
                             render=render)
