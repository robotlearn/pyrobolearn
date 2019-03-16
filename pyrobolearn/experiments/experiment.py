#!/usr/bin/env python
"""Define the `Experiment` class, which allows the user to define an experiment.

We define the `Experiment` class here which is the highest-level class of our framework. More specifically, it allows
to organize which tasks to run, which metrics to use, and allows to easily compare different models, algos, methods,
and so on.

An experiment should be well-defined, and clearly demonstrate the results.

Dependencies:
- `pyrobolearn.tasks`  (thus `pyrobolearn.envs`, `pyrobolearn.policies`, `pyrobolearn.approximators`,...)
- `pyrobolearn.metrics`
- `pyrobolearn.algos`
"""

from abc import ABCMeta, abstractmethod

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Experiment(object):
    r"""Experiment class.

    An experiment allows to organize which tasks to run (and in which order), which metrics to use to
    evaluate the experiment, and to easily compare different learning models/rl, algos, and so on.
    This represents the highest level of our `pyrobolearn` framework. Sometimes, the robot, policy, or algo
    can be as well specified as an argument when creating an experiment.

    .. seealso: the second highest level of our framework is the `Task` class which defines the policy and
    environment, and is independent of the learning algorithm as well as the metrics.

    Examples:
        # 1st Example
        sim = Bullet()
        experiment = Experiment(sim) # this creates the task (states/actions, rl, environment), algo, metrics,...
        results = experiment.run(train=True, evaluate_metrics=True)
        experiment.plot_metrics()
        reward = experiment.evaluate_policy()

        # 2nd Example
        sim = Bullet()
        robots = [Robot(), Robot()]
        algo = Algo()
        experiment = Experiment(sim, robots, algo)
        results = experiment(train=True, evaluate_metrics=True)
        experiment.plot_metrics()
    """
    __metaclass__ = ABCMeta

    def __init__(self, tasks, policies, algos, metrics):
        """
        Initialize an experiment.

        Args:
            tasks: defines the various tasks. Each task describe the world, the rewards/costs,
            policies:
            algos:
            metrics:
        """
        self.tasks = tasks
        self.policies = policies
        self.algos = algos
        self.metrics = metrics

    def run(self, train=False, evaluate_metrics=False):
        """
        Run the experiment.

        Args:
            train (bool): if False, it does not train the policy(ies) using the algo(s).
            evaluate_metrics (bool): if False, it does not evaluate the policy using the metric.
        """

        # algos: train the rl on the given tasks
        if train:
            pass

        # metrics: evaluate the rl/algos using the given metrics
        if evaluate_metrics:
            pass

    def evaluate_policy(self):
        """
        Evaluate the policies on the tasks using their corresponding objective functions.
        """
        pass

    def plot_metrics(self):
        """
        Plot the results. This is let to the user to define this function.
        """
        pass

    def get_task(self, idx=None):
        pass

    def get_policy(self, idx=None):
        pass

    def get_state(self, idx=None, policy_idx=None):
        pass

    def get_action(self, idx=None, policy_idx=None):
        pass

    def get_environment(self, idx=None):
        pass

    def get_world(self, env_idx=None):
        pass

    def get_reward(self, env_idx=None, reward_idx=None):
        pass

    def get_algorithm(self, idx=None):
        pass

    def get_metric(self, idx=None):
        pass

    def change_simulator(self, simulator):
        pass


class GymExperiment(Experiment):
    r"""Gym Environment Experiment"""

    def __init__(self, gym_env, policy, algo):
        pass


class WalkingExperiment(Experiment):

    def __init__(self, policies=None, algos=None):
        tasks = [WalkingTask()]
        metrics = None

        if policies is None:
            pass

        if algos is None:
            algos = [PPO()]

        super(WalkingExperiment, self).__init__(tasks, metrics)
