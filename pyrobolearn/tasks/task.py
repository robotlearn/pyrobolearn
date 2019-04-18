#!/usr/bin/env python
"""Define the abstract 'Task'/'Scenario' class.

A task represents a certain learning paradigm, and use specific metrics with respect to that paradigm.
Learning paradigms include imitation learning (IL), reinforcement learning (RL), transfer learning (TL),
active learning (AL), etc.

IL, AL, and RL tasks groups the environment and policies together. The task allows you thus to run the process
that happens between the environment and the policies. That is, the environment produces the states and possible
rewards while the policies take the states and produce actions.

Tasks can be sequenced one after another and represented as a directed graph / state machine. For instance,
the first task might be to climb stairs, then the second one might be to open a door, etc. This allows to combine
different scenarios in a modular way.

Dependencies:
- `pyrobolearn.envs`
- `pyrobolearn.policies`
"""

import collections
import copy
import time
from abc import ABCMeta
from itertools import count
import numpy as np

from pyrobolearn.envs import Env, gym
from pyrobolearn.policies import Policy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Task(object):
    r"""Task class.

    The Task defines the policy (and thus the learning model), environment (with the world and rewards), the states,
    and actions. For some tasks, the robot and/or policy can be provided as an input to the task.
    The `Task` represents the second highest level of our `pyrobolearn` framework. It is notably independent of the
    learning algorithm that is used to train the policy (i.e. learning model).
    For instance, a 'walking task' should be independent of the robot, policy, learning algorithm, and sometimes,
    of the particular terrain. However, the task already defines the various rewards that might be useful for this
    one along with a default world.

    .. seealso: * The highest level of our framework is the `Experiment` class which defines in addition the metrics
                used to evaluate our tasks and algos. The algorithm is often defined in that class but can sometimes
                be given as an input.
                * The next lower level of our framework is the `Environment` and `Policy` classes.
                * If there are multiple tasks, the `Scheduler` class which organizes how to run them might interest
                  the user.

    The task is often given to the learning algorithm, which can then train the policy in the corresponding
    environment.
    """
    __metaclass__ = ABCMeta

    def __init__(self, environment, policies):
        if not isinstance(environment, (Env, gym.Env)):
            raise TypeError("Expecting 'environment' to be an instance of Env or gym.Env")
        if isinstance(policies, collections.Iterable):
            for policy in policies:
                if not isinstance(policy, Policy):
                    raise TypeError("Expecting 'policies' to be a list/tuple of Policy instances")
        elif isinstance(policies, Policy):
            policies = [policies]
        else:
            raise TypeError("Expecting 'policies' to be an instance of Policy, or list/tuple of policies")

        self.env = environment
        self.policies = policies
        self._done = False
        self._succeeded = False

    ##############
    # Properties #
    ##############

    @property
    def done(self):
        """
        Return if the task is done or not.
        """
        return self._done

    @property
    def succeeded(self):
        """
        Return if the task succeeded or not.
        """
        return self._succeeded

    @property
    def failed(self):
        """
        Check if the task failed or not.
        """
        return not self.succeeded

    @property
    def simulator(self):
        """
        Return the simulator.
        """
        return self.env.simulator

    @property
    def world(self):
        """
        Return the world instance.
        """
        return self.env.world

    @property
    def policy(self):
        """
        Return the policies.
        """
        if len(self.policies) == 1:
            return self.policies[0]
        return self.policies

    @property
    def learning_model(self):
        """
        Return the learning models.
        """
        if len(self.policies) == 1:
            return self.policies[0].model
        return [policy.model for policy in self.policies]

    @property
    def environment(self):
        """
        Return the environment.
        """
        return self.env

    @property
    def rewards(self):
        """
        Return the rewards.
        """
        return self.env.rewards

    @property
    def states(self):
        """
        Return the states.
        """
        return self.env.states

    @property
    def actions(self):
        """
        Return the actions.
        """
        if len(self.policies) == 1:
            return self.policies[0].actions
        return [policy.actions for policy in self.policies]

    ###########
    # Methods #
    ###########

    def is_finished(self):
        """
        Check if the task is finished or not.
        """
        return self.done

    def has_succeeded(self):
        """
        Check if the task has succeeded.
        """
        return self.succeeded

    def has_failed(self):
        """
        Check if the task has failed.
        """
        return self.failed

    def reset(self):
        """
        Reset the task; reset the environment and policies
        """
        # reset variables
        self._done = False
        self._succeeded = False
        # reset env and policies
        self.env.reset()
        for policy in self.policies:
            policy.reset()

    def run(self, num_steps=None, dt=0, use_terminating_condition=False, render=False):
        """
        Reset and run the task until it is done, or the current time step matches num_steps.
        """
        if num_steps is None:
            num_steps = np.infty

        # results = []
        total_rewards = np.zeros(len(self.policies))
        self.reset()

        for t in count():
            if t >= num_steps:
                break
            rewards = self.step(render=render)
            # result = self.step(render=render)
            # results.append(result)
            total_rewards += rewards
            if use_terminating_condition and self._done:
                break
            time.sleep(dt)

        # return results
        if total_rewards.size == 1:
            return total_rewards[0]
        return total_rewards

    def step(self, deterministic=True, render=False):
        """
        Perform one step.
        """
        if render:
            self.env.render()
        else:
            self.env.hide()

        # results = []
        rewards = []
        for policy in self.policies:
            # prev_obs = copy.deepcopy(policy.states.data)
            actions = policy.act(policy.states, deterministic=deterministic)
            obs, rew, done, info = self.env.step(actions)
            self._done = done
            # d = {'prev_obs': prev_obs, 'actions': copy.deepcopy(actions.data),
            #      'obs': copy.deepcopy(policy.states.data), 'rew': rew, 'done': done}
            # results.append(d)
            rewards.append(rew)
        # return results
        return np.array(rewards)

    def get_policy(self, idx=None):
        if idx is None:
            return self.policies
        return self.policies[idx]

    def get_learning_model(self, idx=None):
        if idx is None:
            return [policy.model for policy in self.policies]
        return self.policies[idx].model

    def save(self, filename):
        pass


# alias
Scenario = Task
