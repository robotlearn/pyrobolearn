#!/usr/bin/env python
"""Provide a wrapper around the gym framework and the `gym.Env` class

This provides a wrapper around the gym framework such that is compatible with the pyrobolearn framework.
With this, users can use everything the various tools (models, algos, etc) defined in the pyrobolearn framework
on the OpenAI gym library.
"""

import inspect
import functools
import numpy as np
import gym
from gym import *
import warnings
warnings.simplefilter("ignore")

from pyrobolearn.states.gym_states import GymState
from pyrobolearn.actions.gym_actions import GymAction, Action
# from pyrobolearn.rewards import GymReward
# from terminating_condition import GymTerminatingCondition


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def make(env_id):
    """Create the OpenAI Gym environment and return a wrapped version of it."""
    env = gym.make(env_id)
    env = GymEnvWrapper(env)
    return env


def create(env_id):
    """Create the OpenAI Gym environment and return a wrapped version of it with the associated state and action."""
    env = gym.make(env_id)
    env = GymEnvWrapper(env)
    return env, env.state, env.action


class GymEnvWrapper(gym.Env):
    r"""Gym Environment wrapper

    This update the data of the GymState, GymAction, and GymReward when performing a step in a gym environment.
    """

    def __init__(self, env, state=None, action=None):
        # set environment
        self.env = env

        # set state and action
        self.state = state
        self.action = action

        # define the observation and action space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # self.reward = GymReward(1)
        # self.done = GymTerminatingCondition(done=False)

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, environment):
        if not isinstance(environment, gym.Env):
            raise TypeError("Expecting the given environment to be an instance of `gym.Env`")
        self._env = environment

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        if s is None:
            s = GymState(self.env)
        self._state = s

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, a):
        if a is None:
            a = GymAction(self.env)
        self._action = a

    def __getattr__(self, name):
        """Get the functions from the Gym Environment"""
        attribute = getattr(self.env, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute)
        return attribute

    def step(self, actions):
        """perform a step in the environment and set the data for the GymState and GymAction"""
        if isinstance(actions, Action):
            actions = actions.data[0]
        if isinstance(self.action_space, gym.spaces.Discrete) and isinstance(actions, np.ndarray):
            actions = actions[0]
        observations, reward, done, info = self.env.step(actions)
        self.state.data = observations
        self.action.data = actions
        # self.reward.value = reward
        # self.done.done = done
        return self.state, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.state.data = observation
        return self.state

    def render(self, mode='human'):
        self.env.render(mode)

    def __repr__(self):
        return self.env.__repr__()

    def __str__(self):
        return self.env.__str__()


# Test
if __name__ == '__main__':
    env, state, action = create('CartPole-v1')

    print("Env: {}".format(env))

    print("\nState: {}".format(state))
    print("-- shape: {}".format(state.shape))
    print("-- space: {}".format(state.space))

    print("\nAction: {}".format(action))
    print("-- shape: {}".format(action.shape))
    print("-- space: {}".format(action.space))
