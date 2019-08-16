#!/usr/bin/env python
"""Provide a wrapper around the gym framework and the `gym.Env` class

This provides a wrapper around the gym framework such that is compatible with the pyrobolearn framework.
With this, users can use everything the various tools (models, algos, etc) defined in the pyrobolearn framework
on the OpenAI gym library.
"""

import inspect
import functools
import numpy as np
import torch
import gym
# import baselines
import warnings
warnings.simplefilter("ignore")

from pyrobolearn.states.gym_states import GymState, State
from pyrobolearn.actions.gym_actions import GymAction, Action
from pyrobolearn.states.processors import StateProcessor
from pyrobolearn.rewards.processors import RewardProcessor
from pyrobolearn.rewards import GymReward
# from terminating_condition import GymTerminatingCondition


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def from_gym(env, state_processors=None, reward_processors=None):
    """wraps a gym environment.

    Args:
        env (gym.Env): gym environment.
        state_processors ((a list of) StateProcessor types/classes, None): State processors.
        reward_processors ((a list of) RewardProcessor types/classes, None): Reward processors.
    """
    env = GymEnvWrapper(env, state_processors, reward_processors)
    return env


def make(env_id, state_processors=None, reward_processors=None):
    """Create the OpenAI Gym environment and return a wrapped version of it.

    Args:
        env_id (str): gym environment string.
        state_processors ((a list of) StateProcessor types/classes, None): State processors.
        reward_processors ((a list of) RewardProcessor types/classes, None): Reward processors.
    """
    return from_gym(gym.make(env_id), state_processors, reward_processors)


def create(env_id, state_processors=None, reward_processors=None):
    """Create the OpenAI Gym environment and return a wrapped version of it with the associated state and action.

    Args:
        env_id (str): gym environment string.
        state_processors ((a list of) StateProcessor types/classes, None): State processors.
        reward_processors ((a list of) RewardProcessor types/classes, None): Reward processors.
    """
    env = from_gym(gym.make(env_id), state_processors, reward_processors)
    return env, env.state, env.action


class GymEnvWrapper(gym.Env):
    r"""Gym Environment wrapper

    This update the data of the GymState, GymAction, and GymReward when performing a step in a gym environment.
    """

    def __init__(self, env, state_processors=None, reward_processors=None):
        """
        Initialize the gym Env wrapper.

        Args:
            env (gym.Env): gym environment.
            state_processors ((a list of) StateProcessor types/classes): State processors.
            reward_processors ((a list of) RewardProcessor types/classes): Reward processors.
        """
        # set environment
        self.env = env

        # set state, action, and reward
        self._state = GymState(self.env)
        self._action = GymAction(self.env)
        self._reward = GymReward(0., range=self.env.reward_range)
        # self.done = GymTerminatingCondition(done=False)

        # define the observation and action space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # set the processors
        self.state_processors = state_processors
        self.reward_processors = reward_processors

        # rendering
        self.is_rendering = False

    ##############
    # Properties #
    ##############

    @property
    def env(self):
        """Return the environment instance."""
        return self._env

    @env.setter
    def env(self, environment):
        """Set the environment instance."""
        if not isinstance(environment, gym.Env):  # , baselines.common.vec_env.VecEnv)):
            raise TypeError("Expecting the given environment to be an instance of `gym.Env`")
        self._env = environment
        self._state = GymState(self.env)
        self._action = GymAction(self.env)

    @property
    def state(self):
        """Return the state instance."""
        return self._state

    # @state.setter
    # def state(self, s):
    #     """Set the state; if None, wraps the `gym.Env` to get the state."""
    #     if s is None:
    #         s =
    #     elif not isinstance(s, GymState):
    #         raise TypeError("Expecting the given state to be None or an instance of `GymState`, instead got: "
    #                         "{}".format(type(s)))
    #     self._state = s

    @property
    def action(self):
        """Return the action instance."""
        return self._action

    # @action.setter
    # def action(self, a):
    #     """Set the action; wraps the `gym.Env` to get the original action."""
    #     if a is None:
    #         a = GymAction(self.env)
    #     elif not isinstance(a, GymAction):
    #         raise TypeError("Expecting the given action to be None or an instance of `GymAction`, instead got: "
    #                         "{}".format(type(s)))
    #     self._action = a

    @property
    def reward(self):
        """Return the reward instance."""
        return self._reward

    @property
    def reward_range(self):
        """Return the range of the reward function; a tuple corresponding to the min and max possible rewards"""
        return self._reward.range

    @property
    def state_processors(self):
        """Return the processors that have to be applied on the state."""
        return self._state_processors

    @state_processors.setter
    def state_processors(self, processors):
        """Set the state processors."""
        if processors is None:
            processors = []
        elif issubclass(processors, StateProcessor):
            processors = [processors]
        elif isinstance(processors, (list, tuple, set)):
            for idx, processor in enumerate(processors):
                if not issubclass(processor, StateProcessor):
                    raise TypeError("Expecting the {} item to be a subclass of `StateProcessor`, instead got: "
                                    "{}".format(idx, type(processor)))
        else:
            raise TypeError("Expecting the given state processors to be None, or a (list of) subclass of "
                            "`StateProcessor`, instead got: {}".format(type(processors)))

        state = self.state
        for processor in processors:
            state = processor(state)
        self._state_processors = state

    @property
    def reward_processors(self):
        """Return the processors that have to be applied on the reward."""
        return self._reward_processors

    @reward_processors.setter
    def reward_processors(self, processors):
        """Set the reward processors."""
        if processors is None:
            processors = []
        elif issubclass(processors, RewardProcessor):
            processors = [processors]
        elif isinstance(processors, (list, tuple, set)):
            for idx, processor in enumerate(processors):
                if not issubclass(processor, RewardProcessor):
                    raise TypeError("Expecting the {} item to be a subclass of `RewardProcessor`, instead got: "
                                    "{}".format(idx, type(processor)))
        else:
            raise TypeError("Expecting the given reward processors to be None, or a (list of) subclass of "
                            "`RewardProcessor`, instead got: {}".format(type(processors)))

        reward = self.reward
        for processor in processors:
            reward = processor(reward)
        self._reward_processors = reward

    ###########
    # Methods #
    ###########

    @staticmethod
    def _convert_state_to_data(states, convert=True):
        """Convert a `State` to a list of numpy arrays or a numpy array."""
        if convert:
            data = []
            for state in states:
                if isinstance(state, State):
                    state = state.merged_data
                if isinstance(state, list) and len(state) == 1:
                    state = state[0]
                data.append(state)
            if len(data) == 1:
                data = data[0]
            return data
        return states

    def reset(self):
        """Reset the gym environment."""
        observation = self.env.reset()
        self.state.data = observation
        self.reward_processors.reset()
        return self._convert_state_to_data(self.state)

    def step(self, actions):
        """perform a step in the gym environment and set the data for the GymState and GymAction"""
        if isinstance(actions, Action):  # convert from `Action` to numpy
            actions = actions.merged_data[0]
        elif isinstance(actions, torch.Tensor):  # convert from torch to numpy
            if actions.requires_grad:
                actions = actions.detach().numpy()
            else:
                actions = actions.numpy()
        if isinstance(self.action_space, gym.spaces.Discrete) and isinstance(actions, np.ndarray):
            actions = actions[0]  # float

        # perform a step with the original gym environment
        observations, reward, done, info = self.env.step(actions)

        # set the data
        self.state.data = observations
        self.action.data = actions
        self.reward.value = reward
        # self.done.done = done

        # process the data
        observations = self.state_processors()
        reward = self.reward_processors()

        observations = self._convert_state_to_data(observations)

        # return the data
        return observations, reward, done, info

    def render(self, mode='human'):
        """Render the gym environment."""
        self.is_rendering = True
        self.env.render(mode)

    def hide(self):
        """Hide the gym environment (not used)."""
        self.is_rendering = False

    def close(self):
        """Close the gym environment."""
        self.env.close()

    def seed(self, seed=None):
        """Set the given seed to the gym environment."""
        self.env.seed(seed)

    #############
    # Operators #
    #############

    def __getattr__(self, name):
        """Get the functions from the Gym Environment"""
        attribute = getattr(self.env, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute)
        return attribute

    # def __repr__(self):
    #     """Return a representing object."""
    #     return self.env.__repr__()

    def __str__(self):
        """Return a string describing the class."""
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
