#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the vectorized PRL environment.
"""

import numpy as np
import gym
from stable_baselines.common.vec_env import VecEnv  # , VecNormalize
import warnings
warnings.simplefilter("ignore")

from pyrobolearn.envs.env import Env


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class VecPRLEnv(VecEnv):
    r"""Vectorized PRL environment.
    """

    def __init__(self, env, num_envs=1):
        """
        Initialize the Vectorized environment.

        Args:
            env (Env): PRL environment.
            num_envs (int): number of environments.
        """

        # check the environment type
        if not isinstance(env, Env):
            raise TypeError("Expecting the given `env` to be an instance of `Env`, but got instead: "
                            "{}".format(type(env)))
        self.env = env

        self.num_envs = num_envs if isinstance(num_envs, int) else 1

        observation_space = env.observation_space
        action_space = env.action_space
        if action_space is None:
            raise ValueError("The action space has not been defined for the given environment.")

        super(VecPRLEnv, self).__init__(num_envs=num_envs, observation_space=observation_space,
                                        action_space=action_space)

    def reset(self):
        """
        Reset all the environments and return an array of observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        Returns:
            list[int, np.array[int]], list[float, np.array[float]]: observation
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environment's resources.
        """
        pass

    @abstractmethod
    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        pass

    @abstractmethod
    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        pass

    @abstractmethod
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        pass

    def step(self, actions):
        """
        Step the environments with the given action

        :param actions: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def render(self, *args, **kwargs):
        """
        Gym environment rendering

        :param mode: (str) the rendering type
        """
        logger.warn('Render not defined for %s' % self)
