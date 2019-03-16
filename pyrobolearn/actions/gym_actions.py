#!/usr/bin/env python
"""Define the OpenAI Gym Action

This defines the OpenAI Gym action such that it is compatible with the pyrobolearn framework. It notably decouples
loosely the actions from the gym environment. Specifically, the `GymAction` allows to extract the shape of the action
from the gym environment, and keep it as an attribute of the class. This can then be used by other classes such as
the various policies defined in the pyrobolearn framework.
"""

import gym
from action import Action

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GymAction(Action):
    r"""OpenAI Gym Action.
    """

    def __init__(self, gym_env):
        """
        Initialize the OpenAI Gym action.

        Args:
            gym_env (gym.Env): OpenAI gym environment
        """

        # check types
        if not isinstance(gym_env, gym.Env):
            raise TypeError("Expecting the `gym_env` argument to be an instance of the `gym.Env` class.")
        self.env = gym_env

        # set data and space
        space = self.env.action_space
        data = space.sample()

        # call super constructor
        super(GymAction, self).__init__(data=data, space=space)

    def _write(self, data=None):
        pass


# Tests
if __name__ == '__main__':
    import gym

    # create environment
    env = gym.make('CartPole-v1')

    # create gym action
    actions = GymAction(env)

    # print some information
    print("Action: {}".format(actions))
    print("Shape: {}".format(actions.shape))
    print("Space: {}".format(actions.space))
