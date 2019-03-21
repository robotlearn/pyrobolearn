#!/usr/bin/env python
"""Define the OpenAI Gym State

This defines the OpenAI Gym state such that it is compatible with the pyrobolearn framework. It notably decouples
loosely the states from the gym environment. Specifically, the `GymState` allows to extract the shape of the state
from the gym environment, and keep it as an attribute of the class. This can then be used by other classes such as
the various policies defined in the pyrobolearn framework.
"""

import gym

from pyrobolearn.states.state import State

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GymState(State):
    r"""OpenAI Gym State.

    This defines the OpenAI Gym state such that it is compatible with the pyrobolearn framework. It notably decouples
    loosely the states from the gym environment. Specifically, the `GymState` allows to extract the shape of the state
    from the gym environment, and keep it as an attribute of the class. This can then be used by other classes such as
    the various policies defined in the pyrobolearn framework.

    See Also: `GymEnv`
    """

    def __init__(self, gym_env):
        """
        Initialize the OpenAI Gym state.

        Args:
            gym_env (gym.Env): OpenAI gym environment
        """

        # check types
        if not isinstance(gym_env, gym.Env):
            raise TypeError("Expecting the `gym_env` argument to be an instance of the `gym.Env` class.")
        self.env = gym_env

        # set data and space
        space = self.env.observation_space
        data = space.sample()

        # call super constructor
        super(GymState, self).__init__(data=data, space=space)

    def _reset(self):
        pass

    def _read(self):
        pass


# Tests
if __name__ == '__main__':
    # create environment
    env = gym.make('CartPole-v1')

    # create gym state
    states = GymState(env)

    # print some information
    print("State: {}".format(states))
    print("Shape: {}".format(states.shape))
    print("Space: {}".format(states.space))
