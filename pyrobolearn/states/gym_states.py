#!/usr/bin/env python
"""Define the OpenAI Gym State

This defines the OpenAI Gym state such that it is compatible with the pyrobolearn framework. It notably decouples
loosely the states from the gym environment. Specifically, the `GymState` allows to extract the shape of the state
from the gym environment, and keep it as an attribute of the class. This can then be used by other classes such as
the various policies defined in the pyrobolearn framework.
"""

import gym
import copy

from pyrobolearn.states.state import State

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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

    def __init__(self, gym_env, window_size=1, axis=None, ticks=1):
        """
        Initialize the OpenAI Gym state.

        Args:
            gym_env (gym.Env): OpenAI gym environment
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """

        # check types
        if not isinstance(gym_env, gym.Env):
            raise TypeError("Expecting the `gym_env` argument to be an instance of the `gym.Env` class.")
        self.env = gym_env

        # set data and space
        space = self.env.observation_space
        data = space.sample()

        # call super constructor
        super(GymState, self).__init__(data=data, space=space, window_size=window_size, axis=axis, ticks=ticks)

    def _reset(self):
        pass

    def _read(self):
        pass

    def __copy__(self):
        """Return a shallow copy of the state. This can be overridden in the child class."""
        return self.__class__(gym_env=self.env, window_size=self.window_size, axis=self.axis, ticks=self.ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the state. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        env = memo.get(self.env, self.env)  # copy.deepcopy(self.env, memo)
        state = self.__class__(gym_env=env, window_size=self.window_size, axis=self.axis, ticks=self.ticks)

        memo[self] = state
        return state


# Tests
if __name__ == '__main__':
    # create environment
    env = gym.make('CartPole-v1')

    # create gym state
    states = GymState(env, window_size=1, axis=None)

    # print some information
    print("State: {}".format(states))
    print("Shape: {}".format(states.shape))
    print("Space: {}".format(states.space))
