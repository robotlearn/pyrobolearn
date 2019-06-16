#!/usr/bin/env python
"""Define basic states

This includes notably the fixed, functional, and counter states.
"""

import numpy as np

from pyrobolearn.states.state import State
from pyrobolearn.actions import Action


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FixedState(State):
    r"""Fixed State.

    This is a dummy fixed state which always returns the value it was initialized with.
    """

    def __init__(self, value, window_size=1, axis=None, ticks=1):
        """
        Initialize the dummy fixed state.

        Args:
            value (int, float, object): always return this value.
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
        super(FixedState, self).__init__(data=value, window_size=window_size, axis=axis, ticks=ticks)


class FunctionalState(State):
    r"""Functional State.

    This is a state which accepts a function which has to output the data.
    """

    def __init__(self, function, window_size=1, axis=None, ticks=1, *args, **kwargs):
        """
        Initialize the functional state.

        Args:
            function (callable): callable function or class that has to output the next state data.
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
            *args: list of arguments given to the function.
            **kwargs: dictionary of arguments given to the function.
        """
        self.function = function
        self.args, self.kwargs = args, kwargs
        data = function(*args, **kwargs)  # call one time to get data
        super(FunctionalState, self).__init__(data=data, window_size=window_size, axis=axis, ticks=ticks)

    def _reset(self):
        """Reset the functional state."""
        self.data = self.function(*self.args, **self.kwargs)

    def _read(self):
        """Read the next functional state data."""
        self.data = self.function(*self.args, **self.kwargs)


class CounterState(State):
    r"""Counter State.

    Counts the number of time this step has been called.
    """

    def __init__(self, cnt=0, window_size=1, axis=None, ticks=1):
        """
        Initialize the counter state.

        Args:
            cnt (int): initial value for the counter.
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
        self.count = cnt
        if isinstance(cnt, int):
            cnt = np.array([cnt])
        if not (isinstance(cnt, np.ndarray) and cnt.size == 1 and len(cnt.shape) == 1
                and cnt.dtype.kind in np.typecodes['AllInteger']):
            raise TypeError("Expecting an int, or a numpy array (integer) with size 1")
        super(CounterState, self).__init__(data=cnt, window_size=window_size, axis=axis, ticks=ticks)

    def _reset(self):
        """Reset the counter state."""
        self.data = self.count

    def _read(self):
        """Read the next counter state."""
        self.data = self.last_data + 1


class PreviousActionState(State):
    r"""Previous Action State

    This state copies the previous action data.
    """

    def __init__(self, action, window_size=1, axis=None, ticks=1):
        """
        Initialize the previous action state.

        Args:
            action (Action): action to copy the data from.
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
        if not isinstance(action, Action):
            raise TypeError("Expecting the action to be an instance of Action, instead got {}".format(action))
        self.action = action
        super(PreviousActionState, self).__init__(window_size=window_size, axis=axis, ticks=ticks)

    def _reset(self):
        """Reset the action state."""
        self.data = self.action.data

    def _read(self):
        """Read the next action state."""
        self.data = self.action.data


# Tests
if __name__ == '__main__':
    s1 = FixedState([1, 2])
    s2 = FixedState(3)
    s3 = FixedState([4, 5, 6])
    s = s1 + s2 + s2 + s3 + s1
    fused = s1 & s2

    print("\nStates:")
    print("s1 = {}".format(s1))
    print("s2 = {}".format(s2))
    print("s3 = {}".format(s3))
    print("s = s1 + s2 + s2 + s3 + s1 = {}".format(s))
    print("s1.fuse() = {}".format(s1.fuse()))
    print("s.fuse() = {}".format(s.fuse()))
    print("fused = s1 & s2 = {}".format(fused))

    print("\nSome dimensions:")
    print("s.shape: {}".format(s.shape))
    print("s.dimension: {}".format(s.dimension))
    print("s.max_dimension: {}".format(s.max_dimension()))
    print("s.size: {}".format(s.size))
    print("s.total_size: {}".format(s.total_size()))
    print("len(s) = {}".format(len(s)))
    print("len(s1) = {}".format(len(s1)))
    # print(s2 + s1)

    print("\nIndexing: ")
    print("s[0] = {}".format(s[0]))
    print("s[1:3] = {}".format(s[1:3]))

    print("s1 = {}".format(s1))
    s1[1] = 7
    print("s1[1] = 7 --> {}".format(s1))
    print("s = {}".format(s))
    s[0] = FixedState([8, 9, 10, 11])
    print("s[0] = [8,9] --> {}".format(s))
    # for state in s:
    #    print(state)

    print("\nDifference: ")
    print("s - s2 = {}".format(s - s2))

    s = CounterState()
    print("\nCounter State:")
    print(s.reset())
    for i in range(10):
        print(s())
