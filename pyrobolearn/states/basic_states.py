#!/usr/bin/env python
"""Define basic states

This includes notably the fixed, functional, and counter states.
"""

import numpy as np

from pyrobolearn.states.state import State
from pyrobolearn.actions.action import Action


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
    def __init__(self, value):
        super(FixedState, self).__init__(data=value)


class FunctionalState(State):
    r"""Functional State.

    This is a state which accepts a function which has to output the data.
    """
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args, self.kwargs = args, kwargs
        data = function(*args, **kwargs)  # call one time to get data
        super(FunctionalState, self).__init__(data=data)

    def _reset(self):
        self.data = self.function(*self.args, **self.kwargs)

    def _read(self):
        self.data = self.function(*self.args, **self.kwargs)


class CounterState(State):
    r"""Counter State.

    Counts the number of time this step has been called.
    """

    def __init__(self, cnt=-1):
        self.cnt = cnt
        if isinstance(cnt, int):
            cnt = np.array([cnt])
        if not (isinstance(cnt, np.ndarray) and cnt.size == 1 and len(cnt.shape) == 1
                and cnt.dtype.kind in np.typecodes['AllInteger']):
            raise TypeError("Expecting an int, or a numpy array (integer) with size 1")
        super(CounterState, self).__init__(data=cnt)

    def _reset(self):
        self.data = self.cnt

    def _read(self):
        self.data = self._data + 1


class PreviousActionState(State):
    r"""Previous Action State

    This state copies the previous action.
    """

    def __init__(self, action):
        if not isinstance(action, Action):
            raise TypeError("Expecting the action to be an instance of Action, instead got {}".format(action))
        self.action = action
        super(PreviousActionState, self).__init__()

    def _reset(self):
        self.data = self.action.data

    def _read(self):
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
    print("s.maxDimension: {}".format(s.maxDimension()))
    print("s.size: {}".format(s.size))
    print("s.totalSize: {}".format(s.totalSize()))
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
