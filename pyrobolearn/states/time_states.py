#!/usr/bin/env python
"""Define the various time states

This includes notably the absolute, relative, and cumulative time states.
"""

from abc import ABCMeta
import time
import numpy as np
from state import State


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TimeState(State):
    r"""Time state (abstract class)"""
    __metaclass__ = ABCMeta
    pass


class AbsoluteTimeState(TimeState):
    """Absolute time state

    Returns the absolute time.
    """

    def __init__(self):
        data = np.array([time.time()])
        super(AbsoluteTimeState, self).__init__(data=data)

    def _read(self):
        self._data[0] = time.time()


class RelativeTimeState(TimeState):
    """Relative time state

    Returns the time difference from last time.
    """

    def __init__(self):
        data = np.array([0.0])
        super(RelativeTimeState, self).__init__(data=data)

    def _reset(self):
        self.current_time = time.time()
        self._data[0] = 0.0

    def _read(self):
        next_time = time.time()
        self._data[0] = next_time - self.current_time
        self.current_time = next_time


class CumulativeTimeState(TimeState):
    r"""Cumulative time state

    Return the cumulative time.
    """

    def __init__(self):
        data = np.array([0.0])
        super(CumulativeTimeState, self).__init__(data=data)

    def _reset(self):
        self._data[0] = 0.0
        self.current_time = time.time()

    def _read(self):
        next_time = time.time()
        self._data[0] += (next_time - self.current_time)
        self.current_time = next_time


class PhaseState(TimeState):
    r"""Phase State
    """

    def __init__(self, num_steps=100, max_value=1., rate=1):
        data = np.array([0.0])
        self.cnt = 0
        self.rate = rate
        self.max_value = max_value
        if num_steps < 2:
            num_steps = 2
        self.dphase = float(max_value) / (num_steps - 1)
        super(PhaseState, self).__init__(data=data)

    def _reset(self):
        self._data[0] = 0.0
        self.cnt = 0

    def _read(self):
        if (self.cnt % self.rate) == 0:
            if self._data[0] < self.max_value:
                self._data[0] += self.dphase
        self.cnt += 1


# Tests the different time states
if __name__ == '__main__':
    s = AbsoluteTimeState()
    print("\nAbsolute Time State:")
    print(s.reset())
    for i in range(10):
        print(s())

    s = RelativeTimeState()
    print("\nRelative Time State:")
    print(s.reset())
    for i in range(10):
        print(s())

    s = CumulativeTimeState()
    print("\nCumulative Time State:")
    print(s.reset())
    for i in range(10):
        print(s())

    combined = AbsoluteTimeState() + RelativeTimeState() + CumulativeTimeState()
    fused = AbsoluteTimeState() & RelativeTimeState() & CumulativeTimeState()

    print("\nCombined state: {}".format(combined))
    print("\nFused state: {}".format(fused))
    for i in range(4):
        print(combined.read())
        print(fused.read())
