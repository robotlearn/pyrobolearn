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
__license__ = "MIT"
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
        self.data = time.time()


class RelativeTimeState(TimeState):
    """Relative time state

    Returns the time difference from last time.
    """

    def __init__(self):
        data = np.array([0.0])
        super(RelativeTimeState, self).__init__(data=data)

    def _reset(self):
        self.current_time = time.time()
        self.data = 0.0

    def _read(self):
        next_time = time.time()
        self.data = next_time - self.current_time
        self.current_time = next_time


class CumulativeTimeState(TimeState):
    r"""Cumulative time state

    Return the cumulative time.
    """

    def __init__(self):
        data = np.array([0.0])
        super(CumulativeTimeState, self).__init__(data=data)

    def _reset(self):
        self.data = 0.0
        self.current_time = time.time()

    def _read(self):
        next_time = time.time()
        self.data = self._data + (next_time - self.current_time)
        self.current_time = next_time


class PhaseState(TimeState):
    r"""(Linear) Phase State

    Each call to the phase state will forward linearly in time with a value of `(end - start) / (num_steps - 1)`.
    This means, it will take for the phase `num_steps` to reach the `end` value starting from the `start` value.
    Once `end` is reached, it will stop forwarding in time, and will return that `end` value.
    """

    def __init__(self, num_steps=100, start=0, end=1., rate=1):
        self.cnt = 0
        self.num_steps = num_steps
        self.rate = rate
        self.end_value = end
        self.start_value = start
        self.sign = np.sign(end - start)
        if num_steps < 2:
            num_steps = 2
        self.dphase = float((end - start) / (num_steps - 1.))
        data = np.array([start]) - self.dphase
        super(PhaseState, self).__init__(data=data)

    def _reset(self):
        self.data = np.array([self.start_value]) - self.dphase
        self.cnt = 0

    def _read(self):
        if (self.cnt % self.rate) == 0:
            if self.sign > 0 and self._data[0] < self.end_value:
                self.data = np.minimum(self._data + self.dphase, self.end_value)
            elif self.sign < 0 and self._data[0] > self.end_value:
                self.data = np.maximum(self._data + self.dphase, self.end_value)
        self.cnt += 1


class ExponentialPhaseState(TimeState):
    r"""Exponential Phase State

    Let's assume the phase is described by the following differential equation: `ds/dt = a * s(t)` then solving
    it results in `s(t) = s(0) * exp(a * t)`. Initially, `t` starts from `t_0` and is incremented by `dt` at each call,
    and reaches `t_f` after `num_steps` specified by the user.

    This class is notably useful for Phase states that decay exponentially.
    """
    def __init__(self, num_steps=100, s0=1., sf=None, t0=0., tf=1., a=-1., rate=1):
        """

        Args:
            num_steps (int): number of steps to reach `T`.
            s0 (float): initial phase value.
            sf (float): possible end phase value. Depending on the sign of `a`, it will stop
            t0 (float): initial time value.
            tf (float): final time value. With the `num_steps` it allows the computation of `dt`.
            a (float): speed constant.
            rate (int): rate at which to update the phase
        """
        if tf < t0:
            raise ValueError("The final time value must be bigger than the inital time value; we don't go back in "
                             "time!")
        self.cnt = 0
        self.rate = rate
        self.t0, self.tf = t0, tf
        self.dt = (tf - t0) / (num_steps - 1)
        self.t = self.t0 - self.dt
        self.s0, self.sf = s0, sf
        self.a = a
        data = np.array([self.s0]) * np.exp(self.a * self.t)
        super(ExponentialPhaseState, self).__init__(data=data)

    def _reset(self):
        self.cnt = 0
        self.t = self.t0 - self.dt
        self.data = np.array([self.s0]) * np.exp(self.a * self.t)

    def _read(self):
        if (self.cnt % self.rate) == 0:
            self.t += self.dt
            if self.t < self.tf:
                self.data = np.array([self.s0]) * np.exp(self.a * self.t)
                if self.sf is not None:
                    if self.a < 0:
                        self.data = np.maximum(self._data, self.sf)
                    elif self.a > 0:
                        self.data = np.minimum(self._data, self.sf)
        self.cnt += 1


# alias
DecayPhaseState = ExponentialPhaseState


class RhythmicPhase(PhaseState):
    r"""Rhythmic Phase state

    The PhaseState starts from `start` and ends with `end`, calling after `num_steps` will just return `end`.
    In this class, we cycle through the phase `[start, end[`; that is, once the end is reached it restarts
    automatically from `start`
    """

    def __init__(self, num_steps=100, start=0, end=1., rate=1):
        super(RhythmicPhase, self).__init__(num_steps=num_steps, start=start, end=end, rate=rate)

    def _read(self):
        if (self.cnt % self.rate) == 0:
            self.data = self._data + self.dphase
            if self.sign > 0 and self._data[0] >= self.end_value:
                self.data = np.array([self.start_value])
            if self.sign < 0 and self._data[0] <= self.end_value:
                self.data = np.array([self.start_value])
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
        print(s.torch_data)

    s = PhaseState(num_steps=100, start=1, end=-1)
    print("\nPhase Time State:")
    print(s.reset())
    for i in range(100):
        print(s())

    s = ExponentialPhaseState(num_steps=100, s0=1, a=-1)
    print("\nPhase Time State:")
    print(s.reset())
    for i in range(200):
        print(s())

    s = RhythmicPhase(num_steps=10, start=1, end=2)
    print("\nPhase Time State:")
    print(s.reset())
    for i in range(100):
        print(s())

    combined = AbsoluteTimeState() + RelativeTimeState() + CumulativeTimeState()
    fused = AbsoluteTimeState() & RelativeTimeState() & CumulativeTimeState()

    print("\nCombined state: {}".format(combined))
    print("\nFused state: {}".format(fused))
    for i in range(4):
        print(combined.read())
        print(fused.read())
