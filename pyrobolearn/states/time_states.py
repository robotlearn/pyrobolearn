#!/usr/bin/env python
"""Define the various time states

This includes notably the absolute, relative, and cumulative time states.
"""

from abc import ABCMeta
import time
import numpy as np

from pyrobolearn.states.state import State


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

    def __init__(self, window_size=1, axis=None, ticks=1):
        """
        Initialize the absolule time state.

        Args:
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
        data = np.array([time.time()])
        super(AbsoluteTimeState, self).__init__(data=data, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next absolute time state."""
        self.data = np.array([time.time()])


class RelativeTimeState(TimeState):
    """Relative time state

    Returns the time difference from last time.
    """

    def __init__(self, window_size=1, axis=None, ticks=1):
        """
        Initialize the relative time state.

        Args:
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
        data = np.array([0.0])
        super(RelativeTimeState, self).__init__(data=data, window_size=window_size, axis=axis, ticks=ticks)

    def _reset(self):
        """Reset the relative time state."""
        self.current_time = time.time()
        self.data = np.array([0.0])

    def _read(self):
        """Read the next relative time state."""
        next_time = time.time()
        self.data = next_time - self.current_time
        self.current_time = next_time


class CumulativeTimeState(TimeState):
    r"""Cumulative time state

    Return the cumulative time.
    """

    def __init__(self, window_size=1, axis=None, ticks=1):
        """
        Initialize the cumulative time state.

        Args:
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
        data = np.array([0.0])
        super(CumulativeTimeState, self).__init__(data=data, window_size=window_size, axis=axis, ticks=ticks)

    def _reset(self):
        """Reset the cumulative time state."""
        self.data = np.array([0.0])
        self.current_time = time.time()

    def _read(self):
        """Read the next cumulative time state."""
        next_time = time.time()
        self.data = self.last_data + (next_time - self.current_time)
        self.current_time = next_time


class PhaseState(TimeState):
    r"""(Linear) Phase State

    Each call to the phase state will forward linearly in time with a value of `(end - start) / (num_steps - 1)`.
    This means, it will take for the phase `num_steps` to reach the `end` value starting from the `start` value.
    Once `end` is reached, it will stop forwarding in time, and will return that `end` value.
    """

    def __init__(self, num_steps=100, start=0, end=1., window_size=1, axis=None, ticks=1):
        """
        Initialize the linear phase state.

        Args:
            num_steps (int): the number of time steps.
            start (float): initial phase value.
            end (float): final phase value. Once the phase value is bigger than the final phase value, it will
                always return that final phase value.
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
        self.num_steps = num_steps
        self.end_value = end
        self.start_value = start
        self.sign = np.sign(end - start)
        if num_steps < 2:
            num_steps = 2
        self.dphase = float((end - start) / (num_steps - 1.))
        data = np.array([start])  # - self.dphase
        super(PhaseState, self).__init__(data=data, window_size=window_size, axis=axis, ticks=ticks)

    def _reset(self):
        """Reset the linear phase state."""
        self.data = np.array([self.start_value])  # - self.dphase

    def _read(self):
        """Read the next linear phase state."""
        if self.sign > 0 and self.last_data[0] < self.end_value:
            self.data = np.minimum(self.last_data + self.dphase, self.end_value)
        elif self.sign < 0 and self.last_data[0] > self.end_value:
            self.data = np.maximum(self.last_data + self.dphase, self.end_value)


class ExponentialPhaseState(TimeState):
    r"""Exponential Phase State

    Let's assume the phase is described by the following differential equation: `ds/dt = a * s(t)` then solving
    it results in `s(t) = s(0) * exp(a * t)`. Initially, `t` starts from `t_0` and is incremented by `dt` at each call,
    and reaches `t_f` after `num_steps` specified by the user.

    This class is notably useful for Phase states that decay exponentially.
    """

    def __init__(self, num_steps=100, s0=1., sf=None, t0=0., tf=1., a=-1., window_size=1, axis=None, ticks=1):
        """
        Initialize the exponential phase state.

        Args:
            num_steps (int): number of steps to reach `T`.
            s0 (float): initial phase value.
            sf (float): possible end phase value. Depending on the sign of `a`, it will stop
            t0 (float): initial time value.
            tf (float): final time value. With the `num_steps` it allows the computation of `dt`.
            a (float): speed constant.
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
        if tf < t0:
            raise ValueError("The final time value must be bigger than the inital time value; we don't go back in "
                             "time!")
        self.t0, self.tf = t0, tf
        self.dt = (tf - t0) / (num_steps - 1)
        self.t = self.t0  # - self.dt
        self.s0, self.sf = s0, sf
        self.a = a
        data = np.array([self.s0]) * np.exp(self.a * self.t)
        super(ExponentialPhaseState, self).__init__(data=data, window_size=window_size, axis=axis, ticks=ticks)

    def _reset(self):
        """Reset the exponential phase state."""
        self.t = self.t0  # - self.dt
        self.data = np.array([self.s0]) * np.exp(self.a * self.t)

    def _read(self):
        """Read the next exponential phase state."""
        self.t += self.dt
        if self.t < self.tf:
            self.data = np.array([self.s0]) * np.exp(self.a * self.t)
            if self.sf is not None:
                if self.a < 0:
                    self.data = np.maximum(self.last_data, self.sf)
                elif self.a > 0:
                    self.data = np.minimum(self.last_data, self.sf)


# alias
DecayPhaseState = ExponentialPhaseState


class RhythmicPhase(PhaseState):
    r"""Rhythmic Phase state

    The PhaseState starts from `start` and ends with `end`, calling after `num_steps` will just return `end`.
    In this class, we cycle through the phase `[start, end[`; that is, once the end is reached it restarts
    automatically from `start`
    """

    def __init__(self, num_steps=100, start=0, end=1., window_size=1, axis=None, ticks=1):
        """
        Initialize the rhythmic phase state.

        Args:
            num_steps (int): the number of time steps.
            start (float): initial phase value.
            end (float): final phase value. Once the phase value is bigger than the final phase value, it will
                always return that final phase value.
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
        super(RhythmicPhase, self).__init__(num_steps=num_steps, start=start, end=end,
                                            window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next rhythmic phase state."""
        self.data = self.last_data + self.dphase
        if self.sign > 0 and self.last_data[0] >= self.end_value:
            self.data = np.array([self.start_value])
        if self.sign < 0 and self.last_data[0] <= self.end_value:
            self.data = np.array([self.start_value])


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

    s = CumulativeTimeState()  # window_size=2, axis=0, ticks=2)
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
    for i in range(110):
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
        print("combined.read: {}".format(combined.read()))
        print("fused.read: {}".format(fused.read()))

    print("Fused does not update the data...")

    s1 = CumulativeTimeState()
    s2 = PhaseState()
    s3 = AbsoluteTimeState()
    s_c1 = s1 + s2
    s_c2 = s2 + s3
    s = s_c1 + s_c2
    print("In the following, we just update the s_c1[s1, s2]: \n")
    for i in range(3):
        print("s[s_c1, s_c2]: {}".format(s))
        print("s_c1[s1,s2]: {}".format(s_c1))
        print("s_c2[s2,s3]: {}".format(s_c2))
        s_c1()
        print("")
        print(s.data)
        print(s.merged_data)
        print(s1.merged_data)
