#!/usr/bin/env python
"""Define some common terminal conditions for the environment.
"""

import copy
import numpy as np

from pyrobolearn.terminal_conditions.terminal_condition import TerminalCondition


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TimeLimitCondition(TerminalCondition):
    r"""Time Limit Terminal Condition

    """

    def __init__(self, num_steps, btype='neutral'):
        """
        Initialize the time limit terminal condition.

        Args:
            num_steps (int):
            btype (bool, str, None): if the terminal condition represents a failure or success condition. If None, it
                represents a neutral terminal condition (which is neither a failure or success condition, but just
                means the episode is over). If string, it has to be among {"success", "failure", "neutral"}.
        """
        super(TimeLimitCondition, self).__init__(btype=btype)
        self.num_steps = num_steps
        self.cnt = 0

    def reset(self):
        """
        Reset the terminal condition.
        """
        self.cnt = 0

    def check(self):
        """
        Check if the terminating condition has been fulfilled, and return True or False accordingly
        """
        if self.cnt >= self.num_steps:
            self._over = True
            return self._over
        self.cnt += 1
        return self._over


class GymTerminalCondition(TerminalCondition):
    r"""OpenAI Gym Terminal Condition

    Returns if the OpenAI Gym environment has terminated. This does not provide any information if the environment
    terminated because the policy succeeded or failed to perform the task.
    """

    def __init__(self, done=False):
        """
        Initialize the Gym terminal condition.

        Args:
            done (bool): True if done.
        """
        super(GymTerminalCondition, self).__init__(btype='neutral')  # None because we don't know
        self._value = done

    def check(self):
        """Check if the terminating condition has been fulfilled, and return True or False accordingly"""
        return self._value


class FixedTerminalCondition(TerminalCondition):
    r"""Fixed terminal condition

    Dummy fixed terminal condition.
    """

    def __init__(self, value=False, btype=None, name=None):
        """
        Initialize the dummy fixed terminal condition.

        Args:
            value (bool): if the terminal condition is over or not. This can be set by the user.
            btype (bool, str, None): if the terminal condition represents a failure or success condition. If None, it
                represents a neutral terminal condition (which is neither a failure or success condition, but just
                means the episode is over). If string, it has to be among {"success", "failure", "neutral"}.
            name (str): name of the final condition
        """
        super(FixedTerminalCondition, self).__init__(btype=btype, name=name)
        self._over = value

    @property
    def value(self):
        return self._over

    @value.setter
    def value(self, value):
        self._over = bool(value)


class FunctionalTerminalCondition(TerminalCondition):
    r"""Functional terminal condition

    Terminal condition that calls the given function every time we check if the task was carried out or not.
    """

    def __init__(self, fct, btype=None, name=None):
        """
        Initialize the functional terminal condition.

        Args:
            fct (callable): function to call each time we check if the task was carried out or not.
            btype (bool, str, None): if the terminal condition represents a failure or success condition. If None, it
                represents a neutral terminal condition (which is neither a failure or success condition, but just
                means the episode is over). If string, it has to be among {"success", "failure", "neutral"}.
            name (str): name of the final condition
        """
        if not callable(fct):
            raise TypeError("Expecting the provided 'fct' to be callable, but received instead: {}".format(fct))
        self._fct = fct

    def check(self):
        """
        Check if the terminating condition has been fulfilled, and return True or False accordingly
        """
        answer = self._fct()
        if isinstance(answer, tuple) and len(answer) == 2:
            self._over, self.btype = answer
        return self._over


# Tests
if __name__ == '__main__':

    condition = TimeLimitCondition(num_steps=2, btype='failure')

    for i in range(4):
        cond = condition()
        print("Iter: {}, type={}, over={}".format(i, condition.type_str(), condition.is_over()))
