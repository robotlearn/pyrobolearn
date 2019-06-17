#!/usr/bin/env python
"""Define the various actuator actions
"""

from abc import ABCMeta
import collections
import numpy as np

from pyrobolearn.actions.action import Action
from pyrobolearn.robots.actuators.actuator import Actuator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ActuatorAction(Action):
    r"""Actuator action (abstract class)
    """
    __metaclass__ = ABCMeta

    def __init__(self, actuators, ticks=1):
        """
        Initialize the sensor state.

        Args:
            actuators (A, list of Actuator): actuator(s).
            ticks (int): number of ticks to sleep before setting the next action data.
        """
        if not isinstance(actuators, collections.Iterable):
            actuators = [actuators]
        for actuator in actuators:
            if not isinstance(actuator, Actuator):
                raise TypeError("Expecting the given 'actuator' to be an instance of `Actuator`, instead got: "
                                "{}".format(type(actuator)))
        self.actuators = actuators
        super(ActuatorAction, self).__init__(ticks=ticks)

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(actuators=self.actuators, ticks=self.ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        actuators = copy.deepcopy(self.actuators, memo)
        action = self.__class__(actuators=actuators, ticks=self.ticks)

        memo[self] = action
        return action
