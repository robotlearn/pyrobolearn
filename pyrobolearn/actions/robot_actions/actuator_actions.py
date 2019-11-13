#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the various actuator actions
"""

from abc import ABCMeta
import collections
import numpy as np
import copy

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

    def __init__(self, actuator, ticks=1):
        """
        Initialize the sensor state.

        Args:
            actuator (Actuator): actuator instance.
            ticks (int): number of ticks to sleep before setting the next action data.
        """
        super(ActuatorAction, self).__init__(ticks=ticks)

        # set the actuator instance
        self.actuator = actuator

    ##############
    # Properties #
    ##############

    @property
    def actuator(self):
        """Return the actuator instance."""
        return self._actuator

    @actuator.setter
    def actuator(self, actuator):
        """Set the actuator instance."""
        if not isinstance(actuator, Actuator):
            raise TypeError("Expecting the given 'actuator' to be an instance of `Actuator`, instead got: "
                            "{}".format(type(actuator)))
        self._actuator = actuator

    ###########
    # Methods #
    ###########

    def _write(self, data):
        """Write the data in the actuator and execute the actuator."""
        # set the data in the actuator
        self.actuator.data = data

        # activate the actuator
        self.actuator.act()

    #############
    # Operators #
    #############

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(actuator=self.actuator, ticks=self.ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        actuators = copy.deepcopy(self.actuator, memo)
        action = self.__class__(actuators=actuators, ticks=self.ticks)

        memo[self] = action
        return action
