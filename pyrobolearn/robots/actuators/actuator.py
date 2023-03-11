#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the various actuators used in robotics.

This is decoupled from the robots such that actuators can be defined outside the robot class and can be selected at
run-time. This is useful for instance when a version of the robot has specific joint motors while another version has
other joint actuators. Additionally, this is important as more realistic motors can result in a better transfer from
simulation to reality.
"""

from abc import ABCMeta

from pyrobolearn.utils.data_structures.queues import FIFOQueue

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Actuator(object):
    r"""Actuator class

    All actuator classes inherit from this class. Actuators such as motors are often attached to the robot joints.
    Other actuators such as speakers, leds, and others are attached to links.
    """
    __metaclass__ = ABCMeta

    def __init__(self, ticks=1, latency=0):
        """
        Initialize the actuator.

        Args:
            ticks (int): number of steps to wait/sleep before acting in the world.
            latency (int, float, None): latency time / step.
        """

        # variable to check if the actuator is enabled
        self._enabled = True

        # set the ticks
        self._ticks = ticks
        self._cnt = -1

        # set the latency
        if latency is None:
            latency = 0
        if not isinstance(latency, (int, float)):
            raise TypeError("Expecting the given 'latency' to be an int or float, instead got: "
                            "{}".format(type(latency)))
        if latency < 0:
            raise ValueError("Expecting the given 'latency' to be a positive number, but got instead: "
                             "{}".format(latency))
        self._latency = latency
        self._data_queue = FIFOQueue(maxsize=self._latency + 1)  # latency is modeled using a queue

        # self.sim = simulator
        self._data = None

    ##############
    # Properties #
    ##############

    # @property
    # def simulator(self):
    #     return self.sim
    #
    # @simulator.setter
    # def simulator(self, simulator):
    #     self.sim = simulator

    @property
    def enabled(self):
        """Return if the actuator is enabled or not."""
        return self._enabled

    @property
    def disabled(self):
        """Return if the actuator is disabled or not."""
        return not self._enabled

    @property
    def data(self):
        """Return the data."""
        return self._data

    @data.setter
    def data(self, data):
        """Set the data."""
        while len(self._data_queue) != self._data_queue.maxsize:  # fill the data queue
            self._data_queue.append(data)

    ###########
    # Methods #
    ###########

    def enable(self):
        """Enable the actuator."""
        self._enabled = True

    def disable(self):
        """Disable the actuator."""
        self._enabled = False

    def compute(self, *args, **kwargs):
        pass

    def act(self):
        """Set the next actuator value."""
        if self._enabled:
            self._cnt += 1
            if (self._cnt % self._ticks) == 0:
                self._data = self._data_queue.get()
                self._act()
                self._cnt = 0

    def _act(self):
        """Act method to be implemented in the child class."""
        raise NotImplementedError

    #############
    # Operators #
    #############

    def __call__(self, *args, **kwargs):
        """Set the next actuator value."""
        self.act()
        # return self.compute(*args, **kwargs)

    # def __repr__(self):
    #     """Return a representation string about the class for debugging and development."""
    #     return self.__class__.__name__

    def __str__(self):
        """Return a readable string about the class."""
        return self.__class__.__name__

    def __copy__(self):
        """Return a shallow copy of the actuator. This can be overridden in the child class."""
        return self.__class__()

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the actuator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        actuator = self.__class__()
        memo[self] = actuator
        return actuator
