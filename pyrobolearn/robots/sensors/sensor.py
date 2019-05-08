#!/usr/bin/env python
"""Define the various sensors used in robotics.

This is decoupled from the robots such that sensors can be defined outside the robot class and can be selected at
run-time. This is useful for instance when a version of the robot has specific sensors while another version has
other sensors. Additionally, this is important as more realistic sensors can result in a better transfer from
simulation to reality. Also, note that some simulators are deterministic and thus the sensor class allows you to
add some noise to the returned sense value. The type of noise can also be selected at runtime.
"""

import copy
from abc import ABCMeta, abstractmethod
import numpy as np

from pyrobolearn.utils.transformation import get_quaternion_product

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Sensor(object):  # sensor attached to a link or joint
    r"""Sensor (abstract class)

    All sensors inherit from this class. Note that any sensors is attached to a body, link, or joint. Because these
    elements can move, the position and orientation of the sensor attached to it can change as well.
    Currently, sensors are massless (i.e. mass=0). Thus be sure that the link to which the sensor is attached, has
    the correct sensor mass / inertia.
    """
    __metaclass__ = ABCMeta

    def __init__(self, simulator, body_id, position=None, orientation=None, rate=1):
        """Initialize the sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique id of the body
            position (vec3): local position of the sensor with respect to the given link
            orientation (vec4): local orientation of the sensor with respect to the given link
            rate (int): number of steps to wait before acquisition of the next sensor value.
        """
        self.sim = simulator
        self.body_id = body_id

        if position is None:
            position = [0., 0., 0.]
        self.local_position = np.array(position)

        if orientation is None:
            orientation = [0., 0., 0., 1.]
        self.local_orientation = np.array(orientation)

        self.rate = rate
        self.cnt = -1

        # data from last acquisition
        self.data = None

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self.sim

    @property
    def position(self):
        """
        Return the body's CoM position in the Cartesian world frame.
        """
        position = self.sim.get_base_position(self.body_id)
        position += self.local_position
        return position

    @property
    def orientation(self):
        """
        Return the body's CoM orientation in the Cartesian world frame.
        """
        orientation = self.sim.get_base_orientation(self.body_id)
        orientation = get_quaternion_product(self.local_orientation, orientation)
        return orientation

    @abstractmethod
    def _sense(self):
        """Sense method to be implemented in the child class."""
        raise NotImplementedError

    def sense(self):
        """Get the next sensor value."""
        self.cnt += 1
        if self.cnt % self.rate == 0:
            self.data = self._sense()
            self.cnt = 0
            return self.data
        return self.data

    #############
    # Operators #
    #############

    # alias
    def __call__(self):
        """Get the next sensor value."""
        return self.sense()

    # def __repr__(self):
    #     """Return a representation string about the class for debugging and development."""
    #     return self.__class__.__name__

    def __str__(self):
        """Return a readable string about the class."""
        return self.__class__.__name__

    def __copy__(self):
        """Return a shallow copy of the sensor. This can be overridden in the child class."""
        return self.__class__(simulator=self.simulator, body_id=self.body_id, position=self.local_position,
                              orientation=self.local_orientation, rate=self.rate)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the sensor. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        simulator = copy.deepcopy(self.simulator, memo)
        body_id = copy.deepcopy(self.body_id)
        position = copy.deepcopy(self.local_position)
        orientation = copy.deepcopy(self.local_orientation)
        sensor = self.__class__(simulator=simulator, body_id=body_id, position=position, orientation=orientation,
                                rate=self.rate)
        memo[self] = sensor
        return sensor
