#!/usr/bin/env python
"""Define the link sensors used in robotics.

These include IMU, contact, Camera, and other sensors.
"""

import copy
from abc import ABCMeta, abstractmethod

from pyrobolearn.utils.transformation import get_quaternion_product
from pyrobolearn.robots.sensors.sensor import Sensor


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinkSensor(Sensor):
    r"""Link Sensor (abstract)

    Sensor attached to a link.
    """

    __metaclass__ = ABCMeta

    def __init__(self, simulator, body_id, link_id=None, position=None, orientation=None, rate=1):
        """Initialize the sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique id of the body
            link_id (int): unique id of the link
            position (vec3): local position of the sensor with respect to the given link
            orientation (vec4): local orientation of the sensor with respect to the given link
            rate (int): number of steps to wait before acquisition of the next sensor value.
        """
        super(LinkSensor, self).__init__(simulator, body_id, position, orientation, rate)
        self.link_id = link_id

    @property
    def position(self):
        """
        Return the link position in the Cartesian world frame.
        """
        position = self.sim.get_link_state(self.body_id, self.link_id)[0]
        position += self.local_position
        return position

    @property
    def orientation(self):
        """
        Return the link orientation in the Cartesian world frame.
        """
        orientation = self.sim.get_link_state(self.body_id, self.link_id)[1]
        orientation = get_quaternion_product(self.local_orientation, orientation)
        return orientation

    @abstractmethod
    def _sense(self):
        raise NotImplementedError

    def __copy__(self):
        """Return a shallow copy of the sensor. This can be overridden in the child class."""
        return self.__class__(simulator=self.simulator, body_id=self.body_id, link_id=self.link_id,
                              position=self.local_position, orientation=self.local_orientation, rate=self.rate)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the sensor. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        simulator = memo.get(self.simulator, self.simulator)  # copy.deepcopy(self.simulator, memo)
        body_id = copy.deepcopy(self.body_id)
        link_id = copy.deepcopy(self.link_id)
        position = copy.deepcopy(self.local_position)
        orientation = copy.deepcopy(self.local_orientation)
        sensor = self.__class__(simulator=simulator, body_id=body_id, link_id=link_id, position=position,
                                orientation=orientation, rate=self.rate)
        memo[self] = sensor
        return sensor
