#!/usr/bin/env python
"""Define the link sensors used in robotics.

These include IMU, contact, Camera, and other sensors.
"""

from abc import ABCMeta, abstractmethod

from pyrobolearn.robots.sensors.sensor import Sensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinkSensor(Sensor):
    r"""Link Sensor (abstract)

    Sensor attached to a link.
    """

    __metaclass__ = ABCMeta

    def __init__(self, simulator, body_id, link_id=None, position=None, orientation=None, refresh_rate=1):
        """Initialize the sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique id of the body
            link_id (int): unique id of the link
            position (vec3): local position of the sensor with respect to the given link
            orientation (vec4): local orientation of the sensor with respect to the given link
            refresh_rate (int): number of steps to wait before acquisition of the next sensor value.
        """
        super(LinkSensor, self).__init__(simulator, body_id, position, orientation, refresh_rate)
        self.link_id = link_id

    @property
    def position(self):
        """
        Return the link position in the Cartesian world frame.
        """
        position = self.pos_converter(self.sim.getLinkState(self.body_id, self.link_id)[0])
        position += self.local_position
        return position

    @property
    def orientation(self):
        """
        Return the link orientation in the Cartesian world frame.
        """
        orientation = self.orientation_converter(self.sim.getLinkState(self.body_id, self.link_id)[1])
        orientation = self.local_orientation * orientation
        return orientation

    @abstractmethod
    def _sense(self):
        raise NotImplementedError
