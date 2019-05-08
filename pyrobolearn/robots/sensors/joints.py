#!/usr/bin/env python
"""Define the joint sensors used in robotics.

This mainly include encoders.
"""

import copy
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


class JointSensor(Sensor):
    r"""Joint Sensor (abstract)

    Sensor attached to a joint.
    """
    __metaclass__ = ABCMeta

    def __init__(self, simulator, body_id, joint_id, position=None, orientation=None, rate=1):
        """Initialize the sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique id of the body
            joint_id (int): unique id of the joint
            position (vec3): local position of the sensor with respect to the given joint
            orientation (vec4): local orientation of the sensor with respect to the given joint
            rate (int): number of steps to wait before acquisition of the next sensor value.
        """
        super(JointSensor, self).__init__(simulator, body_id, position, orientation, rate)
        self.joint_id = joint_id

    @property
    def position(self):
        """
        Return the joint position
        """
        return self.sim.get_joint_state(self.body_id, self.joint_id)[0]

    @abstractmethod
    def _sense(self):
        raise NotImplementedError

    def __copy__(self):
        """Return a shallow copy of the sensor. This can be overridden in the child class."""
        return self.__class__(simulator=self.simulator, body_id=self.body_id, joint_id=self.joint_id,
                              position=self.local_position, orientation=self.local_orientation, rate=self.rate)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the sensor. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        simulator = copy.deepcopy(self.simulator, memo)
        body_id = copy.deepcopy(self.body_id)
        joint_id = copy.deepcopy(self.joint_id)
        position = copy.deepcopy(self.local_position)
        orientation = copy.deepcopy(self.local_orientation)
        sensor = self.__class__(simulator=simulator, body_id=body_id, joint_id=joint_id, position=position,
                                orientation=orientation, rate=self.rate)
        memo[self] = sensor
        return sensor



class Encoder(JointSensor):
    r"""Encoder joint sensor

    The encoder is a sensor that measures rotation allowing to determine the angle, displacement, velocity, or
    acceleration.
    """

    def __init__(self, simulator, body_id, joint_id, noise=None):
        super(Encoder, self).__init__(simulator, body_id, joint_id)
