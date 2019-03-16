#!/usr/bin/env python
"""Define the joint sensors used in robotics.

This mainly include encoders.
"""

from abc import ABCMeta, abstractmethod

from sensor import Sensor

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

    def __init__(self, simulator, body_id, joint_id, position=None, orientation=None, refresh_rate=1):
        """Initialize the sensor.

        Args:
            simulator (Simulator): simulator
            body_id (int): unique id of the body
            joint_id (int): unique id of the joint
            position (vec3): local position of the sensor with respect to the given joint
            orientation (vec4): local orientation of the sensor with respect to the given joint
            refresh_rate (int): number of steps to wait before acquisition of the next sensor value.
        """
        super(JointSensor, self).__init__(simulator, body_id, position, orientation, refresh_rate)
        self.joint_id = joint_id

    @property
    def position(self):
        """
        Return the joint position
        """
        return self.sim.JointState(self.body_id, self.joint_id)[0]

    @abstractmethod
    def _sense(self):
        raise NotImplementedError


class Encoder(JointSensor):
    r"""Encoder joint sensor

    The encoder is a sensor that measures rotation allowing to determine the angle, displacement, velocity, or
    acceleration.
    """

    def __init__(self, simulator, body_id, joint_id, noise=None):
        super(Encoder, self).__init__(simulator, body_id, joint_id)
