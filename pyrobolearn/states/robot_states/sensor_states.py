#!/usr/bin/env python
"""Define the various sensor states

This includes notably the camera, contact, IMU, force/torque sensors and others.
"""

from abc import ABCMeta

from pyrobolearn.states.robot_states.robot_states import RobotState


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SensorState(RobotState):  # TODO: define refresh_rate & frequency
    r"""Sensor state (abstract class)
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot):
        super(SensorState, self).__init__(robot)


class CameraState(SensorState):
    r"""Camera state
    """

    def __init__(self, robot, camera=None):
        super(CameraState, self).__init__(robot)

    def _read(self):
        pass


class ContactState(SensorState):
    r"""Contact state

    Return the contact states between a link of the robot and an object in the world (including the floor).
    """

    def __init__(self, robot, contacts=None):
        super(ContactState, self).__init__(robot)

    def _read(self):
        pass


class FeetContactState(ContactState):
    r"""Feet Contact State

    Return the contact states between
    """

    def __init__(self, robot, contacts=None):
        super(FeetContactState, self).__init__(robot, contacts)

    def _read(self):
        pass
