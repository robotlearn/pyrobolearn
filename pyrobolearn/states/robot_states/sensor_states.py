#!/usr/bin/env python
"""Define the various sensor states

This includes notably the camera, contact, IMU, force/torque sensors and others.
"""

from abc import ABCMeta
import collections
import numpy as np

from pyrobolearn.states.robot_states.robot_states import RobotState
from pyrobolearn.robots.legged_robot import LeggedRobot

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
        """Initialize the contact state.

        Args:
            robot (Robot): robot instance.
            contacts (int, list of int, ContactSensor, list of ContactSensor, None): link id(s) or contact sensor(s).
                If None, it will check if the robot has some contact sensors. If there are no contact sensors, it
                will check the contact with all the links.
        """
        super(ContactState, self).__init__(robot)
        self.contacts = contacts
        # read the data
        self._read()

    def _read(self):
        contacts = [self.robot.simulator.get_contact_points(body1=self.robot.id, link1_id=link_id)
                    for link_id in self.contacts]
        contacts = np.array([int(len(contact) > 0) for contact in contacts])
        self.data = contacts


class FeetContactState(ContactState):
    r"""Feet Contact State

    Return the contact states between the foot of the robot and an object in the world (including the floor).
    """

    def __init__(self, robot, contacts=None):
        # check if the robot has feet
        if not isinstance(robot, LeggedRobot):
            raise TypeError("Expecting the robot to be an instance of `LeggedRobot`, instead got: "
                            "{}".format(type(robot)))
        if len(robot.feet) == 0:
            raise ValueError("The given robot has no feet; please set the `feet` attribute in the robot.")

        # check if the contact sensors or link ids are valid
        if contacts is None:
            feet = robot.feet
            feet_ids = []
            for foot in feet:
                if isinstance(foot, int):
                    feet_ids.append(foot)
                elif isinstance(foot, collections.Iterable):
                    for f in foot:
                        feet_ids.append(f)
                else:
                    raise TypeError("Expecting the list of feet ids to be a list of integers.")
            contacts = feet_ids
        super(FeetContactState, self).__init__(robot, contacts)
