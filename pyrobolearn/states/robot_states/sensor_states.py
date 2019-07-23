#!/usr/bin/env python
"""Define the various sensor states

This includes notably the camera, contact, IMU, force/torque sensors and others.
"""

import copy
from abc import ABCMeta
import collections
import numpy as np

# from pyrobolearn.states.robot_states.robot_states import RobotState
from pyrobolearn.states.state import State
from pyrobolearn.robots.legged_robot import LeggedRobot
from pyrobolearn.robots.sensors.sensor import Sensor
from pyrobolearn.robots.sensors.contact import ContactSensor
from pyrobolearn.robots.sensors.camera import CameraSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SensorState(State):
    r"""Sensor state (abstract class)
    """
    __metaclass__ = ABCMeta

    def __init__(self, sensor, window_size=1, axis=None, ticks=1, update=False):
        """
        Initialize the sensor state.

        Args:
            sensor (Sensor): sensor instance.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
            update (bool): if we should update the sensor, or not. Note that this is normally carried out by the
                `robot.step` method (which is itself called by `world.step`), so normally you shouldn't set it to True.
        """
        super(SensorState, self).__init__(window_size=window_size, axis=axis, ticks=ticks)

        # set the sensor instance
        self.sensor = sensor

        self._update = bool(update)

    ##############
    # Properties #
    ##############

    @property
    def sensor(self):
        """Return the sensor instance."""
        return self._sensor

    @sensor.setter
    def sensor(self, sensor):
        """Set the sensor instance."""
        if not isinstance(sensor, Sensor):
            raise TypeError("Expecting the given 'sensor' to be an instance of `Sensor`, instead got: "
                            "{}".format(type(sensor)))
        self._sensor = sensor

    ###########
    # Methods #
    ###########

    def _read(self):
        """Read the sensor values."""
        # update the sensor if specified (normally we don't need to do it as it is carried out by robot.step, or
        # world.step)
        if self._update:
            self.sensor.sense(apply_noise=True)

        # get the data from the sensor
        self.data = self.sensor.data

    #############
    # Operators #
    #############

    def __copy__(self):
        """Return a shallow copy of the state. This can be overridden in the child class."""
        return self.__class__(sensor=self.sensor, window_size=self.window_size, axis=self.axis, ticks=self.ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the state. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        sensor = copy.deepcopy(self.sensor, memo)
        state = self.__class__(sensor=sensor, window_size=self.window_size, axis=self.axis, ticks=self.ticks)

        memo[self] = state
        return state


class CameraState(SensorState):
    r"""Camera state
    """

    def __init__(self, camera, window_size=1, axis=None, ticks=1):
        """
        Initialize the camera sensor state.

        Args:
            camera (CameraSensor): camera sensor(s).
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        self.camera = camera
        super(CameraState, self).__init__(sensor=camera, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        pass


class ContactState(SensorState):
    r"""Contact state

    Return the contact states between a link of the robot and an object in the world (including the floor).
    """

    def __init__(self, contacts, window_size=1, axis=None, ticks=1):
        """Initialize the contact state.

        Args:
            contacts (ContactSensor, list of ContactSensor): list of contact sensor(s).
                If None, it will check if the robot has some contact sensors. If there are no contact sensors, it
                will check the contact with all the links.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        if not isinstance(contacts, collections.Iterable):
            contacts = [contacts]
        for contact in contacts:
            if not isinstance(contact, ContactSensor):
                raise TypeError("Expecting the given 'contact' to be an instance of `ContactSensor`, instead got: "
                                "{}".format(type(contact)))
        self.contacts = contacts
        super(ContactState, self).__init__(sensor=contacts, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        contacts = np.array([int(contact.is_in_contact()) for contact in self.contacts])
        self.data = contacts


class FeetContactState(ContactState):
    r"""Feet Contact State

    Return the contact states between the foot of the robot and an object in the world (including the floor).
    """

    def __init__(self, robot, contacts=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the feet contact state.

        Args:
            robot (LeggedRobot): legged robot.
            contacts (ContactSensor, list of ContactSensor, None): list of contact sensors.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        # check if the robot has feet
        if not isinstance(robot, LeggedRobot):
            raise TypeError("Expecting the robot to be an instance of `LeggedRobot`, instead got: "
                            "{}".format(type(robot)))
        if len(robot.feet) == 0:
            raise ValueError("The given robot has no feet; please set the `feet` attribute in the robot.")
        self.robot = robot

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
        super(FeetContactState, self).__init__(contacts, window_size=window_size, axis=axis, ticks=ticks)
