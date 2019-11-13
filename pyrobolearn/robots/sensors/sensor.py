#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the various sensors used in robotics.

This is decoupled from the robots such that sensors can be defined outside the robot class and can be selected at
run-time. This is useful for instance when a version of the robot has specific sensors while another version has
other sensors. Additionally, this is important as more realistic sensors can result in a better transfer from
simulation to reality. Also, note that some simulators are deterministic and thus the sensor class allows you to
add some noise to the returned sense value. The type of noise can also be selected at runtime.
"""

# TODO: convert ticks to rate when setting real-time

import sys
import copy
from abc import ABCMeta, abstractmethod
import numpy as np

from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.robots.base import Body
from pyrobolearn.robots.noise.noise import Noise, NoNoise
from pyrobolearn.utils.transformation import get_quaternion_product


# define long for Python 3.x
if int(sys.version[0]) == 3:
    long = int


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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

    def __init__(self, simulator, body_id, noise=None, ticks=1, latency=None, position=None, orientation=None):
        """
        Initialize the sensor.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int, Body): unique id of the body, or body instance.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
            position (np.array[float[3]], None): local position of the sensor with respect to the given link. If None,
                it will be the zero vector.
            orientation (np.array[float[4]], None): local orientation of the sensor with respect to the given link
                (expressed as a quaternion [x,y,z,w]). If None, it will be the unit quaternion [0,0,0,1].
        """
        # setting simulator
        if not isinstance(simulator, Simulator):
            raise TypeError("Expecting the given 'simulator' to be an instance of `Simulator`, but got instead: "
                            "{}".format(type(simulator)))
        self.sim = simulator

        # set the body id
        if isinstance(body_id, Body):
            body_id = body_id.id
        elif not isinstance(body_id, (int, long)):
            raise TypeError("Expecting the given 'body_id' to be an int or an instance of `Body`, but got instead: "
                            "{}".format(type(body_id)))
        if body_id < 0:
            raise ValueError("Expecting the given 'body_id' to be a positive integer, but got instead: "
                             "{}".format(body_id))
        self.body_id = body_id

        # set the local position of the sensor
        if position is None:
            position = [0., 0., 0.]
        self.local_position = np.asarray(position)

        # set the local orientation of the sensor
        if orientation is None:
            orientation = [0., 0., 0., 1.]
        self.local_orientation = np.asarray(orientation)

        # set the ticks / rate
        if ticks < 1:
            raise ValueError("Expecting the given 'ticks' to be a positive number, but got instead: {}".format(ticks))
        self._ticks = ticks
        self._cnt = -1

        # set the noise
        if noise is None:
            noise = NoNoise()
        if not isinstance(noise, Noise):
            raise TypeError("Expecting the given 'noise' to be an instance of Noise, instead got: "
                            "{}".format(type(noise)))
        self._noise = noise

        # variable to check if the sensor is enabled
        self._enabled = True

        # set the latency
        if latency is None:
            latency = 0
        if not isinstance(latency, (int, float)):
            raise TypeError("Expecting the given 'latency' to be an int or float, instead got: "
                            "{}".format(type(latency)))
        if latency < 0:
            raise ValueError("Expecting the given 'latency' to be a positive number, but got instead: "
                             "{}".format(latency))
        self._latency = latency + 1
        self._latent_cnt = -1

        # data from last acquisition
        self._data = None
        self._latent_data = None  # self._sense()

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self.sim

    # @property
    # def position(self):
    #     """Return the body's base position in the Cartesian world frame."""
    #     position = self.sim.get_base_position(self.body_id)
    #     position += self.local_position
    #     return position
    #
    # @property
    # def orientation(self):
    #     """Return the body's base orientation in the Cartesian world frame."""
    #     orientation = self.sim.get_base_orientation(self.body_id)
    #     orientation = get_quaternion_product(orientation, self.local_orientation)
    #     return orientation

    @property
    def enabled(self):
        """Return if the sensor is enabled or not."""
        return self._enabled

    @property
    def disabled(self):
        """Return if the sensor is disabled or not."""
        return not self._enabled

    @property
    def data(self):
        """Return the data."""
        return self._data

    @property
    def latent_data(self):
        """Return the latent data."""
        return self._latent_data

    ###########
    # Methods #
    ###########

    def reset(self):
        """Reset sensor."""
        self._latent_data = self._sense()

    def clean(self):
        """clean sensor values."""
        pass

    def enable(self):
        """Enable the sensor."""
        self._enabled = True

    def disable(self):
        """Disable the sensor."""
        self._enabled = False

    @abstractmethod
    def _sense(self, apply_noise=True):
        """Sense method to be implemented in the child class. This has to apply the noise if you wish to have
        noisy data.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            np.array: sensed data.
        """
        raise NotImplementedError

    def sense(self, apply_noise=True):
        """Get the next sensor value.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.
        """
        if self._enabled:
            self._cnt += 1
            if (self._cnt % self._ticks) == 0:  # if time to update
                if self._latency == 0:  # if no latency
                    self._data = self._sense()
                    self._latent_data = self._data
                else:  # if latency
                    self._latent_cnt += 1
                    if (self._latent_cnt % self._latency) == 0:  # TODO: use FIFO queue instead (see actuator.py)?
                        self._data = self._latent_data
                        self._latent_data = self._sense(apply_noise=apply_noise)
                        self._latent_cnt = 0
                self._cnt = 0
        return self._data

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
                              orientation=self.local_orientation, noise=self._noise, ticks=self._ticks,
                              latency=self._latency)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the sensor. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        simulator = memo.get(self.simulator, self.simulator)  # copy.deepcopy(self.simulator, memo)
        body_id = copy.deepcopy(self.body_id)
        position = copy.deepcopy(self.local_position)
        orientation = copy.deepcopy(self.local_orientation)
        noise = copy.deepcopy(self._noise)
        sensor = self.__class__(simulator=simulator, body_id=body_id, position=position, orientation=orientation,
                                noise=noise, ticks=self._ticks, latency=self._latency)
        memo[self] = sensor
        return sensor
