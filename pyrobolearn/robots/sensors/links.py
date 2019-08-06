#!/usr/bin/env python
"""Define the link sensors used in robotics.

These include IMU, contact, Camera, and other sensors.
"""

import copy
import time
import numpy as np
from abc import ABCMeta

from pyrobolearn.utils.transformation import get_quaternion_product, get_rotated_point_from_quaternion
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

    def __init__(self, simulator, body_id, link_id=-1, position=None, orientation=None, noise=None, ticks=1,
                 latency=None):
        """Initialize the link sensor.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique id of the body.
            link_id (int): unique id of the link.
            position (np.array[float[3]], None): local position of the sensor with respect to the given link. If None,
                it will be the zero vector.
            orientation (np.array[float[4]], None): local orientation of the sensor with respect to the given link
                (expressed as a quaternion [x,y,z,w]). If None, it will be the unit quaternion [0,0,0,1].
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
        """
        super(LinkSensor, self).__init__(simulator, body_id=body_id, position=position, orientation=orientation,
                                         noise=noise, ticks=ticks, latency=latency)
        # set link id
        if not isinstance(link_id, int):
            raise TypeError("Expecting the given 'link_id' to be an int, instead got: {}".format(type(link_id)))
        self.link_id = link_id

        # link state
        self._state = {}
        self._prev_state = {}

    ##############
    # Properties #
    ##############

    @property
    def position(self):
        """
        Return the link position in the Cartesian world frame.
        """
        position = self.sim.get_link_state(self.body_id, self.link_id)[0]  # world position of CoM (o^w_1)
        orientation = self.sim.get_link_state(self.body_id, self.link_id)[1]  # world orientation of CoM (R^w_1)
        # p^w = o^w_1 + R^w_1 p^1
        # TODO: check if we are giving p^1 or p^2  --> p^w = o^w_1 + R^w_1 R^1_2 p^2 ??
        #  The local position that we are giving, is it in the new orientated frame or the local frame?
        position += get_rotated_point_from_quaternion(orientation, self.local_position)
        return position

    @property
    def orientation(self):
        """
        Return the link orientation in the Cartesian world frame.
        """
        orientation = self.sim.get_link_state(self.body_id, self.link_id)[1]  # world orientation of CoM (R^w_1)
        # R^w_2 = R^w_1 R^1_2
        orientation = get_quaternion_product(orientation, self.local_orientation)
        return orientation

    ###########
    # Methods #
    ###########

    def clean(self):
        """Clean sensor values."""
        # update previous and current states
        self._prev_state = self._state
        self._state = {}

    def get_link_world_position(self):
        r"""
        Return the CoM position (in the Cartesian world space coordinates) of the link associated with the sensor.

        Returns:
            np.array[float[3]]: the link CoM position in the world space
        """
        # check if cached
        if 'pos' in self._state:
            pos = self._state['pos'][0]  # (3,)
        else:
            if self.link_id == -1:  # base link
                pos = self.sim.get_base_position(self.body_id)
            else:  # other link
                pos = self.sim.get_link_world_positions(body_id=self.body_id, link_ids=self.link_id)  # (3,)
            self._state['pos'] = [pos, time.time()]

        # return position
        return pos

    def get_link_world_velocity(self):
        r"""
        Return the linear and angular velocities (expressed in the Cartesian world space coordinates) of the link
        associated with the sensor.

        Returns:
            np.array[6]: linear and angular velocity of the link in the Cartesian world space
        """
        # check if cached
        if 'vel' in self._state:
            vel = self._state['vel'][0]  # (6,)
        else:
            if self.link_id == -1:  # base link
                vel = np.concatenate(self.sim.get_base_velocity(self.body_id))
            else:  # other link
                vel = self.sim.get_link_world_velocities(body_id=self.body_id, link_ids=self.link_id)
            self._state['vel'] = [vel, time.time()]

        # return velocity
        return vel

    def get_link_world_acceleration(self):
        r"""
        Return the linear and angular accelerations (expressed in the Cartesian world space coordinates) of the link
        associated with the sensor.

        Returns:
            np.array[6]: linear and angular acceleration of the link in the Cartesian world space
        """
        # check if cached
        if 'acc' in self._state:
            acc = self._state['acc'][0]
        else:

            # if the simulator keep in memory the accelerations, return it
            if self.sim.supports_acceleration():
                acc = self.sim.get_link_world_accelerations(self.body_id, link_ids=self.link_id)  # (6,)
            else:  # else, use finite difference

                # get current link world velocities and time
                if 'vel' not in self._state:
                    self.get_link_world_velocity()
                vel, t = self._state['vel']  # (6,)

                # if we did not cache the previous base velocity
                if 'vel' not in self._prev_state:
                    acc = np.zeros(6)  # (6,)
                else:
                    # retrieve previous link world velocities and time
                    vel_prev, t_prev = self._prev_state['vel']  # (6,)

                    # compute time difference
                    if self.sim.use_real_time():  # if the simulator is in real-time mode
                        dt = (t - t_prev)
                    else:  # if we are stepping in the simulator
                        dt = self.sim.timestep

                    # get current link positions
                    pos = self.get_link_world_position()  # (3,)

                    # separate linear and angular velocities
                    lin_vel, ang_vel = vel[:3], vel[3:]  # (3,)
                    lin_vel_prev, ang_vel_prev = vel_prev[:3], vel_prev[3:]  # (3,)

                    # compute base acceleration
                    ang_acc = (ang_vel - ang_vel_prev) / dt
                    lin_acc = (lin_vel - lin_vel_prev) / dt
                    lin_acc += np.cross(ang_acc, pos) + np.cross(ang_vel, np.cross(ang_vel, pos))
                    acc = np.concatenate((lin_acc, ang_acc))  # (6,)

            # cache the acceleration
            self._state['acc'] = [acc, time.time()]

        return acc

    #############
    # Operators #
    #############

    def __copy__(self):
        """Return a shallow copy of the sensor. This can be overridden in the child class."""
        return self.__class__(simulator=self.simulator, body_id=self.body_id, link_id=self.link_id,
                              position=self.local_position, orientation=self.local_orientation, rate=self._ticks)

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
                                orientation=orientation, rate=self._ticks)
        memo[self] = sensor
        return sensor
