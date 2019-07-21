#!/usr/bin/env python
"""Define the altimeter sensor which measures the altitude of an object.
"""

import numpy as np

from pyrobolearn.robots.sensors.links import LinkSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class AltimeterSensor(LinkSensor):
    r"""Altimeter sensor

    'An altimeter or an altitude meter is an instrument used to measure the altitude of an object above a fixed level.'
    [1] Here the fixed level is set to be the ground.

    References:
        - [1] Altimeter (Wikipedia): https://en.wikipedia.org/wiki/Altimeter
    """

    def __init__(self, simulator, body_id, link_id=-1, noise=None, ticks=1, latency=None, position=None,
                 orientation=None):
        """
        Initialize the altimeter sensor.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            link_id (int): unique id of the link.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
            position (np.array[3], None): local position of the sensor with respect to the given link. If None, it will
                be the zero vector.
            orientation (np.array[4], None): local orientation of the sensor with respect to the given link (expressed
                as a quaternion [x,y,z,w]). If None, it will be the unit quaternion [0,0,0,1].
        """
        super(AltimeterSensor, self).__init__(simulator, body_id=body_id, link_id=link_id, noise=noise, ticks=ticks,
                                              latency=latency, position=position, orientation=orientation)

    def _sense(self, apply_noise=True):
        """
        Sense the altitude.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            float: altitude (in z-direction).
        """
        # check if the simulator supports that sensor
        if self.sim.supports_sensors("altimeter"):
            return self.sim.get_sensor("altimeter", self.body_id, self.link_id).sense()

        z = self.get_link_world_position()[-1]  # get the z direction
        if apply_noise:
            z = self._noise(z)

        return z
