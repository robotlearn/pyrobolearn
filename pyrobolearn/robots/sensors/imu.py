#!/usr/bin/env python
"""Define the IMU sensor used in robotics.

'An inertial measurement unit (IMU) is an electronic device that measures and reports a body's specific force,
angular rate, and sometimes the magnetic field surroundings the body, using a combination of accelerometers
and gyroscopes, sometimes also magnetometers' [1]

References:
    - [1] Inertial measurement unit: https://en.wikipedia.org/wiki/Inertial_measurement_unit
"""

import numpy as np

import pyrobolearn as prl
from pyrobolearn.robots.sensors.links import LinkSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class IMUSensor(LinkSensor):
    r"""Inertial Measurement Unit (IMU) sensor.

    The IMU sensor measures the linear and angular motion using accelerometers and gyroscopes. Specifically,
    3-axis accelerometers allow to measure the accelerations along its axes, and 3-axis gyroscopes measure the
    angular velocities around its axes. Sometimes, theses sensors also have a 3-axis magnetometer which measures
    the magnetic field. [1]

    Note that real IMUs typically accumulate errors over time, and thus the integrated values drift over time.

    A very wide variety of IMUs exists, depending on application types, with performance ranging [:

    - from 0.1 deg/s to 0.001 deg/h for gyroscope
    - from 100 mg to 10^{-6}g for accelerometers

    References:
        - [1] Inertial measurement unit: https://en.wikipedia.org/wiki/Inertial_measurement_unit
        - [2] "IMU, what for: performance per application infographic":
            https://www.thalesgroup.com/en/worldwide/aerospace/topaxyz-inertial-measurement-unit-imu/infographic
    """

    def __init__(self, simulator, body_id, link_id=-1, noise=None, ticks=1, latency=None, position=None,
                 orientation=None):
        """
        Initialize the IMU sensor.

        Args:
            simulator (Simulator): simulator instance.
            body_id (int): unique body id.
            link_id (int): unique id of the link.
            noise (None, Noise): noise to be added.
            ticks (int): number of steps to wait/sleep before acquisition of the next sensor value.
            latency (int, float, None): latency time / step.
            position (np.array[float[3]], None): local position of the sensor with respect to the given link. If None,
                it will be the zero vector.
            orientation (np.array[float[4]], None): local orientation of the sensor with respect to the given link
                (expressed as a quaternion [x,y,z,w]). If None, it will be the unit quaternion [0,0,0,1].
        """
        super(IMUSensor, self).__init__(simulator, body_id=body_id, link_id=link_id, noise=noise, ticks=ticks,
                                        latency=latency, position=position, orientation=orientation)

    def _sense(self, apply_noise=True):
        """Sense using the IMU sensor.

        Args:
            apply_noise (bool): if we should apply the noise or not. Note that the sensor might already have some noise.

        Returns:
            np.array[float[6]]: concatenation of linear accelerations and angular velocities
        """
        # if the simulator supports IMU sensors, return the sensed data
        if self.simulator.supports_sensors("imu"):
            return self.simulator.get_sensor("imu", self.body_id, self.link_id).sense()

        # linear acceleration
        acceleration = self.get_link_world_acceleration()[:3]

        # angular velocity
        angular_velocity = self.get_link_world_velocity()[3:]

        # concatenate the data and apply the noise
        data = np.concatenate((acceleration, angular_velocity))
        if apply_noise:
            data = self._noise(data)

        # return the noisy data
        return data
