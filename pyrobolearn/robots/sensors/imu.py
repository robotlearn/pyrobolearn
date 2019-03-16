#!/usr/bin/env python
"""Define the IMU sensor used in robotics.

'An inertial measurement unit (IMU) is an electronic device that measures and reports a body's specific force,
angular rate, and sometimes the magnetic field surroundings the body, using a combination of accelerometers
and gyroscopes, sometimes also magnetometers' (Wikipedia)
"""

from links import LinkSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class IMUSensor(LinkSensor):
    r"""Inertial Measurement Unit (IMU) sensor.

    The IMU sensor measures the linear and angular motion using accelerometers and gyroscopes. Specifically,
    3-axis accelerometers allow to measure the accelerations along its axes, and 3-axis gyroscopes measure the
    angular velocities around its axes. Sometimes, theses sensors also have a 3-axis magnetometer which measures
    the magnetic field.

    Note that real IMUs typically accumulate errors over time, and thus the integrated values drift over time.
    """

    def __init__(self, simulator, body_id, link_id, position, orientation):
        super(IMUSensor, self).__init__(simulator, body_id, link_id, position, orientation)

    def _sense(self):
        pass
