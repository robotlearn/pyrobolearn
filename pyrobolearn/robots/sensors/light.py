#!/usr/bin/env python
"""Define light / laser sensors (except cameras).
"""

import numpy as np

from links import LinkSensor

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LightSensor(LinkSensor):
    r"""Light Sensor
    """
    pass


class LaserSensor(LinkSensor):
    r"""Laser Sensor

    """

class ProximityLaserSensor(LaserSensor):
    r"""Proximity Laser Sensor

    """
    pass


class IRProximitySensor(ProximityLaserSensor):
    r"""Infrared proximity sensor

    Compared to a proximity laser sensor, this sensor is influenced by the ambient light.
    """
    pass


class LidarSensor(LinkSensor):
    r"""Lidar Sensor
    """
    pass
