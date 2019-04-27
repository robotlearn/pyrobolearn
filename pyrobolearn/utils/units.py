#!/usr/bin/env python
"""Convert units from one system to another one.
"""

import numpy as np

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def inches_to_meters(inch):
    """
    Convert the inches to meters.

    Args:
        inch (float): inches.

    Returns:
        float: meters
    """
    return inch * 0.0254


def meters_to_inches(meter):
    """
    Convert the meters to inches.

    Args:
        meter (float): meters.

    Returns:
        float: inches
    """
    return meter / 0.0254


def rpm_to_rad_per_second(rpm):
    """
    Convert the revolutions per minute to rad/sec.

    Args:
        rpm (float): revolutions per minute.

    Returns:
        float: rad/sec
    """
    return rpm * 2 * np.pi/60


def rad_per_second_to_rpm(omega):
    """
    Convert rad/sec to revolutions/minute.

    Args:
        omega (float): angular velocity (rad/sec)

    Returns:
        float: revolutions/minute
    """
    return omega * 60 / (2*np.pi)
