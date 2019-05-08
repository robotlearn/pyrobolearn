#!/usr/bin/env python
"""Provide the Unmanned Aerial Vehicle (UAV) robot abstract classes.
"""

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class UAVRobot(Robot):
    r"""Unmanned Aerial Vehicle Robot

    Vehicles/Robots that operate in the air. These are also called drones.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        super(UAVRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

        self.propellers = []  # list of propellers id

    @property
    def num_propellers(self):
        """Return the number of propellers"""
        return len(self.propellers)


class FixedWingUAV(UAVRobot):
    r"""Fixed Wing Robot

    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        super(FixedWingUAV, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)


class RotaryWingUAV(UAVRobot):
    r"""Rotary Wing UAV

    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        super(RotaryWingUAV, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
