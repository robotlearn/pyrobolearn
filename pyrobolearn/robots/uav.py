#!/usr/bin/env python
"""Provide the Unmanned Aerial Vehicle (UAV) robot abstract classes.
"""

from pyrobolearn.robots.robot import Robot


class UAVRobot(Robot):
    r"""Unmanned Aerial Vehicle Robot

    Vehicles/Robots that operate in the air. These are also called drones.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(UAVRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.propellers = []  # list of propellers id

    @property
    def num_propellers(self):
        """Return the number of propellers"""
        return len(self.propellers)


class FixedWingUAV(UAVRobot):
    r"""Fixed Wing Robot

    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(FixedWingUAV, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)


class RotaryWingUAV(UAVRobot):
    r"""Rotary Wing UAV

    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(RotaryWingUAV, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
