#!/usr/bin/env python
"""Provide the Unmanned Aerial Vehicle (UAV) robot abstract classes.
"""

from robot import Robot


class UAVRobot(Robot):
    r"""Unmanned Aerial Vehicle Robot

    Vehicles/Robots that operate in the air. These are also called drones.
    """

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(UAVRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)

        self.propellers = []  # list of propellers id

    @property
    def num_propellers(self):
        """Return the number of propellers"""
        return len(self.propellers)


class FixedWingUAV(UAVRobot):
    r"""Fixed Wing Robot

    """

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(FixedWingUAV, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)


class RotaryWingUAV(UAVRobot):
    r"""Rotary Wing UAV

    """

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(RotaryWingUAV, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
