#!/usr/bin/env python
"""Provide the Unmanned Underwater Vehicle (UUV) robot abstract classes.
"""

from robot import Robot


class UUVRobot(Robot):
    r"""Unmanned Underwater Vehicle Robot

    Vehicles/Robots that operate under water.
    """

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(UUVRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
