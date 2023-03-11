#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the various Linear Inverted Pendulum Models (LIPMs).

This includes: LIPM2D, DualLIPM2D, LIPM3D, DualLIPM3D
"""

# TODO: finish to implement this

import os

from pyrobolearn.robots.legged_robot import LeggedRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LIPM2D(LeggedRobot):
    r"""Linear Inverted Pendulum Model 2D

    This class describes the 2D Linear Inverted Pendulum Model (2D-LIPM), which underlies walking behaviors, and is
    often used as a template model in locomotion. The 2D version constraints the possible motion to belong to the xz
    plan.

    See Also:
        - LIPM3D: the 3D version of the 2D-LIPM
        - DualLIPM2D: the dual version of the 2D-LIPM
        - DualLIPM3D: the dual version of the 3D-LIPM
    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/templates/lipm2d.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = False

        super(LIPM2D, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'lipm_2d'


class DualLIPM2D(LeggedRobot):
    r"""Dual Linear Inverted Pendulum Model 2D

    This class describes the dual 2D Linear Inverted Pendulum Model (dual 2D-LIPM), which underlies walking behaviors,
    and is often used as a template model in locomotion. The 2D version constraints the possible motions to belong to
    the xz plan.

    See Also:
        - LIPM3D: the 3D version of the 2D-LIPM
        - DualLIPM3D: the dual version of the 3D-LIPM
    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/templates/dual_lipm2d.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = False

        super(DualLIPM2D, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'dual_lipm_2d'


class LIPM3D(LeggedRobot):
    r"""Linear Inverted Pendulum Model 3D

    This class describes the 3D Linear Inverted Pendulum Model (3D-LIPM), which underlies walking behaviors, and is
    often used as a template model in locomotion.

    See Also:
        - DualLIPM3D: the dual version of the 3D-LIPM
    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/templates/lipm3d.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = False

        super(LIPM3D, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'lipm_3d'


class DualLIPM3D(LeggedRobot):
    r"""Dual Linear Inverted Pendulum Model 3D

    This class describes the dual 3D Linear Inverted Pendulum Model (dual 3D-LIPM), which underlies walking behaviors,
    and is often used as a template model in locomotion.

    See Also:
        - SLIP: the spring-loaded inverted pendulum which is more suitable for running and jumping behaviors.
    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/templates/dual_lipm3d.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = False

        super(DualLIPM3D, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'dual_lipm_3d'


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = LIPM2D(sim)

    # print information about the robot
    robot.print_info()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
