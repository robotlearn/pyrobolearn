#!/usr/bin/env python
"""Provide the various Spring-Loaded Inverted Pendulum (SLIP) models.

This includes: SLIP2D, DualSLIP2D, SLIP3D, DualSLIP3D
"""

# TODO: finish to implement this

import os

from pyrobolearn.robots.legged_robot import LeggedRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SLIP2D(LeggedRobot):
    r"""Spring-Loaded Inverted Pendulum (2D)

    This class describes the 2D Spring-Loaded Inverted Pendulum (2D-SLIP) model, which underlies dynamic locomotive
    behaviors (such as running and jumping), and is often used as a template model in locomotion. The 2D version
    constraints the possible motion to belong to the xz plan.

    See Also:
        - SLIP3D: the 3D version of the 2D-SLIP
        - DualSLIP2D: the dual version of the 2D-SLIP
        - DualSLIP3D: the dual version of the 3D-SLIP
    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/templates/SLIP2d.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = False

        super(SLIP2D, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'SLIP_2d'

        # create mass sphere

        # create leg (capsule or cylinder with sphere at the end)

        # create spring (capsules or cylinders)

        # create prismatic constraint for spring

        # create planar constraint

        # simulate the force due to the spring (in step)


class DualSLIP2D(LeggedRobot):
    r"""Dual Linear Inverted Pendulum Model 2D

    This class describes the dual 2D Spring-Loaded Inverted Pendulum (dual 2D-SLIP) model, which underlies dynamic
    locomotive behaviors (such as running and jumping), and is often used as a template model in locomotion. The 2D
    version constraints the possible motion to belong to the xz plan.

    See Also:
        - SLIP3D: the 3D version of the 2D-SLIP
        - DualSLIP3D: the dual version of the 3D-SLIP
    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/templates/dual_SLIP2d.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = False

        super(DualSLIP2D, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'dual_SLIP_2d'


class SLIP3D(LeggedRobot):
    r"""Linear Inverted Pendulum Model 3D

    This class describes the 3D Spring-Loaded Inverted Pendulum (3D-SLIP) model, which underlies dynamic locomotive
    behaviors (such as running and jumping), and is often used as a template model in locomotion.

    See Also:
        - DualSLIP3D: the dual version of the 3D-SLIP
    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/templates/SLIP3d.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = False

        super(SLIP3D, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'SLIP_3d'


class DualSLIP3D(LeggedRobot):
    r"""Dual Linear Inverted Pendulum Model 3D

    This class describes the dual 3D Spring-Loaded Inverted Pendulum (dual 3D-SLIP) model, which underlies dynamic
    locomotive behaviors (such as running and jumping), and is often used as a template model in locomotion.
    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/templates/dual_SLIP3d.xml'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1.)
        if fixed_base is None:
            fixed_base = False

        super(DualSLIP3D, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'dual_SLIP_3d'


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
    robot = SLIP2D(sim)

    # print information about the robot
    robot.print_info()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
