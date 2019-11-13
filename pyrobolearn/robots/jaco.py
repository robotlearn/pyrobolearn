#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Jaco robotic platform.
"""

import os

from pyrobolearn.robots.manipulator import Manipulator
from pyrobolearn.robots.gripper import AngularGripper


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Jaco(Manipulator):
    r"""Jaco (manipulator) robot

    References:
        - [1] https://github.com/JenniferBuehler/jaco-arm-pkgs
        - [2] https://github.com/Kinovarobotics/kinova-ros
        - [3] https://github.com/RIVeR-Lab/wpi_jaco
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=True, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/jaco/jaco.urdf'):
        """
        Initialize the Jaco manipulator.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(Jaco, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'jaco'


class JacoGripper(AngularGripper):
    r"""Jaco Gripper

    References:
        - [1] https://github.com/JenniferBuehler/jaco-arm-pkgs
        - [2] https://github.com/Kinovarobotics/kinova-ros
        - [3] https://github.com/RIVeR-Lab/wpi_jaco
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0.707, 0, 0.707), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/jaco/jaco_gripper.urdf'):
        """
        Initialize the Jaco gripper.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the gripper will be fixed in the world.
            scale (float): scaling factor that is used to scale the gripper.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0.707, 0, 0.707)
        if fixed_base is None:
            fixed_base = True

        super(JacoGripper, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'jaco_gripper'


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
    robot = Jaco(sim)

    # print information about the robot
    robot.print_info()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
