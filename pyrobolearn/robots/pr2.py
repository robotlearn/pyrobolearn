#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the PR2 robotic platform.
"""

import os

from pyrobolearn.robots.wheeled_robot import WheeledRobot
from pyrobolearn.robots.manipulator import BiManipulator
from pyrobolearn.robots.gripper import AngularGripper


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PR2(WheeledRobot, BiManipulator):
    r"""PR2 robot

    References:
        - [1] http://www.willowgarage.com/pages/pr2/overview
        - [2] https://github.com/pr2/pr2_common
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/pr2/pr2.urdf'):
        """
        Initialize the PR2 robot.

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
            fixed_base = False

        super(PR2, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'pr2'


class PR2Gripper(AngularGripper):
    r"""PR2 Gripper

    References:
        - [1] http://www.willowgarage.com/pages/pr2/overview
        - [2] https://github.com/pr2/pr2_common
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, -0.707, 0, 0.707), fixed_base=True, scale=1.,
                 urdf=os.path.dirname(os.path.abspath(__file__)) + '/urdfs/pr2/pr2_gripper.urdf'):
        """
        Initialize the PR2 Gripper.

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
            orientation = (0, -0.707, 0, 0.707)
        if fixed_base is None:
            fixed_base = True

        super(PR2Gripper, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'pr2_gripper'


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
    robot = PR2(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
