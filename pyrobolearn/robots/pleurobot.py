#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Pleurobot robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import QuadrupedRobot
from pyrobolearn.robots.uuv import UUVRobot
from pyrobolearn.robots.usv import USVRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Pleurobot(QuadrupedRobot, UUVRobot, USVRobot):
    r"""Pleurobot Salamander robot

    References:
        - [1] https://github.com/KM-RoBoTa/pleurobot_ros_pkg
        - [2] https://biorob.epfl.ch/pleurobot
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/pleurobot/pleurobot.urdf'):
        """
        Initialize the Pleurobot robot.

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

        super(Pleurobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'pleurobot'


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
    robot = Pleurobot(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        # robot.compute_and_draw_com_position()
        # robot.compute_and_draw_projected_com_position()
        world.step(sleep_dt=1./240)
