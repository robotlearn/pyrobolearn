#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Ballbot robotic platform.
"""

import os

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO inertia are not corrects
class Ballbot(Robot):
    r"""Ballbot robot

    References:
        - [1] https://github.com/eborghi10/BB-8-ROS
        - [2] http://www.theconstructsim.com/bb-8-gazebo-model/
    """

    def __init__(self, simulator, position=(0, 0, 0.), orientation=(0, 0, 0, 1), fixed_base=False,
                 scale=1., urdf=os.path.dirname(__file__) + '/urdfs/ballbot/ballbot.urdf'):
        """
        Initialize the Ballbot robot.

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

        super(Ballbot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.ball = self.sim.load_urdf(os.path.dirname(__file__) + '/urdfs/ballbot/ball.urdf', position, orientation)
        self.name = 'ballbot'


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
    robot = Ballbot(sim)

    # print information about the robot
    robot.print_info()

    for i in count():
        # robot.set_joint_velocities([0, -1, 0])

        # step in simulation
        world.step(sleep_dt=1./240)
