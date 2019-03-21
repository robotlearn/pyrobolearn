#!/usr/bin/env python
"""Provide the Pleurobot robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import QuadrupedRobot
from pyrobolearn.robots.uuv import UUVRobot
from pyrobolearn.robots.usv import USVRobot


class Pleurobot(QuadrupedRobot, UUVRobot, USVRobot):
    r"""Pleurobot Salamander robot

    References:
        [1] https://github.com/KM-RoBoTa/pleurobot_ros_pkg
        [2] https://biorob.epfl.ch/pleurobot
    """

    def __init__(self, simulator, init_pos=(0, 0, 0), init_orient=(0, 0, 0, 1), useFixedBase=False, scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/pleurobot/pleurobot.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Pleurobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'pleurobot'


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Pleurobot(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        # robot.computeAndDrawCoMPosition()
        # robot.computeAndDrawProjectedCoMPosition()
        world.step(sleep_dt=1./240)
