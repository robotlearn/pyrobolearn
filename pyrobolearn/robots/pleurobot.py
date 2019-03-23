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

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=False, scaling=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/pleurobot/pleurobot.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Pleurobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
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
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        # robot.compute_and_draw_com_position()
        # robot.compute_and_draw_projected_com_position()
        world.step(sleep_dt=1./240)
