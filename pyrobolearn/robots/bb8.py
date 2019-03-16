#!/usr/bin/env python
"""Provide the BB8 robotic platform.
"""

import os
from robot import Robot


class BB8(Robot):
    r"""BB8 robot

    References:
        [1] https://github.com/eborghi10/BB-8-ROS
        [2] http://www.theconstructsim.com/bb-8-gazebo-model/
    """

    def __init__(self, simulator, init_pos=(0, 0, 0.4), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1., urdf_path=os.path.dirname(__file__) + '/urdfs/bb8/bb8.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.4)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.4,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(BB8, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'bb8'


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
    robot = BB8(sim)

    # print information about the robot
    robot.printRobotInfo()

    for i in count():
        robot.setJointVelocities([0, -1, 0])

        # step in simulation
        world.step(sleep_dt=1./240)
