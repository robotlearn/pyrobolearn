#!/usr/bin/env python
"""Provide the Ballbot robotic platform.
"""

import os

from pyrobolearn.robots.robot import Robot


# TODO inertia are not corrects
class Ballbot(Robot):
    r"""BB8 robot

    References:
        [1] https://github.com/eborghi10/BB-8-ROS
        [2] http://www.theconstructsim.com/bb-8-gazebo-model/
    """

    def __init__(self, simulator, init_pos=(0, 0, 0.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1., urdf_path=os.path.dirname(__file__) + '/urdfs/ballbot/ballbot.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Ballbot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.ball = self.sim.loadURDF(os.path.dirname(__file__) + '/urdfs/ballbot/ball.urdf', init_pos, init_orient)
        self.name = 'ballbot'


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
    robot = Ballbot(sim)

    # print information about the robot
    robot.printRobotInfo()

    for i in count():
        # robot.setJointVelocities([0, -1, 0])

        # step in simulation
        world.step(sleep_dt=1./240)
