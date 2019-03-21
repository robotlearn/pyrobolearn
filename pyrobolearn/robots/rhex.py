#!/usr/bin/env python
"""Provide the Rhex robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import HexapodRobot


# TODO add inertial tags
class Rhex(HexapodRobot):
    r"""Rhex Hexapod robot

    "RHex is a bio-inspired, hexapedal robot designed for locomotion in rough terrain." [1]
    It was created by researchers at the University of Michigan and McGill University.

    References:
        [1] https://robots.ieee.org/robots/rhex/
        [2] https://www.rhex.web.tr/
        [3] https://github.com/grafoteka/rhex
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.12),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/rhex/rhex.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.12)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.12,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Rhex, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'rhex'

        self.legs = [[self.getLinkIds(link + str(idx))] for link, idx in zip(['leg']*6, range(1, 7))
                     if link + str(idx) in self.link_names]

        self.feet = [self.getLinkIds(link + str(idx)) for link, idx in zip(['leg']*6, range(1, 7))
                     if link + str(idx) in self.link_names]

        self.leg_axis = np.ones(len(self.feet))

    def drive(self, speed):
        if isinstance(speed, (int, float)):
            speed = speed * np.ones(len(self.feet))
            speed = speed * self.leg_axis
        self.setJointVelocities(speed, self.feet)


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
    robot = Rhex(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider(robot.right_back_leg)

    # run simulation
    for i in count():
        # robot.updateJointSlider()
        robot.drive(2)
        # step in simulation
        world.step(sleep_dt=1./240)
