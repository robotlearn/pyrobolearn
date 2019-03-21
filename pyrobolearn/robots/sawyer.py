#!/usr/bin/env python
"""Provide the Sawyer robotic platform.
"""

import os

from pyrobolearn.robots.wheeled_robot import WheeledRobot
from pyrobolearn.robots.manipulator import ManipulatorRobot


class Sawyer(ManipulatorRobot, WheeledRobot):
    r"""Sawyer robot

    Sawyer robot built by Rethink Robotics.

    References:
        [1] Rethink Robotics
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.92),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=True,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/sawyer/sawyer.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.92)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.92,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(Sawyer, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'sawyer'

        self.head = self.getLinkIds('head') if 'head' in self.link_names else None

        self.arms = [[self.getLinkIds(link) for link in ['right_l0', 'right_l1', 'right_l2', 'right_l3', 'right_l4',
                                                         'right_l5', 'right_l6'] if link in self.link_names]]

        self.hands = [self.getLinkIds(link) for link in ['right_l6'] if link in self.link_names]


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
    robot = Sawyer(sim)

    # print information about the robot
    robot.printRobotInfo()

    # # Position control using sliders
    robot.addJointSlider()

    # run simulator
    for _ in count():
        robot.updateJointSlider()
        world.step(sleep_dt=1./240)
