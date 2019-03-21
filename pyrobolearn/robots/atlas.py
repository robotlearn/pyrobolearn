#!/usr/bin/env python
"""Provide the Atlas robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot


class Atlas(BipedRobot, BiManipulatorRobot):
    r"""Atlas robot

    Atlas robot developed by Boston Dynamics.

    References:
        [1] Boston Dynamics: https://www.bostondynamics.com/atlas
        [2] URDF: https://github.com/openai/roboschool/tree/master/roboschool/models_robot/atlas_description
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 1.),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/atlas/atlas_v4_with_multisense.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 1.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (1.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Atlas, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'atlas'

        self.head = self.getLinkIds('head') if 'head' in self.link_names else None

        self.torso = [self.getLinkIds(link) for link in ['ltorso', 'mtorso', 'utorso'] if link in self.link_names]

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['l_uglut', 'l_lglut', 'l_uleg', 'l_lleg', 'l_talus', 'l_foot'],
                                   ['r_uglut', 'r_lglut', 'r_uleg', 'r_lleg', 'r_talus', 'r_foot']]]

        self.feet = [self.getLinkIds(link) for link in ['l_foot', 'r_foot'] if link in self.link_names]

        self.arms = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['l_clav', 'l_scap', 'l_uarm', 'l_larm', 'l_ufarm', 'l_lfarm', 'l_hand'],
                                   ['r_clav', 'r_scap', 'r_uarm', 'r_larm', 'r_ufarm', 'r_lfarm', 'r_hand']]]

        self.hands = [self.getLinkIds(link) for link in ['l_hand', 'r_hand'] if link in self.link_names]


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
    robot = Atlas(sim)

    # print information about the robot
    robot.printRobotInfo()

    # position control using sliders
    robot.addJointSlider(robot.getLeftLegIds())

    # run simulator
    for _ in count():
        robot.updateJointSlider()
        world.step(sleep_dt=1./240)
