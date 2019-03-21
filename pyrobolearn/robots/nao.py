#!/usr/bin/env python
"""Provide the Nao robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot
from pyrobolearn.robots.hand import TwoHand


class Nao(BipedRobot, BiManipulatorRobot, TwoHand):
    r"""Nao robot

    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.35),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/nao/nao_v40.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.35)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.35,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Nao, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'nao'

        self.neck = self.getLinkIds('Neck') if 'Neck' in self.link_names else None
        self.head = self.getLinkIds('Head') if 'Head' in self.link_names else None

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['LPelvis', 'LHip', 'LThigh', 'LTibia', 'LAnklePitch', 'l_ankle'],
                                   ['RPelvis', 'RHip', 'RThigh', 'RTibia', 'RAnklePitch', 'r_ankle']]]

        self.feet = [self.getLinkIds(link) for link in ['l_sole', 'r_sole'] if link in self.link_names]

        self.arms = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['LShoulder', 'LBicep', 'LElbow', 'LForeArm', 'l_wrist', 'l_gripper'],
                                   ['RShoulder', 'RBicep', 'RElbow', 'RForeArm', 'r_wrist', 'r_gripper']]]

        self.hands = [self.getLinkIds(link) for link in ['l_gripper', 'r_gripper'] if link in self.link_names]

        self.fingers = [[self.getLinkIds(link) for link in links if link in self.link_names]
                        for links in [['LThumb1_link', 'LThumb2_link'],
                                      ['LFinger11_link', 'LFinger12_link', 'LFinger13_link'],
                                      ['LFinger21_link', 'LFinger22_link', 'LFinger23_link'],
                                      ['RThumb1_link', 'RThumb2_link'],
                                      ['RFinger11_link', 'RFinger12_link', 'RFinger13_link'],
                                      ['RFinger21_link', 'RFinger22_link', 'RFinger23_link']]]

        self.left_fingers = [0, 1, 2]
        self.right_fingers = [3, 4, 5]


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
    robot = Nao(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider(robot.getLeftArmIds())

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        world.step(sleep_dt=1./240)
