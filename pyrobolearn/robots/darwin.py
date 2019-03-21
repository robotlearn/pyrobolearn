#!/usr/bin/env python
"""Provide the Darwin robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot


class Darwin(BipedRobot, BiManipulatorRobot):
    r"""Darwin robot

    References:
        [1] https://github.com/HumaRobotics/darwin_description
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.34),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/darwin/darwin.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.34)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos += (0.34,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Darwin, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'darwin'

        # self.torso = [self.getLinkIds(link) for link in ['DWL', 'DWS', 'DWYTorso'] if link in self.link_names]
        #
        # self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
        #              for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
        #                            ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]
        #
        # self.feet = [self.getLinkIds(link) for link in ['LFoot', 'RFoot'] if link in self.link_names]
        #
        # self.arms = [[self.getLinkIds(link) for link in links if link in self.link_names]
        #              for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
        #                            ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]
        #
        # self.hands = [self.getLinkIds(link) for link in ['LSoftHand', 'RSoftHand'] if link in self.link_names]

        # move a little bit the arms
        # self.setJointPositions([-np.pi/12, np.pi/12], [self.joints[i] for i in [4,11]]) # shoulder
        # self.setJointPositions([-np.pi / 4, -np.pi / 4], [self.joints[i] for i in [6, 13]]) # elbow
        # self.setJointPositions([-np.pi/4, -np.pi/4], [self.joints[i] for i in [17,23]])
        # for _ in range(10):
        #     self.sim.stepSimulation()


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
    robot = Darwin(sim, useFixedBase=False)

    # print information about the robot
    robot.printRobotInfo()
    print(robot.link_names)

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        world.step(sleep_dt=1./240)
