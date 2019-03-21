#!/usr/bin/env python
"""Provide the Coman robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot


class Coman(BipedRobot, BiManipulatorRobot):
    r"""Coman robot

    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.5),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/coman/coman.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.5)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.5,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Coman, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'coman'

        self.torso = [self.getLinkIds(link) for link in ['DWL', 'DWS', 'DWYTorso'] if link in self.link_names]

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
                                   ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]

        self.feet = [self.getLinkIds(link) for link in ['LFoot', 'RFoot'] if link in self.link_names]

        self.arms = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
                                   ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]

        self.hands = [self.getLinkIds(link) for link in ['LSoftHand', 'RSoftHand'] if link in self.link_names]

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
    robot = Coman(sim, useFixedBase=True)

    # print information about the robot
    robot.printRobotInfo()
    print(robot.link_names)

    # # Position control using sliders
    # robot.addJointSlider()

    robot.changeTransparency()
    # robot.drawLinkCoMs()
    robot.drawLinkFrames()
    # robot.drawBoundingBoxes(robot.right_leg[4])

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        # robot.computeAndDrawCoMPosition()
        # robot.computeAndDrawProjectedCoMPosition()
        world.step(sleep_dt=1./240)
