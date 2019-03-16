#!/usr/bin/env python
"""Provide the Centauro robotic platform.
"""

import os
import numpy as np

from legged_robot import QuadrupedRobot
from wheeled_robot import WheeledRobot
from manipulator import BiManipulatorRobot


class Centauro(WheeledRobot, QuadrupedRobot, BiManipulatorRobot):

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 1.),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/centauro/centauro_stick.urdf'
                 # centauro_stick.urdf, centauro_soft_hand.urdf, centauro_heri.urdf,
                 # centauro_schunk_handL.urdf, centauro_schunk_hand.urdf
                 ):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 1.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (1.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Centauro, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase)
        self.name = 'centauro'

        self.necks = [self.getLinkIds(link) for link in ['neck_'+str(i) for i in range(1, 4)]]

        # define wheels
        self.wheels = [self.getLinkIds(link) for link in ['wheel_'+str(i) for i in range(1, 5)]]
        self.wheel_directions = np.array([-1., 1., -1., 1])

        # define legs and feet
        self.legs = [[self.getLinkIds(link) for link in links]
                     for links in [['hip1_1', 'hip2_1', 'knee_1', 'ankle1_1', 'ankle2_1', 'wheel_1'],
                                   ['hip1_2', 'hip2_2', 'knee_2', 'ankle1_2', 'ankle2_2', 'wheel_2'],
                                   ['hip1_3', 'hip2_3', 'knee_3', 'ankle1_3', 'ankle2_3', 'wheel_3'],
                                   ['hip1_4', 'hip2_4', 'knee_4', 'ankle1_4', 'ankle2_4', 'wheel_4']]]

        self.feet = self.wheels

        # define arms and hands
        self.arms = [[self.getLinkIds(link) for link in links]
                     for links in [['arm1_1', 'arm1_2', 'arm1_3', 'arm1_4', 'arm1_5', 'arm1_6', 'arm1_7'],
                                   ['arm2_1', 'arm2_2', 'arm2_3', 'arm2_4', 'arm2_5', 'arm2_6', 'arm2_7']]]

        self.hands = [self.getLinkIds(link) for link in ['arm1_8', 'arm2_8']]


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # load robot
    robot = Centauro(sim)  # , useFixedBase=True)

    # print information about the robot
    robot.printRobotInfo()
    print("Number of Legs: {}".format(robot.getNumberOfLegs()))
    print("Number of Arms: {}".format(robot.getNumberOfArms()))

    # robot.addJointSlider(robot.getRightFrontLegIds())
    robot.drive(speed=3)

    # run simulator
    for _ in count():
        robot.updateJointSlider()
        world.step(sleep_dt=1./240)
