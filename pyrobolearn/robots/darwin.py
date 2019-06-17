#!/usr/bin/env python
"""Provide the Darwin robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Darwin(BipedRobot, BiManipulatorRobot):
    r"""Darwin robot

    References:
        [1] https://github.com/HumaRobotics/darwin_description
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.34),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/darwin/darwin.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.34)
        if len(position) == 2:  # assume x, y are given
            position += (0.34,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Darwin, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'darwin'

        # self.torso = [self.get_link_ids(link) for link in ['DWL', 'DWS', 'DWYTorso'] if link in self.link_names]
        #
        # self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
        #              for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
        #                            ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]
        #
        # self.feet = [self.get_link_ids(link) for link in ['LFoot', 'RFoot'] if link in self.link_names]
        #
        # self.arms = [[self.get_link_ids(link) for link in links if link in self.link_names]
        #              for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
        #                            ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]
        #
        # self.hands = [self.get_link_ids(link) for link in ['LSoftHand', 'RSoftHand'] if link in self.link_names]

        # move a little bit the arms
        # self.set_joint_positions([-np.pi/12, np.pi/12], [self.joints[i] for i in [4,11]]) # shoulder
        # self.set_joint_positions([-np.pi / 4, -np.pi / 4], [self.joints[i] for i in [6, 13]]) # elbow
        # self.set_joint_positions([-np.pi/4, -np.pi/4], [self.joints[i] for i in [17,23]])
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
    robot = Darwin(sim, fixed_base=False)

    # print information about the robot
    robot.print_info()
    print(robot.link_names)

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
