#!/usr/bin/env python
"""Provide the Coman robotic platform.
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


class Coman(BipedRobot, BiManipulatorRobot):
    r"""Coman robot

    References:
        [1] https://github.com/ADVRHumanoids/iit-coman-ros-pkg
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.5),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/coman/coman.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.5)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.5,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Coman, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'coman'

        self.torso = [self.get_link_ids(link) for link in ['DWL', 'DWS', 'DWYTorso'] if link in self.link_names]

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
                                   ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]

        self.feet = [self.get_link_ids(link) for link in ['LFoot', 'RFoot'] if link in self.link_names]

        self.arms = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
                                   ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]

        self.hands = [self.get_link_ids(link) for link in ['LSoftHand', 'RSoftHand'] if link in self.link_names]

        # move a little bit the arms
        # self.set_joint_positions([-np.pi/12, np.pi/12], [self.joints[i] for i in [4,11]]) # shoulder
        # self.set_joint_positions([-np.pi / 4, -np.pi / 4], [self.joints[i] for i in [6, 13]]) # elbow
        # self.set_joint_positions([-np.pi/4, -np.pi/4], [self.joints[i] for i in [17,23]])
        # for _ in range(10):
        #     self.sim.stepSimulation()


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Coman(sim, fixed_base=True)

    # print information about the robot
    robot.print_info()
    print(robot.link_names)

    # # Position control using sliders
    # robot.add_joint_slider()

    robot.change_transparency()
    # robot.draw_link_coms()
    robot.draw_link_frames()
    # robot.draw_joint_frames()
    # robot.draw_joint_axes()
    # robot.draw_bounding_boxes(robot.right_leg[4])

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        # robot.compute_and_draw_com_position()
        # robot.compute_and_draw_projected_com_position()
        world.step(sleep_dt=1./240)
