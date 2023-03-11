#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Atlas robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Atlas(BipedRobot, BiManipulator):
    r"""Atlas robot

    Atlas robot developed by Boston Dynamics.

    References:
        - [1] Boston Dynamics: https://www.bostondynamics.com/atlas
        - [2] URDF: https://github.com/openai/roboschool/tree/master/roboschool/models_robot/atlas_description
    """

    def __init__(self, simulator, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/atlas/atlas_v4_with_multisense.urdf'):
        """
        Initialize the Atlas robot.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 1.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (1.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Atlas, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'atlas'

        self.head = self.get_link_ids('head') if 'head' in self.link_names else None

        self.torso = [self.get_link_ids(link) for link in ['ltorso', 'mtorso', 'utorso'] if link in self.link_names]

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['l_uglut', 'l_lglut', 'l_uleg', 'l_lleg', 'l_talus', 'l_foot'],
                                   ['r_uglut', 'r_lglut', 'r_uleg', 'r_lleg', 'r_talus', 'r_foot']]]

        self.feet = [self.get_link_ids(link) for link in ['l_foot', 'r_foot'] if link in self.link_names]

        self.arms = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['l_clav', 'l_scap', 'l_uarm', 'l_larm', 'l_ufarm', 'l_lfarm', 'l_hand'],
                                   ['r_clav', 'r_scap', 'r_uarm', 'r_larm', 'r_ufarm', 'r_lfarm', 'r_hand']]]

        self.hands = [self.get_link_ids(link) for link in ['l_hand', 'r_hand'] if link in self.link_names]


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
    robot = Atlas(sim)

    # print information about the robot
    robot.print_info()

    # position control using sliders
    robot.add_joint_slider(robot.left_leg)

    # run simulator
    for _ in count():
        robot.update_joint_slider()
        world.step(sleep_dt=1./240)
