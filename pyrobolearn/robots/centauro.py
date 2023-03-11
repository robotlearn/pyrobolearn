#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Centauro robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import QuadrupedRobot
from pyrobolearn.robots.wheeled_robot import WheeledRobot
from pyrobolearn.robots.manipulator import BiManipulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Centauro(WheeledRobot, QuadrupedRobot, BiManipulator):
    r"""Centauro robot

    References:
        - [1] https://github.com/ADVRHumanoids/centauro-simulator
    """

    def __init__(self, simulator, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/centauro/centauro_stick.urdf'
                 # centauro_stick.urdf, centauro_soft_hand.urdf, centauro_heri.urdf,
                 # centauro_schunk_handL.urdf, centauro_schunk_hand.urdf
                 ):
        """
        Initialize the Centauro robot.

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

        super(Centauro, self).__init__(simulator, urdf, position, orientation, fixed_base, scale=scale)
        self.name = 'centauro'

        self.necks = [self.get_link_ids(link) for link in ['neck_' + str(i) for i in range(1, 4)]]

        # define wheels
        self.wheels = [self.get_link_ids(link) for link in ['wheel_' + str(i) for i in range(1, 5)]]
        self.wheel_directions = np.array([-1., 1., -1., 1])

        # define legs and feet
        self.legs = [[self.get_link_ids(link) for link in links]
                     for links in [['hip1_1', 'hip2_1', 'knee_1', 'ankle1_1', 'ankle2_1', 'wheel_1'],
                                   ['hip1_2', 'hip2_2', 'knee_2', 'ankle1_2', 'ankle2_2', 'wheel_2'],
                                   ['hip1_3', 'hip2_3', 'knee_3', 'ankle1_3', 'ankle2_3', 'wheel_3'],
                                   ['hip1_4', 'hip2_4', 'knee_4', 'ankle1_4', 'ankle2_4', 'wheel_4']]]

        self.feet = self.wheels

        # define arms and hands
        self.arms = [[self.get_link_ids(link) for link in links]
                     for links in [['arm1_1', 'arm1_2', 'arm1_3', 'arm1_4', 'arm1_5', 'arm1_6', 'arm1_7'],
                                   ['arm2_1', 'arm2_2', 'arm2_3', 'arm2_4', 'arm2_5', 'arm2_6', 'arm2_7']]]

        self.hands = [self.get_link_ids(link) for link in ['arm1_8', 'arm2_8']]

        # load joint configurations
        srdf = os.path.dirname(__file__) + '/urdfs/centauro/centauro.srdf'
        self.load_joint_configurations(srdf)
        # print(self._joint_configuration.keys())
        joint_ids, joint_values = self._joint_configuration['home']
        self.reset_joint_states(q=joint_values, joint_ids=joint_ids)


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # load robot
    robot = Centauro(sim)  # , useFixedBase=True)

    # print information about the robot
    robot.print_info()
    print("Number of Legs: {}".format(robot.num_legs))
    print("Number of Arms: {}".format(robot.num_arms))

    # robot.add_joint_slider(robot.left_arm)
    # robot.drive(speed=3)

    # run simulator
    for _ in count():
        robot.update_joint_slider()
        world.step(sleep_dt=1./240)
