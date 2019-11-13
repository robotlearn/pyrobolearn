#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Baxter robotic platform.
"""

import os

from pyrobolearn.robots.manipulator import BiManipulator
from pyrobolearn.robots.gripper import ParallelGripper


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Baxter(BiManipulator):
    r"""Baxter robot

    Baxter robot built by Rethink Robotics.

    References:
        - [1] Rethink Robotics: https://www.rethinkrobotics.com/
        - [2] https://github.com/RethinkRobotics/baxter_common
    """

    def __init__(self, simulator, position=(0, 0, 0.95), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/baxter/baxter.urdf'):
        """
        Initialize the Baxter robot.

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
            position = (0., 0., 0.95)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.95,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Baxter, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'baxter'

        self.head = self.get_link_ids('head') if 'head' in self.link_names else None

        self.arms = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['left_upper_shoulder', 'left_lower_shoulder', 'left_upper_elbow',
                                    'left_lower_elbow', 'left_upper_forearm', 'left_lower_forearm',
                                    'left_wrist', 'l_gripper_l_finger', 'l_gripper_r_finger'],
                                   ['right_upper_shoulder', 'right_lower_shoulder', 'right_upper_elbow',
                                    'right_lower_elbow', 'right_upper_forearm', 'right_lower_forearm',
                                    'right_wrist', 'r_gripper_l_finger', 'r_gripper_r_finger']]]

        self.hands = [self.get_link_ids(link) for link in ['left_gripper', 'right_gripper'] if link in self.link_names]


class BaxterGripper(ParallelGripper):
    r"""Baxter Gripper

    Baxter robot built by Rethink Robotics.

    References:
        - [1] Rethink Robotics: https://www.rethinkrobotics.com/
        - [2] https://github.com/RethinkRobotics/baxter_common
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/baxter/baxter_gripper.urdf'):
        """
        Initialize the Baxter gripper.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the gripper will be fixed in the world.
            scale (float): scaling factor that is used to scale the gripper.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(BaxterGripper, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'baxter_gripper'


if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Baxter(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
