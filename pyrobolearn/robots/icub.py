# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the ICub robotic platform.
"""

# TODO: fix bug

import os

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ICub(BipedRobot, BiManipulator):
    r"""ICub robot

    References:
        - [1] http://www.icub.org/
        - [2] https://github.com/robotology-playground/icub-models
        - [3] https://github.com/robotology-playground/icub-model-generator
    """

    def __init__(self, simulator, position=(0, 0, 0.7), orientation=(0, 0, 1, 0), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/icub/icub-v2.5+.urdf'):
        """
        Initialize the ICub robot.

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
            position = (0., 0., 0.7)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.7,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(ICub, self).__init__(simulator, urdf, position, orientation, fixed_base, scale=scale)
        self.name = 'icub'

        self.head = self.get_link_ids('head') if 'head' in self.link_names else None
        self.neck = [self.get_link_ids(link) for link in ['neck_1', 'neck_2'] if link in self.link_names]
        self.torso = [self.get_link_ids(link) for link in ['torso_1', 'torso_2', 'chest'] if link in self.link_names]

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['l_hip_1', 'l_hip_2', 'l_upper_leg', 'l_lower_leg', 'l_ankle_1', 'l_ankle_2'],
                                   ['r_hip_1', 'r_hip_2', 'r_upper_leg', 'r_lower_leg', 'r_ankle_1', 'r_ankle_2']]]

        self.feet = [self.get_link_ids(link) for link in ['l_sole', 'r_sole'] if link in self.link_names]

        self.arms = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['l_shoulder_1', 'l_shoulder_2', 'l_shoulder_3', 'l_elbow_1', 'l_forearm',
                                    'l_wrist_1', 'l_hand'],
                                   ['r_shoulder_1', 'r_shoulder_2', 'r_shoulder_3', 'r_elbow_1', 'r_forearm',
                                    'r_wrist_1', 'r_hand']]]

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
    robot = ICub(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
