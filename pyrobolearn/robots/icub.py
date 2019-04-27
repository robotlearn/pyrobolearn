#!/usr/bin/env python
"""Provide the ICub robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ICub(BipedRobot, BiManipulatorRobot):
    r"""ICub robot

    References:
        [1] http://www.icub.org/
        [2] https://github.com/robotology-playground/icub-models
        [3] https://github.com/robotology-playground/icub-model-generator
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.7),
                 orientation=(0, 0, 1, 0),
                 fixed_base=False,
                 scaling=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/icub/icub-v2.5+.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.7)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.7,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(ICub, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling=scaling)
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
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

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
