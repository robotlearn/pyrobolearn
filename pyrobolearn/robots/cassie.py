#!/usr/bin/env python
"""Provide the Cassie robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import BipedRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Cassie(BipedRobot):
    r"""Cassie robot

    This class describes the cassie robot developed by Agility Robotics.

    References:
        [1] http://www.agilityrobotics.com/sims/
        [2] "Feedback Control For Cassie With Deep Reinforcement Learning", Xie et al., 2018
            https://arxiv.org/abs/1803.05580
        [3] https://github.com/agilityrobotics/cassie-gazebo-sim
        [4] https://github.com/UMich-BipedLab/Cassie_Model
        [5] https://github.com/UMich-BipedLab/cassie_description
        [6] https://github.com/erwincoumans/pybullet_robots/tree/master/data/cassie
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, .8),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/cassie/cassie.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.8)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.8,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Cassie, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'cassie'

        # TODO: create constraints in pybullet
        # From the user guide: "URDF, SDF and MJCF specify articulated bodies as a tree-structures without loops.
        # The 'createConstraint' allows you to connect specific links of bodies to close those loops."

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['left_pelvis_rotation', 'left_hip', 'left_thigh', 'left_knee', 'left_shin',
                                    'left_tarsus', 'left_toe'],
                                   ['right_pelvis_rotation', 'right_hip', 'right_thigh', 'right_knee', 'right_shin',
                                    'right_tarsus', 'right_toe']]]

        self.feet = [self.get_link_ids(link) for link in ['left_toe', 'right_toe'] if link in self.link_names]

        # set joint angles to home position
        self.set_joint_home_positions()

    def get_home_joint_positions(self):
        """Return the joint positions for the home position"""
        return np.array([0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9, 0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9])


if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Cassie(sim)

    # print information about the robot
    robot.print_info()

    # position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        robot.move_joint_home_positions()
        world.step(sleep_dt=1./240)
