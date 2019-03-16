#!/usr/bin/env python
"""Provide the Cassie robotic platform.
"""

import os
import numpy as np
from legged_robot import BipedRobot


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
                 init_pos=(0, 0, .8),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/cassie/cassie.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.8)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.8,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Cassie, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'cassie'

        # TODO: create constraints in pybullet
        # From the user guide: "URDF, SDF and MJCF specify articulated bodies as a tree-structures without loops.
        # The 'createConstraint' allows you to connect specific links of bodies to close those loops."

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['left_pelvis_rotation', 'left_hip', 'left_thigh', 'left_knee', 'left_shin',
                                    'left_tarsus', 'left_toe'],
                                   ['right_pelvis_rotation', 'right_hip', 'right_thigh', 'right_knee', 'right_shin',
                                    'right_tarsus', 'right_toe']]]

        self.feet = [self.getLinkIds(link) for link in ['left_toe', 'right_toe'] if link in self.link_names]

        # set joint angles to home position
        self.setJointHomePositions()

    def getHomeJointPositions(self):
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
    robot.printRobotInfo()

    # position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        robot.moveJointHomePositions()
        world.step(sleep_dt=1./240)
