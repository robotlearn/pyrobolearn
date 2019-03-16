#!/usr/bin/env python
"""Provide the Baxter robotic platform.
"""

import os
from manipulator import BiManipulatorRobot


class Baxter(BiManipulatorRobot):
    r"""Baxter robot

    Baxter robot built by Rethink Robotics.

    References:
        [1] Rethink Robotics
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.95),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/baxter/baxter.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.95)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.95,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Baxter, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'baxter'

        self.head = self.getLinkIds('head') if 'head' in self.link_names else None

        self.arms = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['left_upper_shoulder', 'left_lower_shoulder', 'left_upper_elbow',
                                    'left_lower_elbow', 'left_upper_forearm', 'left_lower_forearm',
                                    'left_wrist', 'l_gripper_l_finger', 'l_gripper_r_finger'],
                                   ['right_upper_shoulder', 'right_lower_shoulder', 'right_upper_elbow',
                                    'right_lower_elbow', 'right_upper_forearm', 'right_lower_forearm',
                                    'right_wrist', 'r_gripper_l_finger', 'r_gripper_r_finger']]]

        self.hands = [self.getLinkIds(link) for link in ['left_gripper', 'right_gripper'] if link in self.link_names]


if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Baxter(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        world.step(sleep_dt=1./240)
