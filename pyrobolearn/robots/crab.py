#!/usr/bin/env python
"""Provide the Crab robotic platform.
"""

import os
from legged_robot import HexapodRobot


class Crab(HexapodRobot):
    r"""Crab Hexapod robot

    References:
        [1] http://wiki.ros.org/Robots/HexapodRobot
        [2] https://github.com/tuuzdu/crab_project
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.12),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/crab/crab.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.12)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.12,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Crab, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'crab'

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['coxa_l1', 'femur_l1', 'tibia_l1'],
                                   [ 'coxa_r1', 'femur_r1', 'tibia_r1'],
                                   ['coxa_l2', 'femur_l2', 'tibia_l2'],
                                   ['coxa_r2', 'femur_r2', 'tibia_r2'],
                                   ['coxa_l3', 'femur_l3', 'tibia_l3'],
                                   ['coxa_r3', 'femur_r3', 'tibia_r3']]]

        self.feet = [self.getLinkIds(link) for link in ['tibia_foot_l1', 'tibia_foot_r1', 'tibia_foot_l2',
                                                        'tibia_foot_r2', 'tibia_foot_l3', 'tibia_foot_r3']
                     if link in self.link_names]


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
    robot = Crab(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    robot.addJointSlider(robot.right_middle_leg)

    # run simulation
    for i in count():
        robot.updateJointSlider()
        # step in simulation
        world.step(sleep_dt=1./240)
