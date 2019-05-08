#!/usr/bin/env python
"""Provide the Crab robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import HexapodRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Crab(HexapodRobot):
    r"""Crab Hexapod robot

    References:
        [1] http://wiki.ros.org/Robots/HexapodRobot
        [2] https://github.com/tuuzdu/crab_project
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.12),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/crab/crab.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.12)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.12,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Crab, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'crab'

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['coxa_l1', 'femur_l1', 'tibia_l1'],
                                   [ 'coxa_r1', 'femur_r1', 'tibia_r1'],
                                   ['coxa_l2', 'femur_l2', 'tibia_l2'],
                                   ['coxa_r2', 'femur_r2', 'tibia_r2'],
                                   ['coxa_l3', 'femur_l3', 'tibia_l3'],
                                   ['coxa_r3', 'femur_r3', 'tibia_r3']]]

        self.feet = [self.get_link_ids(link) for link in ['tibia_foot_l1', 'tibia_foot_r1', 'tibia_foot_l2',
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
    robot.print_info()

    # Position control using sliders
    robot.add_joint_slider(robot.right_middle_leg)

    # run simulation
    for i in count():
        robot.update_joint_slider()
        # step in simulation
        world.step(sleep_dt=1./240)
