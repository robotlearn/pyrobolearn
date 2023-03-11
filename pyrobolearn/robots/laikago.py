#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Laikago robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import QuadrupedRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Laikago(QuadrupedRobot):
    r"""Laikago robot

    References:
        - [1] Laikago: http://www.unitree.cc/e/action/ShowInfo.php?classid=6&id=1
        - [2] https://github.com/erwincoumans/pybullet_robots/tree/master/data/laikago
    """

    def __init__(self, simulator, position=(0, 0, .5), orientation=(0.5, 0.5, 0.5, 0.5), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/laikago/laikago.urdf'):
        """
        Initialize the Laikago robot.

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
            position = (0., 0., 0.5)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.5,)
        if orientation is None:
            orientation = (0.5, 0.5, 0.5, 0.5)
        if fixed_base is None:
            fixed_base = False

        super(Laikago, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'laikago'

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['FL_hip_motor', 'FL_upper_leg', 'FL_lower_leg'],
                                   ['FR_hip_motor', 'FR_upper_leg', 'FR_lower_leg'],
                                   ['RL_hip_motor', 'RL_upper_leg', 'RL_lower_leg'],
                                   ['RR_hip_motor','RR_upper_leg', 'RR_lower_leg']]]

        self.feet = [self.get_link_ids(link) for link in ['FL_lower_leg', 'FR_lower_leg',
                                                        'RL_lower_leg', 'RR_lower_leg'] if link in self.link_names]

    def get_home_joint_positions(self):
        """Return the joint positions for the home position"""
        return np.zeros(self.num_actuated_joints)


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
    robot = Laikago(sim)

    # print information about the robot
    robot.print_info()

    # # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        robot.move_home_joint_positions()
        world.step(sleep_dt=1./240)
