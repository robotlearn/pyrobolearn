#!/usr/bin/env python
"""Provide the Husky robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.wheeled_robot import DifferentialWheeledRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Husky(DifferentialWheeledRobot):
    r"""Husky robot

    Husky robot from Clearpath Robotics [1].

    References:
        [1] Clearpath Robotics: https://www.clearpathrobotics.com/husky-unmanned-ground-vehicle-robot/
        [2] ROS wiki: http://wiki.ros.org/Robots/Husky
        [3] Github: https://github.com/husky/husky
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, .14),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/husky/husky.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.14)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.14,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Husky, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'husky'

        self.wheels = [self.get_link_ids(link) for link in ['front_left_wheel_link', 'front_right_wheel_link',
                                                            'rear_left_wheel_link', 'rear_right_wheel_link']
                       if link in self.link_names]
        self.wheel_directions = np.ones(len(self.wheels))


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
    robot = Husky(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        robot.drive_forward(2)
        world.step(sleep_dt=1./240)
