#!/usr/bin/env python
"""Provide the Epuck robotic platform.
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


class Epuck(DifferentialWheeledRobot):
    r"""Epuck robot

    References:
        [1] http://www.e-puck.org/
        [2] http://www.gctronic.com/doc/index.php/E-Puck
        [3] https://github.com/gctronic/epuck_driver_cpp
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/epuck/epuck.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.0,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Epuck, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'epuck'

        self.wheels = [self.get_link_ids(link) for link in ['left_wheel', 'right_wheel']
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
    robots = []
    for _ in range(5):
        x, y = np.random.uniform(low=-2, high=2, size=2)
        robot = world.load_robot(Epuck, position=(x, y, 0))
        robots.append(robot)

    # print information about the robot
    robots[0].print_info()

    # Position control using sliders
    # robots[0].add_joint_slider()

    # run simulator
    for _ in count():
        # robots[0].update_joint_slider()
        for robot in robots:
            robot.drive(5)
        world.step(sleep_dt=1./240)
