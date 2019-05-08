#!/usr/bin/env python
"""Provide the Fetch robotic platform.
"""

import os

from pyrobolearn.robots.wheeled_robot import WheeledRobot
from pyrobolearn.robots.manipulator import ManipulatorRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Fetch(WheeledRobot, ManipulatorRobot):
    r"""Fetch robot

    References:
        [1] Fetch robotics: https://fetchrobotics.com/
        [2] Fetch description: https://github.com/fetchrobotics/fetch_ros
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.1),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/fetch/fetch.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.1)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.1,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Fetch, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'fetch'


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
    robot = Fetch(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    robot.add_joint_slider()

    # run simulator
    for _ in count():
        robot.update_joint_slider()
        world.step(sleep_dt=1./240)
