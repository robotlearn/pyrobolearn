#!/usr/bin/env python
"""Provide the SEA Hexapod robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import HexapodRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SEAHexapod(HexapodRobot):
    r"""SEA Hexapod robot (from CMU Biorobotics Lab)

    References:
        [1] https://github.com/alexansari101/snake_ws
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.15),
                 orientation=(0, 0, 0.707, 0.707),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/cmu_sea/hexapod.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.15)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.15,)
        if orientation is None:
            orientation = (0, 0, 0.707, 0.707)
        if fixed_base is None:
            fixed_base = False

        super(SEAHexapod, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'sea_hexapod'


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
    robot = SEAHexapod(sim)

    # print information about the robot
    robot.print_info()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
