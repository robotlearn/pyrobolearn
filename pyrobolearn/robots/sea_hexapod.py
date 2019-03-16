#!/usr/bin/env python
"""Provide the SEA Hexapod robotic platform.
"""

import os
from legged_robot import HexapodRobot


class SEAHexapod(HexapodRobot):
    r"""SEA Hexapod robot (from CMU Biorobotics Lab)

    References:
        [1] https://github.com/alexansari101/snake_ws
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.15),
                 init_orient=(0, 0, 0.707, 0.707),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/cmu_sea/hexapod.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.15)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.15,)
        if init_orient is None:
            init_orient = (0, 0, 0.707, 0.707)
        if useFixedBase is None:
            useFixedBase = False

        super(SEAHexapod, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
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
    robot.printRobotInfo()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
