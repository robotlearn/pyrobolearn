#!/usr/bin/env python
"""Provide the Morphex robotic platform.
"""

import os
from legged_robot import HexapodRobot


class Morphex(HexapodRobot):
    r"""Morphex Hexapod robot

    References:
        [1] http://zentasrobots.com/
        [2] https://gist.github.com/lanius/cb8b5e0ede9ff3b2b2c1bc68b95066fb
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.2),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/morphex/morphex.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.2)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.2,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Morphex, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'morphex'


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
    robot = Morphex(sim)

    # print information about the robot
    robot.printRobotInfo()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
