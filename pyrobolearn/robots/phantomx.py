#!/usr/bin/env python
"""Provide the Phantom X robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import HexapodRobot


class PhantomX(HexapodRobot):
    r"""Phantom X Hexapod robot

    References:
        [1] https://github.com/HumaRobotics/phantomx_description
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.2),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/phantomx/phantomx.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.2)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.2,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(PhantomX, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'phantomx'


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
    robot = PhantomX(sim)

    # print information about the robot
    robot.printRobotInfo()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
