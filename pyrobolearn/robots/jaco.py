#!/usr/bin/env python
"""Provide the Jaco robotic platform.
"""

import os
from manipulator import ManipulatorRobot


class Jaco(ManipulatorRobot):
    r"""Jaco (manipulator) robot

    References:
        [1] https://github.com/JenniferBuehler/jaco-arm-pkgs
        [2] https://github.com/Kinovarobotics/kinova-ros
        [3] https://github.com/RIVeR-Lab/wpi_jaco
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=True,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/jaco/jaco.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(Jaco, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'jaco'


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
    robot = Jaco(sim)

    # print information about the robot
    robot.printRobotInfo()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
