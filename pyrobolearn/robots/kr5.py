#!/usr/bin/env python
"""Provide the Kuka KR5 robotic industrial platform.
"""

import os
from manipulator import ManipulatorRobot


class KR5(ManipulatorRobot):
    r"""Kuka KR5 sixx R650 robot

    Payload of 5.00kg and a reach of 650mm or 850mm.

    References:
        [1] Kuka robotics: https://www.kuka.com/en-de
        [2] https://github.com/a-price/KR5sixxR650WP_description
        [3] https://github.com/ros-industrial/kuka_experimental
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=True,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/kuka/kr5/kr5.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(KR5, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'kr5'


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
    robot = KR5(sim)

    # print information about the robot
    robot.printRobotInfo()
    # H = robot.calculateMassMatrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
