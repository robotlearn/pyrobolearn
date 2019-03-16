#!/usr/bin/env python
"""Provide the Kuka LWR robotic platform.
"""

import os
from manipulator import ManipulatorRobot


class KukaLWR(ManipulatorRobot):
    r"""Kuka LWR robot

    LWR stands for 'Light Weight Robot'. This robot has 7 DoFs, and an ATI F/T sensor at the end-effector.
    Payload of 7kg and a range of 790mm.

    References:
        [1] Kuka robotics: https://www.kuka.com/en-de
        [2] https://github.com/CentroEPiaggio/kuka-lwr
        [3] https://github.com/bulletphysics/bullet3/tree/master/data/kuka_lwr
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=True,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/kuka/kuka_lwr/kuka.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(KukaLWR, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'kuka_lwr'


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
    robot = KukaLWR(sim)

    # print information about the robot
    robot.printRobotInfo()
    # H = robot.calculateMassMatrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
