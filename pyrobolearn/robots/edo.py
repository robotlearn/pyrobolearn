#!/usr/bin/env python
"""Provide the e.Do robotic platform.
"""

import os
from manipulator import ManipulatorRobot


class Edo(ManipulatorRobot):
    r"""Edo robot

    E.Do robot developed by Comau.

    References:
        [1] e.Do: https://edo.cloud/
        [2] Comau: https://www.comau.com/EN/our-competences/robotics/eDO
        [3] Github: https://github.com/Comau/eDO_description
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=True,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/edo/edo.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(Edo, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'edo'


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
    robot = Edo(sim)

    # print information about the robot
    robot.printRobotInfo()
    # H = robot.calculateMassMatrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
