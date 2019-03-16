#!/usr/bin/env python
"""Provide the Fetch robotic platform.
"""

import os
from wheeled_robot import WheeledRobot
from manipulator import ManipulatorRobot


class Fetch(WheeledRobot, ManipulatorRobot):
    r"""Fetch robot

    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/fetch/fetch.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.0,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Fetch, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
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
    robot.printRobotInfo()

    # Position control using sliders
    robot.addJointSlider()

    # run simulator
    for _ in count():
        robot.updateJointSlider()
        world.step(sleep_dt=1./240)
