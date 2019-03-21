#!/usr/bin/env python
"""Provide 2d manipulators.
"""

import os
import numpy as np

from pyrobolearn.robots.manipulator import ManipulatorRobot


class Manipulator2D(ManipulatorRobot):
    r"""2D manipulator robot

    References:
        [1] https://github.com/domingoesteban/robolearn_robots_ros
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/manipulator2d/manipulator2d.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Manipulator2D, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'manipulator2d'


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
    robot = Manipulator2D(sim, init_pos=(0, -0.25, 0))
    robot1 = Manipulator2D(sim, init_pos=(0, 0.25, 0))
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        world.step(sleep_dt=1./240)
