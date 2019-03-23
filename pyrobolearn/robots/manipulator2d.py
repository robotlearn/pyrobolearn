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
                 position=(0, 0, 0),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scaling=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/manipulator2d/manipulator2d.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Manipulator2D, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
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
    robot = Manipulator2D(sim, position=(0, -0.25, 0))
    robot1 = Manipulator2D(sim, position=(0, 0.25, 0))
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
