#!/usr/bin/env python
"""Provide the Epuck robotic platform.
"""

import os
import numpy as np
from wheeled_robot import DifferentialWheeledRobot


class Epuck(DifferentialWheeledRobot):
    r"""Epuck robot

    References:
        [1] http://www.e-puck.org/
        [2] http://www.gctronic.com/doc/index.php/E-Puck
        [3] https://github.com/gctronic/epuck_driver_cpp
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/epuck/epuck.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.0,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Epuck, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'epuck'

        self.wheels = [self.getLinkIds(link) for link in ['left_wheel', 'right_wheel']
                       if link in self.link_names]
        self.wheel_directions = np.ones(len(self.wheels))


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
    robots = []
    for _ in range(5):
        x, y = np.random.uniform(low=-2, high=2, size=2)
        robot = world.loadRobot(Epuck, position=(x, y, 0))
        robots.append(robot)

    # print information about the robot
    robots[0].printRobotInfo()

    # Position control using sliders
    # robots[0].addJointSlider()

    # run simulator
    for _ in count():
        # robots[0].updateJointSlider()
        for robot in robots:
            robot.drive(5)
        world.step(sleep_dt=1./240)
