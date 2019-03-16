#!/usr/bin/env python
"""Provide the Walker2D Mujoco model.
"""

import os
from robot import Robot


class Walker2D(Robot):
    r"""Walker 2D Mujoco Model
    """

    def __init__(self,
                 simulator,
                 init_pos=(-0.5, 0, 0.1),
                 init_orient=(0, 0.707, 0, 0.707),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/mjcfs/walker2d.xml'):
        # check parameters
        if init_pos is None:
            init_pos = (-0.5, 0., 0.1)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.1,)
        if init_orient is None:
            init_orient = (0, 0.707, 0, 0.707)
        if useFixedBase is None:
            useFixedBase = False

        super(Walker2D, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'walker2D'


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
    robot = Walker2D(sim)

    # print information about the robot
    robot.printRobotInfo()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
