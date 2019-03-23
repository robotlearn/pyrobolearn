#!/usr/bin/env python
"""Provide the Swimmer Mujoco model.
"""

import os

from pyrobolearn.robots.robot import Robot


class Swimmer(Robot):
    r"""Swimmer Mujoco Model
    """

    def __init__(self,
                 simulator,
                 position=(-0.5, 0, 0.1),
                 orientation=(0, 0.707, 0, 0.707),
                 fixed_base=False,
                 scaling=1.,
                 urdf=os.path.dirname(__file__) + '/mjcfs/swimmer.xml'):
        # check parameters
        if position is None:
            position = (-0.5, 0., 0.1)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.1,)
        if orientation is None:
            orientation = (0, 0.707, 0, 0.707)
        if fixed_base is None:
            fixed_base = False

        super(Swimmer, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
        self.name = 'swimmer'


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
    robot = Swimmer(sim)

    # print information about the robot
    robot.print_info()

    # run simulation
    for i in count():
        # step in simulation
        world.step(sleep_dt=1./240)
