#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the darts world.
"""

import os
import numpy as np

from pyrobolearn.worlds import BasicWorld


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: finish to implement the world, create corresponding environment (in `envs` folder) with state and reward.

class DartsWorld(BasicWorld):
    r"""Darts world

    """

    def __init__(self, simulator, position=(2.37, 0., 1.73), scale=(1., 1., 1.)):
        """
        Initialize the Dart world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the dart board.
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
        """
        super(DartsWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/darts/'
        position = np.asarray(position)

        # dartboard
        self.dartboard = self.load_mesh(mesh_path + 'dartboard.obj', position=position, scale=scale, mass=0, flags=0)

        # load a table
        table_position = np.array([0., 1., 0.])
        self.table = self.load_table(position=table_position)

        # load darts
        # From wikipedia: "a dart cannot weigh more than 50g including the shaft and flight and cannot exceed a total
        # length of 300mm"
        num_darts = 3
        pos_x = -int(num_darts / 2) * 0.1
        self.darts = []
        for i in range(num_darts):
            dart = self.load_mesh(mesh_path + 'dart.obj',
                                  position=table_position + np.array([pos_x + i*0.1, -0.35, 0.65]),
                                  orientation=(0., 0., -0.707, 0.707), mass=0.020, flags=0)
            self.darts.append(dart)

    def reset(self, world_state=None):
        super(DartsWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(DartsWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = DartsWorld(sim)

    # create manipulator
    robot = world.load_robot('kuka_iiwa')

    # attach first dart to robot end effector
    world.attach(body1=robot, body2=world.darts[0], link1=robot.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.],
                 parent_frame_orientation=[0, 0., 0., 1.])

    # run simulation
    for t in count():
        world.step(sim.dt)
