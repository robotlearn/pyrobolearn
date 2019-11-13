#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the kendo world.
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

class KendoWorld(BasicWorld):
    r"""Kendo world

    """

    def __init__(self, simulator, position=(0., 0., 1.), scale=(1., 1., 1.), num_shinai=1):
        """
        Initialize the kendo world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the shinai (i.e. kendo stick).
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
            num_shinai (int): the number of shinai (i.e. kendo stick).
        """
        super(KendoWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/kendo/'
        position = np.asarray(position)

        # load the shinai
        self.shinai = []
        for i in range(num_shinai):
            shinai = self.load_mesh(mesh_path + 'shinai.obj', position=position, scale=scale, mass=0.480, flags=0)
            position += np.array([1., 0., 0.])
            self.shinai.append(shinai)

    def reset(self, world_state=None):
        super(KendoWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(KendoWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = KendoWorld(sim, position=(0., 0., 1.5), num_shinai=2)

    # create manipulators
    robot1 = world.load_robot('kuka_iiwa')
    robot2 = world.load_robot('kuka_iiwa', position=(1., 0.), orientation=(0., 0., 1., 0.))

    # attach shinai to robot end effectors
    world.attach(body1=robot1, body2=world.shinai[0], link1=robot1.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.15],
                 parent_frame_orientation=[0, -0.707, 0., .707])
    world.attach(body1=robot2, body2=world.shinai[1], link1=robot2.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.15],
                 parent_frame_orientation=[0, -0.707, 0., .707])

    # run simulation
    for t in count():
        world.step(sim.dt)
