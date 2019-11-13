#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the flip pancake world.
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

class FlipPancakeWorld(BasicWorld):
    r"""Flip pancake world

    """

    def __init__(self, simulator, position=(0., 0., 0.), scale=(.75, .75, .75)):
        """
        Initialize the world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the pan.
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
        """
        super(FlipPancakeWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/manipulation/pancake/'
        position = np.asarray(position)

        # load the pan
        self.pan = self.load_mesh(mesh_path + 'pan.obj', position=position, scale=scale, mass=0.120, flags=1)

        # load the pancake
        self.pancake = self.load_mesh(mesh_path + 'pancake.obj', position=position + np.array([0., 0., 0.1]),
                                      scale=scale, mass=0.080, flags=0)

    def reset(self, world_state=None):
        super(FlipPancakeWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(FlipPancakeWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = FlipPancakeWorld(sim, position=(0.5, 0., 0.))

    # # create manipulator
    # robot = world.load_robot('kuka_iiwa')
    #
    # # attach first shinai to robot end effector
    # world.attach(body1=robot, body2=world.pan, link1=robot.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
    #              parent_frame_position=[0., 0., 0.2], child_frame_position=[-0.05, 0., 0.],
    #              parent_frame_orientation=[0, -0.707, 0., .707])

    # run simulation
    for t in count():
        world.step(sim.dt)
