#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the baseball world.
"""

import os
import numpy as np

from pyrobolearn.worlds import BasicWorld


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: finish to implement the world, create corresponding environment (in `envs` folder) with state and reward.

class BaseballWorld(BasicWorld):
    r"""Baseball world

    """

    def __init__(self, simulator, position=(0., 0., 1.5), scale=(1., 1., 1.)):
        """
        Initialize the baseball world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the baseball bat.
            scale (tuple/list of 3 float): scale of the bat.
        """
        super(BaseballWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/baseball/'
        position = np.asarray(position)

        # load bat
        self.bat = self.load_mesh(mesh_path + 'bat.obj', position=[0., 0., 2.], scale=scale, mass=0.94, flags=0)
        self.bat_grip_radius = 0.035

        # load ball
        self.ball = self.load_mesh(mesh_path + 'ball.obj', position=[0.2, -0.4, 2.], scale=scale, mass=0.145, flags=0)
        self.ball_radius = 0.0375

    def reset(self, world_state=None):
        super(BaseballWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(BaseballWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = BaseballWorld(sim)

    # create manipulator
    robot = world.load_robot('kuka_iiwa')

    # attach bat to robot end effector
    world.attach(body1=robot, body2=world.bat, link1=robot.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., world.bat_grip_radius], child_frame_position=[0., 0.3, 0.],
                 parent_frame_orientation=[0, 0., 0., 1.])

    # apply force to ball to throw it; f=dp/dt thus dp = f dt (change of momentum)

    # run simulation
    for t in count():
       world.step(sim.dt)
