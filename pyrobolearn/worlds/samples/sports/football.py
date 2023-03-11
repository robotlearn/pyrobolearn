#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the football/soccer world.
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


class FootballWorld(BasicWorld):
    r"""Football/Soccer world

    """

    def __init__(self, simulator, position=(3., 0., 0.), scale=(1., 1., 1.)):
        """
        Initialize the football/soccer world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the goal.
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
        """
        super(FootballWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/football/'
        position = np.asarray(position)

        # load the goal
        self.goal = self.load_mesh(mesh_path + 'goal.obj', position=position, scale=scale, mass=0, flags=1)

        # load the ball
        self.ball = self.load_sphere(position=[0., 0., 1.], radius=0.11, mass=0.450)
        # self.ball = self.load_mesh(mesh_path + 'ball.obj', position=[0., 0., 1.], scale=scale, mass=0.450, flags=0)

        # set the restitution coefficient for the ball
        # Ref: https://www.physics.hku.hk/~phys0607/lectures/chap05.html
        self.change_dynamics(self.ball, restitution=0.8)
        self.change_dynamics(restitution=1.)

    def reset(self, world_state=None):
        super(FootballWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(FootballWorld, self).step(sleep_dt)


# alias
SoccerWorld = FootballWorld


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    sim = prl.simulators.Bullet()

    world = FootballWorld(sim)

    for t in count():
       world.step(sim.dt)
