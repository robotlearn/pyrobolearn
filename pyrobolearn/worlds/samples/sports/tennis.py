#!/usr/bin/env python
r"""Provide the tennis world.
"""

# TODO: the terrain is not in the correct dimensions; fix it!

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


class TennisWorld(BasicWorld):
    r"""Tennis world

    """

    def __init__(self, simulator, position=(0., 0., 0.), scale=(1., 1., 1.)):
        super(TennisWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/tennis/'
        position = np.asarray(position)

        # load terrain
        terrain = self.load_mesh(mesh_path + 'court_without_net.obj', position=position, scale=scale, mass=0, flags=1)
        net = self.load_mesh(mesh_path + 'net.obj', position=position, scale=scale, mass=0, flags=1)

        # load rackets
        self.racket1 = self.load_mesh(mesh_path + 'racket.obj', position=(-6., 0., 1.), scale=scale, mass=0, flags=0,
                                      return_body=True)
        self.racket2 = self.load_mesh(mesh_path + 'racket.obj', position=(6., 0., 1.), scale=scale, mass=0, flags=0,
                                      return_body=True)

        # load ball
        self.ball = self.load_mesh(mesh_path + 'ball.obj', position=(2, 0., 1.), scale=scale, mass=0.0585, flags=0,
                                   return_body=True)

        # set the restitution coefficient for the ball
        # Ref: "Measure the coefficient of restitution for sports balls", Persson, 2012
        self.change_dynamics(self.ball, restitution=0.87)
        self.change_dynamics(restitution=1.)

    def reset(self, world_state=None):
        super(TennisWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(TennisWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    sim = prl.simulators.Bullet()

    world = TennisWorld(sim)

    for t in count():
        world.step(sim.dt)
