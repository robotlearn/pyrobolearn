#!/usr/bin/env python
r"""Provide the ping pong (table tennis) self.
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


class PingPongWorld(BasicWorld):
    r"""Ping Pong world

    """

    def __init__(self, simulator, table_position=(0., 0., 0.76), dimensions=(1., 1., 1.)):
        super(PingPongWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/pingpong/'

        # ping pong
        table = self.load_mesh(mesh_path + 'table.obj', position=table_position, scale=dimensions, mass=0,
                               flags=1)
        self.change_dynamics(table, restitution=0.3)

        net_position = np.asarray(table_position) + np.array([0., 0., 0.01])
        net = self.load_mesh(mesh_path + 'net.obj', position=net_position, scale=dimensions, mass=0, flags=1)

        paddle1 = self.load_mesh(mesh_path + 'paddle.obj', position=(1.1, 0.4, 0.9), scale=dimensions, mass=0.08,
                                 flags=0)
        paddle2 = self.load_mesh(mesh_path + 'paddle.obj', position=(-1.1, -0.4, 0.9), orientation=(0., -1., 0., 0.),
                                 scale=dimensions, mass=0.08, flags=0, return_body=False)
        ball = self.load_sphere(position=(1., 0.35, 1.2), mass=0.0027, radius=0.020, color=None, return_body=True)
        ball.restitution = 2.5


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    sim = prl.simulators.Bullet()

    world = PingPongWorld(sim)

    for t in count():
        world.step(sim.dt)
