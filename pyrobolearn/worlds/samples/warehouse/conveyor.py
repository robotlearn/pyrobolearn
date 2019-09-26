#!/usr/bin/env python
r"""Provide the conveyor world.

This is used by other worlds in the `warehouse` folder.
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

class ConveyorWorld(BasicWorld):
    r"""Conveyor world

    """

    def __init__(self, simulator, position=(0., -1., 0.5), scale=(1.5, 1., 1.)):
        """
        Initialize the world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the conveyor.
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
        """
        super(ConveyorWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/warehouse/conveyor/'
        self.conveyor_position = np.asarray(position)
        self.belt_length = 2 * scale[0]  # rough approximation of the length
        self.belt_width = 0.6 * scale[1]  # rough approximation of the width

        # load the conveyor
        self.conveyor = self.load_mesh(mesh_path + 'conveyor_without_belt.obj', position=position, scale=scale, 
                                       mass=0., flags=0)
        self.belt = self.load_mesh(mesh_path + 'belt.obj', position=position, scale=scale, mass=0., flags=0)

    def reset(self, world_state=None):
        super(ConveyorWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(ConveyorWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = ConveyorWorld(sim)

    # run simulation
    for t in count():
        world.step(sim.dt)
