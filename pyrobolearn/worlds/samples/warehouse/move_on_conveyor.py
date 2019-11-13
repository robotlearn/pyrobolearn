#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Move boxes from one conveyor belt to another one.
"""

import os
import numpy as np

from pyrobolearn.worlds.samples.warehouse.conveyor import ConveyorWorld


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: finish to implement the world, create corresponding environment (in `envs` folder) with state and reward.

class MoveBoxesOnConveyorWorld(ConveyorWorld):
    r"""Move boxes on conveyor belt world

    This provides a world where one box has to be moved from one conveyor to another.
    """

    def __init__(self, simulator, position=(0., -1., 0.5), scale=(1.5, 1., 1.), num_boxes=1):
        """
        Initialize the world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the conveyor.
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
            num_boxes (int): the number of boxes to load
        """
        super(MoveBoxesOnConveyorWorld, self).__init__(simulator, position=position, scale=scale)

        # load 2 other conveyors
        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/warehouse/conveyor/'
        position, orientation = (0.8, 0.5, 0.5), (0., 0., 0.707, 0.707)
        self.conveyor1 = self.load_mesh(mesh_path + 'conveyor_without_belt.obj', position=position,
                                        orientation=orientation, scale=(1., 1., 1.), mass=0., flags=0)
        self.belt1 = self.load_mesh(mesh_path + 'belt.obj', position=position, orientation=orientation,
                                    scale=(1., 1., 1.), mass=0., flags=0)

        position, orientation = (-0.8, 0.5, 0.5), (0., 0., 0.707, 0.707)
        self.conveyor2 = self.load_mesh(mesh_path + 'conveyor_without_belt.obj', position=position,
                                        orientation=orientation, scale=(1., 1., 1.), mass=0., flags=0)
        self.belt2 = self.load_mesh(mesh_path + 'belt.obj', position=position, orientation=orientation,
                                    scale=(1., 1., 1.), mass=0., flags=0)

        # load boxes
        # TODO: randomize the dimensions, color, and mass
        # TODO: make the position dependent of the conveyor position and scale
        self.box = self.load_box(position=self.conveyor_position + np.array([-1.4, 0., 0.5]), mass=1.,
                                 dimensions=(0.2, 0.2, 0.2), return_body=True)

        self.cnt = 0

    def reset(self, world_state=None):
        super(MoveBoxesOnConveyorWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        # move boxes (which have not been picked up)
        if self.box.position[0] < 1.4:
            self.apply_force(self.box, force=[2.5, 0., 0.])
        else:
            self.box.position = [-1.4, -1., 1.]
        super(MoveBoxesOnConveyorWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = MoveBoxesOnConveyorWorld(sim)

    # create manipulator
    robot = world.load_robot('kuka_iiwa', position=(0., -0.3, 0.5))

    # run simulation
    for t in count():
        world.step(sim.dt)
