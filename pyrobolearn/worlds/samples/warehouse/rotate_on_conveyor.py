#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Regroup small boxes that are on a conveyor.
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

class RotateBoxesOnConveyorWorld(ConveyorWorld):
    r"""Rotate boxes on conveyor belt world

    This provides a world where boxes have to be rotated in the correct orientation and put back on the conveyor belt.
    """

    def __init__(self, simulator, position=(0., -1., 0.5), scale=(1.5, 1., 1.), num_boxes=1):
        """
        Initialize the world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the conveyor.
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.

        """
        super(RotateBoxesOnConveyorWorld, self).__init__(simulator, position=position, scale=scale)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/warehouse/boxes/'

        # load boxes
        # TODO: randomize the dimensions and mass
        # TODO: make the position dependent of the conveyor position and scale
        # self.box = self.load_box(position=self.conveyor_position + np.array([-1.4, 0., 0.5]), mass=1.,
        #                          dimensions=(0.2, 0.2, 0.2), return_body=True)
        self.box = self.load_mesh(mesh_path + 'arrow_box.obj',
                                  position=self.conveyor_position + np.array([-1.4, 0., 0.5]),
                                  orientation=(0., 1., 0., 0.), scale=(1., 1., 1.), mass=1., return_body=True)

        self.cnt = 0

    def reset(self, world_state=None):
        super(RotateBoxesOnConveyorWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        # move boxes (which have not been picked up)
        if self.box.position[0] < 1.4:
            self.apply_force(self.box, force=[2.5, 0., 0.])
        else:
            self.box.position = [-1.4, -1., 1.]
            idx = np.random.randint(low=0, high=4)
            orientation = [(0., 0., 0., 1.), (0., 0.707, 0., 0.707), (0., 1., 0., 0.), (0., -0.707, 0., 0.707)][idx]
            self.box.orientation = orientation
        super(RotateBoxesOnConveyorWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = RotateBoxesOnConveyorWorld(sim)

    # create manipulator
    robot = world.load_robot('kuka_iiwa', position=(0., -0.3, 0.5))

    # run simulation
    for t in count():
        world.step(sim.dt)
