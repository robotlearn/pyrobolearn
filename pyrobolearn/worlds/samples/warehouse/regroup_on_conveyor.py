#!/usr/bin/env python
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

class RegroupBoxesOnConveyorWorld(ConveyorWorld):
    r"""Regroup boxes on conveyor belt world

    This provides a world where small boxes that are randomly distributed on the conveyor belt has to be regroup to
    form a certain pattern (like 16 boxes regrouped in a 4x4 grid).
    """

    def __init__(self, simulator, position=(0., -1., 0.5), scale=(1.5, 1., 1.), num_boxes=1, pattern='3x3grid'):
        """
        Initialize the world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the conveyor.
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
            pattern (str): the pattern that has to be formed when grouping the boxes.
        """
        super(RegroupBoxesOnConveyorWorld, self).__init__(simulator, position=position, scale=scale)

        # load boxes
        # TODO: randomize the dimensions, color, and mass
        # TODO: make the position dependent of the conveyor position and scale
        self.boxes = []
        self.init_box_position = self.conveyor_position + np.array([-1.2, 0., 0.5])
        for i in range(num_boxes):
            noise_x = np.random.uniform(low=-0.2, high=0.2, size=1)
            noise_y = np.random.uniform(low=-0.3, high=0.3, size=1)
            noise = np.concatenate((noise_x, noise_y, np.array([0.])))
            box = self.load_box(position=self.init_box_position + noise, mass=1.,
                                dimensions=(0.05, 0.05, 0.05), return_body=True)
            self.boxes.append(box)

        self.cnt = 0

    def reset(self, world_state=None):
        super(RegroupBoxesOnConveyorWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        # move boxes (which have not been picked up)
        for box in self.boxes:
            if box.position[0] < 1.4:
                self.apply_force(box, force=[2.5, 0., 0.])
            else:
                # TODO: randomize x and y position
                noise_x = np.random.uniform(low=-0.2, high=0.2, size=1)
                noise_y = np.random.uniform(low=-0.3, high=0.3, size=1)
                noise = np.concatenate((noise_x, noise_y, np.array([0.])))
                box.position = self.init_box_position + noise
        super(RegroupBoxesOnConveyorWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = RegroupBoxesOnConveyorWorld(sim, num_boxes=3)

    # create manipulator
    robot = world.load_robot('kuka_iiwa', position=(0., -0.3, 0.5))

    # run simulation
    for t in count():
        world.step(sim.dt)
