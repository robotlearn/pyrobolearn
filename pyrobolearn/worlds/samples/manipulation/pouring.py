# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the pouring world.
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

class PouringWorld(BasicWorld):
    r"""Pouring world

    This loads a table, a bottle, and a glass in basic world. The liquid might be simulated based on the simulator.
    """

    def __init__(self, simulator, table_position=(0.8, 0., 0.), bottle_position=(0.6, -0.2, .6),
                 glass_position=(0.6, 0.2, .6), scale=(1., 1., 1.)):
        """
        Initialize the world.

        Args:
            simulator (Simulator): the simulator instance.
            table_position (tuple/list of 3 float, np.array[3]): position of the table.
            bottle_position (tuple/list of 3 float, np.array[3]): position of the bottle.
            glass_position (tuple/list of 3 float, np.array[3]): position of the glass.
            scale (tuple/list of 3 float): scale of the object.
        """
        super(PouringWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/manipulation/glasses/'

        # load table
        self.table = self.load_table(position=table_position, orientation=(0., 0., 0.707, 0.707))

        # load bottle
        self.bottle = self.load_mesh(mesh_path + 'bottle1.obj', position=bottle_position, scale=scale, mass=0.500,
                                     color=(0, 0.416, 0.306, 0.5))

        # load glass
        self.glass = self.load_mesh(mesh_path + 'ikea_glass.obj', position=glass_position, scale=scale, mass=0.250,
                                    color=(0.84705882, 0.89411765, 0.91372549, 0.5))

        # load water (this depends on the simulator)
        # TODO

    def reset(self, world_state=None):
        super(PouringWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(PouringWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = PouringWorld(sim)

    # create manipulator
    robot = world.load_robot('kuka_iiwa', position=(0., 0., 0.2))

    # attach bottle to manipulator end-effector
    world.attach(body1=robot, body2=world.bottle, link1=robot.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., 0.05], child_frame_position=[0., 0., 0.1],
                 parent_frame_orientation=[0, -0.707, 0., .707])

    # reset the joint position
    robot.reset_joint_states(q=[-0.07892626, 0.25348269, -0.23774851, -1.89899545, 0.08092009, -0.57060496, 0.04279614])

    # run simulation
    for t in count():
        world.step(sim.dt)
