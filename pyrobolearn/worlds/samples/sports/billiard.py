#!/usr/bin/env python
r"""Provide the billiard world.
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

class BilliardWorld(BasicWorld):
    r"""Billiard world

    """

    def __init__(self, simulator, position=(0., 0., 0.5), scale=(1., 1., 1.)):
        """
        Initialize the Billiard world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the billiard table.
            scale (tuple/list of 3 float): scale of the billiard table.
        """
        super(BilliardWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/billiards/'
        position = np.asarray(position)

        # load table
        table = self.load_mesh(mesh_path + 'table_without_cloth.obj', position=position, scale=scale, mass=0,
                               color=(133/255., 94/255., 66/255., 1), flags=1)
        table_cloth = self.load_mesh(mesh_path + 'table_cloth.obj', position=position, scale=scale, mass=0,
                                     color=(0.039, 0.424, 0.012, 1), flags=0)  # color=(0.21, 0.35, 0.29, 1)
        # table = self.load_mesh(mesh_path + 'table.obj', position=position, scale=(1., 1., 1.), mass=0, flags=1)

        # load cue
        self.cue1 = self.load_mesh(mesh_path + 'cue.obj', position=position + np.array([-0.5, 0.4, 0.4]), mass=0.595,
                                   scale=(1., 1., 1.), flags=0, return_body=True)

        # load balls
        # the order is based on: https://www.wikihow.com/Rack-a-Pool-Table
        balls = [1, 9, 2, 10, 8, 3, 11, 7, 14, 4, 5, 13, 15, 6, 12]
        z = 0.785       # height
        r = 0.028575    # radius
        d = 2*r         # diameter
        x, y = 0.6, 0.  # x, y positions
        depth = 0       # depth in the triangle when racking the balls
        b = 0           # use to count the number of ball at a particular level in the triangle

        self.balls = []
        self.removed_balls = []

        # load white ball
        ball = self.load_mesh(mesh_path + 'ball_0.obj', position=(-x, 0, z), mass=0.170, flags=0, return_body=True)
        self.balls.append(ball)

        # load color balls
        for ball_id in balls:
            pos = (x + depth*d, y - depth * (r + 0.001) + b * (d + 0.001 * 2), z)
            ball = self.load_mesh(mesh_path + 'ball_' + str(ball_id) + '.obj', position=pos, mass=0.170, flags=0,
                                  return_body=True)
            b += 1
            if depth == (b-1):
                b = 0
                depth += 1
            self.balls.append(ball)

    def reset(self, world_state=None):
        # reset the billiard
        super(BilliardWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        # check if a ball has entered in a pocket by checking their position

        # if white replace it

        # call the parent step
        super(BilliardWorld, self).step(sleep_dt=sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = BilliardWorld(sim)

    # create manipulator
    robot = world.load_robot('kuka_iiwa', position=[-2., 0.2, 0.])

    # attach cue to robot end effector
    # Note that you can detach the cue from the robot end effector using `world.detach`
    world.attach(body1=robot, body2=world.cue1, link1=robot.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[-0., 0., 0.02], child_frame_position=[0., 0., 0.],
                 parent_frame_orientation=[0, 0., 0., 1.])

    # run simulation
    for t in count():
        world.step(sim.dt)
