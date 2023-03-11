#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the tennis world.
"""

# TODO: the terrain is not in the correct dimensions; fix it!

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
        self.racket1 = self.load_mesh(mesh_path + 'racket.obj', position=(-6., 0., 1.), scale=scale, mass=0.260,
                                      flags=0, return_body=True)
        self.racket2 = self.load_mesh(mesh_path + 'racket.obj', position=(6., 0., 1.), scale=scale, mass=0.260,
                                      flags=0, return_body=True)

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

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = TennisWorld(sim)

    # Tests before creating environment
    # load 2 robots
    robot1 = world.load_robot('kuka_iiwa', position=[5, 0., 0.2], fixed_base=True)
    robot2 = world.load_robot('kuka_iiwa', position=[-5, 0., 0.2], fixed_base=True)

    # attach each paddle to the robot's end-effector
    world.attach(body1=robot1, body2=world.racket1, link1=robot1.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., 0.03], child_frame_position=[0., 0., -0.3],
                 parent_frame_orientation=[0, -0.707, 0, 0.707])
    world.attach(body1=robot2, body2=world.racket2, link1=robot2.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., 0.03], child_frame_position=[0., 0., -0.3],
                 parent_frame_orientation=[0, 0.707, 0, 0.707])

    # run simulation
    direction = 1
    dy = np.array([0., 0.005, 0.])
    y_lim = 2.
    for t in count():
        # move the robot base (only valid in the simulator)
        x, y, z = robot1.position
        if y > y_lim:
            robot1.position -= dy
            direction = -1
        elif y < -y_lim and direction == -1:
            robot1.position += dy
            direction = 1
        else:
            robot1.position += direction * dy

        # perform a step in the simulation
        world.step(sim.dt)
