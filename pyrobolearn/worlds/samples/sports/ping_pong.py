#!/usr/bin/env python
r"""Provide the ping pong (table tennis) world.
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

class PingPongWorld(BasicWorld):
    r"""Ping Pong world

    """

    def __init__(self, simulator, position=(0., 0., 0.76), scale=(1., 1., 1.)):
        """
        Initialize the ping pong world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the tennis table.
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
        """
        super(PingPongWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/ping_pong/'

        # ping pong
        table = self.load_mesh(mesh_path + 'table.obj', position=position, scale=scale, mass=0,
                               flags=1)

        net_position = np.asarray(position) + np.array([0., 0., 0.01])
        net = self.load_mesh(mesh_path + 'net.obj', position=net_position, scale=scale, mass=0, flags=1)

        self.paddle1 = self.load_mesh(mesh_path + 'paddle.obj', position=(1.1, 0.4, 0.9),
                                      orientation=(0., -0.707, 0., 0.707), scale=scale, mass=0.08,
                                      color=(0.2, 0.2, 0.2, 1), flags=0, return_body=True)
        self.paddle2 = self.load_mesh(mesh_path + 'paddle.obj', position=(-1.1, -0.4, 0.9),
                                      orientation=(0., 0.707, 0., 0.707), scale=scale,
                                      mass=0.08, flags=0, color=(0.8, 0.2, 0.2, 1), return_body=True)

        # load ping pong ball
        # self.ball = self.load_mesh(mesh_path + 'ball.obj', position=(.9, 0.35, 1.2), scale=scale, mass=0.0027,
        #                            flags=0, return_body=True)
        self.ball = self.load_sphere(position=(.9, 0.35, 1.2), mass=0.0027, radius=0.020, color=None, return_body=True)

        # set the restitution coefficient for the ball
        # Ref: "Measure the coefficient of restitution for sports balls", Persson, 2012
        self.ball.restitution = 0.89
        self.change_dynamics(table, restitution=1.)

    def reset(self, world_state=None):
        super(PingPongWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(PingPongWorld, self).step(sleep_dt)


class BallOnPaddleWorld(BasicWorld):
    r"""Ball on a paddle

    """

    def __init__(self, simulator, position=(0., 0., 0.), orientation=(0., 0.707, 0., 0.707), scale=(1., 1., 1.)):
        """
        Initialize the ball on paddle world.

        Args:
            simulator (Simulator): the simulator instance.
            position (tuple/list of 3 float, np.array[3]): position of the tennis table paddle.
            orientation (tuple/list of 4 float, np.array[4]): orientation of the tennis table paddle (expressed as
                a quaternion)
            scale (tuple/list of 3 float): scaling factors in the (x,y,z) directions.
        """
        super(BallOnPaddleWorld, self).__init__(simulator)

        mesh_path = os.path.dirname(os.path.abspath(__file__)) + '/../../meshes/sports/ping_pong/'
        position = np.asarray(position)

        # load paddle
        self.paddle = self.load_mesh(mesh_path + 'paddle.obj', position=position, orientation=orientation, scale=scale,
                                     mass=0.08, flags=0, color=(0.8, 0.2, 0.2, 1), return_body=True)

        # load ping pong ball
        self.ball = self.load_sphere(position=position + np.array([0., 0., 0.4]), mass=0.0027, radius=0.020,
                                     return_body=True)

        # set the restitution coefficient for the ball
        # Ref: "Measure the coefficient of restitution for sports balls", Persson, 2012
        self.ball.restitution = 0.89
        self.paddle.restitution = 1.

    def reset(self, world_state=None):
        super(BallOnPaddleWorld, self).reset(world_state)

    def step(self, sleep_dt=None):
        super(BallOnPaddleWorld, self).step(sleep_dt)


# Test
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # load world
    world = PingPongWorld(sim)

    # Tests before creating environment
    # load 2 robots
    robot1 = world.load_robot('kuka_iiwa', position=[1.8, 0., 0.2], fixed_base=True)
    robot2 = world.load_robot('kuka_iiwa', position=[-1.8, 0., 0.2], fixed_base=True)

    # attach each paddle to the robot's end-effector
    world.attach(body1=robot1, body2=world.paddle1, link1=robot1.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.],
                 parent_frame_orientation=[0, -0.707, 0, 0.707])
    world.attach(body1=robot2, body2=world.paddle2, link1=robot2.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
                 parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.],
                 parent_frame_orientation=[0, 0.707, 0, 0.707])

    # run the simulation
    direction = 1
    dy = np.array([0., 0.005, 0.])
    y_lim = 0.8
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

        # step in the simulation
        world.step(sim.dt)
