#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the ping pong (table tennis) environment.
"""

# TODO: finish to implement this environment

import pyrobolearn as prl
from pyrobolearn.worlds.samples.sports.ping_pong import PingPongWorld, BallOnPaddleWorld
from pyrobolearn.envs.env import Env


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PingPongEnv(Env):
    r"""Ping Pong environment.

    """

    def __init__(self, simulator, robot='kuka_iiwa', verbose=False):
        """
        Initialize the ping pong environment.

        Args:
            simulator (Simulator, None): simulator instance. If simulator is None, it will use the PyBullet simulator.
            robot (str): robot name.
            verbose (bool): if True, it will print information when creating the environment.
        """
        # check simulator
        if simulator is None:
            simulator = prl.simulators.Bullet()
        elif not isinstance(simulator, prl.simulators.Simulator):
            raise TypeError("Expecting the given 'simulator' to be an instance of `Simulator`, but got instead: "
                            "{}".format(type(simulator)))

        # create world
        world = PingPongWorld(simulator)

        # load 2 manipulators in world
        robot1 = world.load_robot(robot, position=[1.8, 0., 0.2], fixed_base=True)
        robot2 = world.load_robot(robot, position=[-1.8, 0., 0.2], fixed_base=True)
        self.robot1, self.robot2 = robot1, robot2
        if not isinstance(robot1, prl.robots.Manipulator):
            raise TypeError("Expecting a manipulator, but got instead {}".format(type(robot1)))

        # attach each paddle to the robot's end-effector
        world.attach(body1=robot1, body2=world.paddle1, link1=robot1.end_effectors[0], link2=-1,
                     joint_axis=[0., 0., 0.],
                     parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.],
                     parent_frame_orientation=[0, -0.707, 0, 0.707])
        world.attach(body1=robot2, body2=world.paddle2, link1=robot2.end_effectors[0], link2=-1,
                     joint_axis=[0., 0., 0.],
                     parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.],
                     parent_frame_orientation=[0, 0.707, 0, 0.707])

        if verbose:
            self.robot1.print_info()

        # create states
        states = []
        for i, robot in enumerate([robot1, robot2]):
            q_state = prl.states.JointPositionState(robot=robot)
            dq_state = prl.states.JointVelocityState(robot=robot)
            state = q_state + dq_state
            states.append(state)
        if verbose:
            print(states)

        # create actions
        actions = []
        for i, robot in enumerate([robot1, robot2]):
            action = prl.actions.JointPositionAction(robot=robot, kp=robot.kp, kd=robot.kd)
            actions.append(action)
        if verbose:
            print(actions)

        # create reward  # TODO: use geometrical rewards
        reward = None

        # create terminal condition
        terminal_condition = None

        # create initial state generator
        initial_state_generator = None

        super(PingPongEnv, self).__init__(world=world, states=states, rewards=reward, actions=actions,
                                          terminal_conditions=terminal_condition,
                                          initial_state_generators=initial_state_generator)


class BallOnPaddleEnv(Env):
    r"""Ball on a paddle environment.

    """

    def __init__(self, simulator, robot='kuka_iiwa', verbose=False):
        """
        Initialize the ball on paddle environment.

        Args:
            simulator (Simulator, None): simulator instance. If simulator is None, it will use the PyBullet simulator.
            robot (str): robot name.
            verbose (bool): if True, it will print information when creating the environment.
        """
        # check simulator
        if simulator is None:
            simulator = prl.simulators.Bullet()
        elif not isinstance(simulator, prl.simulators.Simulator):
            raise TypeError("Expecting the given 'simulator' to be an instance of `Simulator`, but got instead: "
                            "{}".format(type(simulator)))

        # create world
        world = BallOnPaddleWorld(simulator)

        # load manipulator in world
        self.robot = world.load_robot(robot)
        if not isinstance(self.robot, prl.robots.Manipulator):
            raise TypeError("Expecting a manipulator, but got instead {}".format(type(self.robot)))
        if verbose:
            self.robot.print_info()

        # attach each paddle to the robot's end-effector
        world.attach(body1=self.robot, body2=world.paddle, link1=self.robot.end_effectors[0], link2=-1,
                     joint_axis=[0., 0., 0.], parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.],
                     parent_frame_orientation=[0, 0.707, 0, 0.707])

        world.ball.position = [0., 0., 2.]

        # create state
        q_state = prl.states.JointPositionState(robot=self.robot)
        dq_state = prl.states.JointVelocityState(robot=self.robot)
        state = q_state + dq_state
        if verbose:
            print(state)

        # create action
        action = prl.actions.JointPositionAction(robot=self.robot, kp=self.robot.kp, kd=self.robot.kd)
        if verbose:
            print(action)

        # create reward  # TODO: use geometrical rewards
        reward = None

        # create terminal condition
        terminal_condition = None

        # create initial state generator
        initial_state_generator = None

        super(BallOnPaddleEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                              terminal_conditions=terminal_condition,
                                              initial_state_generators=initial_state_generator)


# Test
if __name__ == '__main__':
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    # env = BallOnPaddleEnv(sim)
    env = PingPongEnv(sim)

    # run the simulation
    for t in count():
        env.step(sleep_dt=sim.dt)
