#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the baseball environment.
"""

# TODO: finish to implement this environment

import pyrobolearn as prl
from pyrobolearn.worlds.samples.sports.baseball import BaseballWorld
from pyrobolearn.envs.env import Env


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BaseballEnv(Env):
    r"""Baseball environment
    """

    def __init__(self, simulator, robot='kuka_iiwa', verbose=False):
        """
        Initialize the baseball environment.

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
        world = BaseballWorld(simulator)

        # load manipulator in world
        self.robot = world.load_robot(robot)
        if not isinstance(self.robot, prl.robots.Manipulator):
            raise TypeError("Expecting a manipulator, but got instead {}".format(type(self.robot)))
        if verbose:
            self.robot.print_info()

        # attach bat to robot end effector
        world.attach(body1=self.robot, body2=world.bat, link1=self.robot.end_effectors[0], link2=-1,
                     joint_axis=[0., 0., 0.], parent_frame_position=[0., 0., world.bat_grip_radius],
                     child_frame_position=[0., 0.3, 0.], parent_frame_orientation=[0., 0., 0., 1.])

        # apply force to ball to throw it; f=dp/dt thus dp = f dt (change of momentum)

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

        super(BaseballEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                          terminal_conditions=terminal_condition,
                                          initial_state_generators=initial_state_generator)


# Test
if __name__ == '__main__':
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = BaseballEnv(sim)

    # run simulation
    for t in count():
        env.step(sleep_dt=sim.dt)
