#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the billiard environment.
"""

# TODO: finish to implement this environment

import pyrobolearn as prl
from pyrobolearn.worlds.samples.sports.billiard import BilliardWorld
from pyrobolearn.envs.env import Env


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BilliardEnv(Env):
    r"""Billiard environment

    """

    def __init__(self, simulator, robot='kuka_iiwa', verbose=False):
        """
        Initialize the billiard environment.

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
        world = BilliardWorld(simulator)

        # load manipulator in world
        self.robot = world.load_robot(robot, position=[-2., 0.2, 0.])
        if not isinstance(self.robot, prl.robots.Manipulator):
            raise TypeError("Expecting a manipulator, but got instead {}".format(type(self.robot)))
        if verbose:
            self.robot.print_info()

        # attach cue to robot end effector
        # Note that you can detach the cue from the robot end effector using `world.detach`
        world.attach(body1=self.robot, body2=world.cue1, link1=self.robot.end_effectors[0], link2=-1,
                     joint_axis=[0., 0., 0.], parent_frame_position=[-0., 0., 0.02], child_frame_position=[0., 0., 0.],
                     parent_frame_orientation=[0., 0., 0., 1.])

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

        super(BilliardEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                          terminal_conditions=terminal_condition,
                                          initial_state_generators=initial_state_generator)


# Test
if __name__ == '__main__':
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    env = BilliardEnv(sim)

    # run simulation
    for t in count():
        env.step(sleep_dt=sim.dt)
