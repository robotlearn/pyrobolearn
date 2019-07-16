#!/usr/bin/env python
"""Provide the inverted pendulum swing-up environment.

This is based on the control problem proposed in OpenAI Gym:
"The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem,
the pendulum starts in a random position, and the goal is to swing it up so it stays upright." [1]

References:
    - [1] Pendulum environment in OpenAI Gym: https://gym.openai.com/envs/Pendulum-v0/
"""

import os
import numpy as np

import pyrobolearn as prl
from pyrobolearn.envs.control.control import ControlEnv


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["OpenAI", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class InvertedPendulumSwingUpEnv(ControlEnv):
    r"""Inverted Pendulum Swing-up Environment

    This is based on the control problem proposed in OpenAI Gym:
    "The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the
    problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright." [1]

    Here are the various environment features:

    - world: basic world with gravity, a basic floor, and the pendulum.
    - state: the state is given by :math:`[np.cos(q_1), np.sin(q_1), dq_1]`
    - action: the action is the joint torque :math:`\tau_1`
    - reward:

    References:
        - [1] Pendulum environment in OpenAI Gym: https://gym.openai.com/envs/Pendulum-v0/
    """

    def __init__(self, simulator=None, verbose=False):
        """
        Initialize the inverted pendulum swing-up environment.

        Args:
            simulator (Simulator): simulator instance.
            verbose (bool): if True, it will print information when creating the environment
        """
        # create basic world
        world = prl.worlds.BasicWorld(simulator)
        robot = world.load_robot('pendulum')
        robot.disable_motor()
        # robot.print_info()

        # create state
        position_state = prl.states.JointTrigonometricPositionState(robot=robot)
        velocity_state = prl.states.JointVelocityState(robot=robot)
        state = position_state + velocity_state

        # create action
        action = prl.actions.JointTorqueAction(robot, f_min=-2., f_max=2.)

        # create reward
        position_cost = prl.rewards.JointPositionCost(prl.states.JointPositionState(robot),
                                                      np.zeros(len(robot.joints)))
        velocity_cost = prl.rewards.JointVelocityCost(velocity_state)
        torque_cost = prl.rewards.JointTorqueCost(prl.states.JointForceTorqueState(robot=robot))
        reward = position_cost + 0.1 * velocity_cost + 0.001 * torque_cost

        # create initial state generator
        initial_state_generator = None

        # create environment using composition
        super(InvertedPendulumSwingUpEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                         initial_state_generators=initial_state_generator)


# Test
if __name__ == "__main__":
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = InvertedPendulumSwingUpEnv(sim)

    # run simulation
    for _ in count():
        env.step(sleep_dt=1./240)
