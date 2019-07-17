#!/usr/bin/env python
"""Provide the inverted pendulum swing-up environment.

This is based on the control problem proposed in OpenAI Gym:
"The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem,
the pendulum starts in a random position, and the goal is to swing it up so it stays upright." [1]

References:
    - [1] Pendulum environment in OpenAI Gym: https://gym.openai.com/envs/Pendulum-v0/
"""

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

    This is based on the control problem proposed in OpenAI Gym [1]:
    "The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the
    problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright." [1]

    Here are the various environment features:

    - world: basic world with gravity, a basic floor, and the pendulum loaded at the center.
    - state: the state is given by :math:`[cos(q_1), sin(q_1), \dot{q}_1]`
    - action: the action is the joint torque :math:`\tau_1`
    - cost: :math:`||d(q,q_{target})||^2 + 0.1 * ||\dot{q}||^2 + 0.001 * ||\tau||^2`, where :math:`d(\cdot, \cdot)`
      is the minimum angle difference between two angles.
    - initial state generator: initialize the joint angle between [-pi, pi] (q=0 when the pendulum is pointing up)
    - physics randomizer: uniform distribution of the mass of the pendulum [mass - mass/10, mass + mass/10]

    References:
        - [1] Pendulum environment in OpenAI Gym: https://gym.openai.com/envs/Pendulum-v0/
    """

    def __init__(self, simulator, verbose=False):
        """
        Initialize the inverted pendulum swing-up environment.

        Args:
            simulator (Simulator): simulator instance.
            verbose (bool): if True, it will print information when creating the environment
        """
        # create basic world with the robot
        world = prl.worlds.BasicWorld(simulator)
        robot = world.load_robot('pendulum')
        robot.disable_motor()
        if verbose:
            robot.print_info()

        # create state: [cos(q_1), sin(q_1), \dot{q}_1]
        trig_position_state = prl.states.JointTrigonometricPositionState(robot=robot)
        velocity_state = prl.states.JointVelocityState(robot=robot)
        state = trig_position_state + velocity_state
        if verbose:
            print("\nObservation: {}".format(state))

        # create action: \tau_1
        action = prl.actions.JointTorqueAction(robot, bounds=(-2., 2.))
        if verbose:
            print("\nAction: {}".format(action))

        # create reward/cost: ||d(q,q_{target})||^2 + 0.1 * ||\dot{q}||^2 + 0.001 * ||\tau||^2
        position_cost = prl.rewards.JointPositionCost(prl.states.JointPositionState(robot),
                                                      target_state=np.zeros(len(robot.joints)),
                                                      update_state=True)
        velocity_cost = prl.rewards.JointVelocityCost(velocity_state)
        torque_cost = prl.rewards.JointTorqueCost(prl.states.JointForceTorqueState(robot=robot), update_state=True)
        reward = position_cost + 0.1 * velocity_cost + 0.001 * torque_cost
        if verbose:
            print("Reward: {}".format(reward))

        # create initial state generator: generate the state each time we reset the environment
        def reset_robot(robot):  # function to disable the motors every time we reset the joint state
            def reset():
                robot.disable_motor()
            return reset

        init_state = prl.states.JointPositionState(robot)
        low, high = np.array([-np.pi] * len(robot.joints)), np.array([np.pi] * len(robot.joints))
        # init_state.data = np.array([np.pi / 2])  # initial data
        # initial_state_generator = prl.states.generators.FixedStateGenerator(state=init_state, fct=reset_robot(robot))
        initial_state_generator = prl.states.generators.UniformStateGenerator(state=init_state, low=low, high=high,
                                                                              fct=reset_robot(robot))

        # create physics randomizer: randomize the mass each time we reset the environment
        masses = robot.get_link_masses(link_ids=robot.joints)
        masses = (masses - masses/10., masses + masses/10.)
        physics_randomizer = prl.physics.LinkPhysicsRandomizer(robot, link_ids=robot.joints, masses=masses)

        # could create terminal conditions (if necessary) such as:
        # - success if we stay at the upper position for more than 20 steps
        # - failure if we can achieve the goal after 10,000 steps
        # In the OpenAI gym, there are no terminal conditions for this problem
        terminal_condition = None

        # create environment using composition
        super(InvertedPendulumSwingUpEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                         initial_state_generators=initial_state_generator,
                                                         physics_randomizers=physics_randomizer,
                                                         terminal_conditions=terminal_condition)


# Test
if __name__ == "__main__":

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = InvertedPendulumSwingUpEnv(sim, verbose=True)

    # run simulation
    env.reset()
    for t in prl.count():
        # if (t % 800) == 0:
        #     env.reset()  # test reset function
        states, rewards, done, info = env.step(sleep_dt=1./240)
        # print("State: {}".format(states))
        print("Reward: {}".format(rewards))
