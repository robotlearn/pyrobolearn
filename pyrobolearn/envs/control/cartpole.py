#!/usr/bin/env python
"""Provide the inverted pole on a cart (Cartpole) environment.

This is based on the control problem proposed in OpenAI Gym:
"A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is
controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it
from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when
the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center." [1]

Note that compared to [1], you can specify the number of links that forms the inverted pole.

References:
    - [1] Cartpole environment in OpenAI Gym: https://gym.openai.com/envs/CartPole-v1/
    - [2] "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem", Barto et al., 1993.
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


class CartpoleEnv(ControlEnv):
    r"""Cartpole Environment

    This is based on the control problem proposed in OpenAI Gym:
    "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is
    controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it
    from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends
    when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center." [1]

    Note that compared to [1], you can specify the number of links that forms the inverted pole.

    Here are the various environment features (from [1]):

    - world: basic world with gravity enabled, a basic floor and the cartpole.
    - state: the state is given by :math:`[x, \dot{x}, q_1, \dot{q}_1]` for one inverted pole with one link.
    - action: discrete forces applied on the cart (+10., -10.)
    - reward: +1 until termination step
    - initial state generator: initialize uniformly the state with [-0.05, 0.05]
    - physics randomizer: uniform distribution of the mass of the  [mass - mass/10, mass + mass/10]
    - terminal conditions:
        - pole angle is more than 12 degrees
        - cart position is more than 2.5m from the center
        - episode length is greater than 200 steps

    References:
        - [1] Cartpole environment in OpenAI Gym: https://gym.openai.com/envs/CartPole-v1/
        - [2] "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem", Barto et al., 1993.
    """

    def __init__(self, simulator=None, num_links=1, num_steps=200, verbose=True):
        """
        Initialize the Cartpole environment.

        Args:
            simulator (Simulator): simulator instance. If None, by default, it will instantiate the Bullet
                simulator.
            num_links (int): the number of links that forms the inverted pendulum.
            verbose (bool): if True, it will print information when creating the environment.
        """
        # simulator
        if simulator is None:
            simulator = prl.simulators.Bullet(render=verbose)

        # create basic world
        world = prl.worlds.World(simulator)
        robot = prl.robots.CartPole(simulator, position=(0., 0., 0.), num_links=num_links, inverted_pole=False)
        world.load_robot(robot)
        robot.disable_motor(robot.joints)
        if verbose:
            robot.print_info()

        # create state: [x, \dot{x}, q_i, \dot{q}_i]
        state = prl.states.JointPositionState(robot) + prl.states.JointVelocityState(robot)
        if verbose:
            print("\nState: {}".format(state))

        # create action: f_cart = (-10., +10.)
        action = prl.actions.JointForceAction(robot=robot, joint_ids=0, discrete_values=[-10., 10.])
        if verbose:
            print("\nAction: {}".format(action))

        # create terminal condition
        pole_angle_condition = prl.terminal_conditions.JointPositionCondition(robot, joint_ids=1,
                                                                              bounds=(-12 * np.pi/180, 12 * np.pi/180),
                                                                              out=False, stay=True)
        cart_position_condition = prl.terminal_conditions.LinkPositionCondition(robot, link_id=1, bounds=(-1., 1.),
                                                                                dim=0, out=False, stay=True)
        time_length_condition = prl.terminal_conditions.TimeLimitCondition(num_steps=num_steps)
        terminal_conditions = [pole_angle_condition, cart_position_condition, time_length_condition]

        # create reward: +1 until termination step
        reward = prl.rewards.TerminalReward(terminal_conditions=terminal_conditions, subreward=1., final_reward=1.)
        if verbose:
            print("\nReward: {}".format(state))

        # create initial state generator: generate the state each time we reset the environment
        def reset_robot(robot):  # function to disable the motors every time we reset the joint state
            def reset():
                robot.disable_motor(robot.joints)
            return reset

        initial_state_generator = prl.states.generators.UniformStateGenerator(state=state, low=-0.05, high=0.05,
                                                                              fct=reset_robot(robot))

        # create environment using composition
        super(CartpoleEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                          terminal_conditions=terminal_conditions,
                                          initial_state_generators=initial_state_generator)


# class CartDoublePoleEnv(Env):
#     r"""CartDoublepole Environment
#
#     This provide the double inverted poles on a cart environment. Compare to the standard inverted pendulum on a cart,
#     the goal this time is to balance two poles of possibly different lengths / masses, which are initialized at
#     different angles but connected at the same joint attached to the cart.
#     """
#
#     def __init__(self, simulator=None, pole_lengths=(1., 1.), pole_masses=(1., 1.), pole_angles=(0., 0.)):
#         """
#         Initialize the double inverted poles on a cart environment.
#
#         Args:
#             simulator (Simulator): simulator instance.
#         """
#         # create basic world
#         world = prl.worlds.BasicWorld(simulator)
#         robot = prl.robots.CartDoublePole(simulator, pole_lengths=pole_lengths, pole_masses=pole_masses,
#                                           pole_angles=pole_angles)
#         world.load_robot(robot)
#
#         # create state
#         state =
#
#         # create action
#         action =
#
#         # create reward
#         reward =
#
#         # create terminal condition
#         terminal_condition =
#
#         # create initial state generator
#         initial_state_generator =
#
#         # create environment using composition
#         super(CartDoublePoleEnv, self).__init__(world=world, states=state, rewards=reward, actions=action)


# Test
if __name__ == "__main__":
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = CartpoleEnv(sim)

    state = env.reset()
    # run simulation
    for _ in count():
        state, reward, done, info = env.step(sleep_dt=1./240)
        print("done: {}, reward: {}, state: {}".format(done, reward, state))

    # # create basic world
    # sim = prl.simulators.Bullet()
    # world = prl.worlds.World(sim)
    # robot = prl.robots.CartPole(sim, num_links=1, inverted_pole=True)
    # robot.disable_motor(robot.joints)
    # world.load_robot(robot)
    #
    # # create state: [x, \dot{x}, q_i, \dot{q}_i]
    # state = prl.states.JointPositionState(robot) + prl.states.JointVelocityState(robot)
    #
    # # create action
    # action = prl.actions.JointForceAction(robot=robot, joint_ids=0, discrete_values=[-10., 10.])
    #
    # flip = 1
    # for i in prl.count():
    #     # if i % 10 == 0:
    #     #     flip = (flip+1) % 2
    #     action(flip)
    #     world.step(sleep_dt=sim.dt)
