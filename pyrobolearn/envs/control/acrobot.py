#!/usr/bin/env python
"""Provide the acrobot environment.

This is based on the control problem proposed in OpenAI Gym:
"The acrobot system includes two joints and two links, where the joint between the two links is actuated. Initially,
the links are hanging downwards, and the goal is to swing the end of the lower link up to a given height." [1]

References:
    - [1] Acrobot environment in OpenAI Gym: https://gym.openai.com/envs/Acrobot-v1/
    - [2] "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding", Sutton, 1996.
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


class AcrobotEnv(ControlEnv):
    r"""Acrobot Environment

    This is based on the control problem proposed in OpenAI Gym [1]:
    "The acrobot system includes two joints and two links, where the joint between the two links is actuated.
    Initially, the links are hanging downwards, and the goal is to swing the end of the lower link up to a given
    height." [1]

    Here are the various environment features:

    - world: basic world with gravity enabled, a basic floor and the acrobot.
    - state: the state is given by :math:`[cos(q_1), sin(q_1), cos(q_2), sin(q_2), \dot{q}_1, \dot{q}_2]`
    - action: discrete joint torques :math:`\tau_2 \in \{-1., 0., +1.\}`
    - reward: -1 if not terminal
    - initial state generator: initialize uniformly the joint position and velocity states between [-0.1, 0.1]
    - physics randomizer: uniform distribution of the mass of the  [mass - mass/10, mass + mass/10]
    - terminal condition: if the end-effector link is above a certain height.

    References:
        - [1] Acrobot environment in OpenAI Gym: https://gym.openai.com/envs/Acrobot-v1/
        - [2] "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding", Sutton, 1996.
    """

    def __init__(self, simulator=None, use_reward_shaping=False, verbose=False):
        """
        Initialize the acrobot environment.

        Args:
            simulator (Simulator): simulator instance.
            use_reward_shaping (bool): if True, it will use a reward that guides how to achieve the goal.
            verbose (bool): if True, it will print information when creating the environment.
        """
        # create basic world
        world = prl.worlds.BasicWorld(simulator)
        robot = world.load_robot('acrobot')
        robot.disable_motor()
        if verbose:
            robot.print_info()

        # create state: [cos(q_1), sin(q_1), cos(q_2), sin(q_2), \dot{q}_1, \dot{q}_2]
        trig_position_state = prl.states.JointTrigonometricPositionState(robot=robot)
        velocity_state = prl.states.JointVelocityState(robot=robot)
        state = trig_position_state + velocity_state
        if verbose:
            print("\nObservation: {}".format(state))

        # create action: \tau_2 in {0., -1., +1.}
        action = prl.actions.JointTorqueAction(robot, joint_ids=robot.joints[-1],
                                               discrete_values=np.array([0., -1., +1.]))
        if verbose:
            print("\nAction: {}".format(action))

        # create terminal condition:
        terminal_condition = prl.terminal_conditions.LinkPositionCondition(robot, link_id=robot.joints[-1],
                                                                           bounds=(2.5, np.infty), dim=2,
                                                                           out=True, stay=False)

        # create reward: -1 if not terminal
        if use_reward_shaping:  # use continuous reward
            # distance_cost = prl.rewards.DistanceCost()
            # orientation_cost = prl.rewards.OrientationCost()
            position_cost = prl.rewards.JointPositionCost(prl.states.JointPositionState(robot),
                                                          target_state=np.zeros(len(robot.joints)),
                                                          update_state=True)
            velocity_cost = prl.rewards.JointVelocityCost(velocity_state)
            torque_cost = prl.rewards.JointTorqueCost(prl.states.JointForceTorqueState(robot=robot), update_state=True)
            # reward = distance_cost + orientation_cost + 0.1 * velocity_cost + 0.01 * torque_cost
            reward = position_cost + 0.1 * velocity_cost + 0.001 * torque_cost
        else:  # use discrete reward
            reward = prl.rewards.TerminalReward(terminal_condition, subreward=-1., final_reward=0.)

        # create initial state generator: generate the state each time we reset the environment
        def reset_robot(robot):  # function to disable the motors every time we reset the joint state
            def reset():
                robot.disable_motor()
            return reset

        init_state = prl.states.JointPositionState(robot) + velocity_state
        num_joints = len(robot.joints)
        low = [[np.pi + 0.1] + [0]*(num_joints-1), [-0.1]*num_joints]
        high = [[np.pi - 0.1] + [0]*(num_joints-1), [0.1]*num_joints]
        initial_state_generator = prl.states.generators.UniformStateGenerator(state=init_state, low=low, high=high,
                                                                              fct=reset_robot(robot))

        # create physics randomizer: randomize the mass each time we reset the environment
        masses = robot.get_link_masses(link_ids=robot.joints)
        masses = (masses - masses / 10., masses + masses / 10.)
        physics_randomizer = prl.physics.LinkPhysicsRandomizer(robot, link_ids=robot.joints, masses=masses)

        # create environment using composition
        super(AcrobotEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                         initial_state_generators=initial_state_generator,
                                         physics_randomizers=physics_randomizer, terminal_conditions=terminal_condition)


# Test
if __name__ == "__main__":
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = AcrobotEnv(sim, verbose=True)

    # run simulation
    env.reset()
    for _ in count():
        env.step(sleep_dt=1./240)
