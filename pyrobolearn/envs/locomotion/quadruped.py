#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the locomotion with quadruped environment.

This is based on [1] and [2] but generalized to other quadruped platforms.

References:
    - [1] PyBullet:
      https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur_gym_env.py
    - [2] RaisimGym: https://github.com/leggedrobotics/raisimGym/blob/master/raisim_gym/env/env/ANYmal/Environment.hpp
"""

import numpy as np

import pyrobolearn as prl
from pyrobolearn.envs.locomotion.locomotion import LocomotionEnv


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Erwin Coumans (Pybullet)", "Jemin Hwangbo et al. (RaisimGym)", "Brian Delhaisse (PRL)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LocomotionQuadrupedBulletEnv(LocomotionEnv):
    r"""Locomotion Quadruped Bullet Environment

    This is based on the locomotion environment provided for the minitaur robot in PyBullet [1] but generalized to
    other quadruped robotic platforms.

    Here are the various environment features:

    - world: basic world with gravity enabled, a basic floor and the quadruped robot.
    - state:
        - joint positions (N)
        - joint velocities (N)
        - joint torques (N)
        - base orientation as quaternion (4)
    - action: PD joint position targets (or joint torques)
    - reward: reward = 1.0 * r_f + 0. * c_d + 0. * c_s + 0.005 c_e
        - forward reward: :math:`r_f = x_t - x_{t-1}` where :math:`x` is the base x-position.
        - drift cost: :math:`c_d = - |y_t - y_{t-1}|` where :math:`y` is the base y-position.
        - shake cost: :math:`c_s = - |z_t - z_{t-1}|` where :math:`z` is the base z-position.
        - energy cost: :math:`c_e = -|\tau * dq| * dt` where :math:`\tau` are the torques, :math:`dq` are the joint
          velocities, and :math:`dt` is the simulation time step.
    - initial state generator:
        - reset base position and orientation to initial position / orientation
        - reset base velocity: [0,0,0,0,0,0]
        - reset joint positions to initial joint positions
        - reset joint velocities to 0
    - physics randomizer:
        - additive base mass noise: U([-0.2, 0.2]) kg
        - additive leg mass noise: U([-0.2, 0.2]) kg
        - the coefficient of friction for the feet is sampled from :math:`U([0.8, 1.5])`.
    - terminal condition:
        - fallen:
            - orientation: :math:`a_z \cdot [0,0,1] < a` where :math:`a_z` is the z-axis of the base, and :math:`a` is
              the angle threshold (0.85).
            - height: :math:`z < h` where :math:`h` is the height threshold.
        - distance limit: :math:`\sqrt{x^2 + y^2} > \text{threshold}`  where :math:`threshold` is set to inf.

    More information:
    - inner control_loop = 5.

    References:
        - [1] PyBullet:
          https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur_gym_env.py
    """

    def __init__(self, simulator=None, robot='minitaur', verbose=False):
        """
        Initialize the locomotion with quadruped environment.

        Args:
            simulator (Simulator, None): simulator instance. If None, by default, it will instantiate the Bullet
                simulator.
            robot (str): robot name.
            verbose (bool): if True, it will print information when creating the environment.
        """
        # create simulator if necessary
        if simulator is None:
            simulator = prl.simulators.Bullet(render=verbose)

        # create basic world
        world = prl.worlds.BasicWorld(simulator)

        # load robot in world
        self.robot = world.load_robot(robot)
        if not isinstance(self.robot, prl.robots.LeggedRobot):  # prl.robots.QuadrupedRobot
            raise TypeError("Expecting a legged robot, but got instead {}".format(type(self.robot)))
        if verbose:
            self.robot.print_info()

        # create state
        q_state = prl.states.JointPositionState(robot=self.robot)
        dq_state = prl.states.JointVelocityState(robot=self.robot)
        tau_state = prl.states.JointForceTorqueState(robot=self.robot)
        quat_state = prl.states.BaseOrientationState(robot=self.robot)
        state = q_state + dq_state + tau_state + quat_state
        if verbose:
            print(state)

        # create action
        action = prl.actions.JointPositionAction(robot=self.robot, kp=self.robot.kp, kd=self.robot.kd)
        if verbose:
            print(action)

        # create terminal condition
        orientation_condition = prl.terminal_conditions.BaseOrientationAxisCondition(self.robot, angle=0.85,
                                                                                     axis=(0., 0., 1.), dim=2,
                                                                                     stay=True, out=False)
        height_condition = prl.terminal_conditions.BaseHeightCondition(self.robot, height=self.robot.base_height/8.,
                                                                       stay=True, out=True)
        distance_condition = prl.terminal_conditions.DistanceCondition(self.robot, distance=float("inf"),
                                                                       dim=[1, 1, 0], stay=True, out=False)
        terminal_condition = [orientation_condition, height_condition, distance_condition]
        if verbose:
            print("Terminal condition: {}".format(terminal_condition))

        # create reward
        forward_reward = prl.rewards.ForwardProgressReward(self.robot, direction=(1., 0., 0.))
        base_position_state = prl.states.BasePositionState(self.robot)
        drift_cost = prl.rewards.DriftCost(base_position_state, update_state=True)  # y component
        shake_cost = prl.rewards.ShakeCost(base_position_state)  # z component
        energy_cost = prl.rewards.JointEnergyCost(self.robot, dt=simulator.dt)
        reward = 1. * forward_reward + 0.005 * energy_cost + 0. * shake_cost + 0. * drift_cost
        if verbose:
            print(reward)

        # create initial state generator
        base_pose_gen = prl.states.generators.FixedStateGenerator(state=prl.states.BasePoseState(self.robot))
        base_vel_gen = prl.states.generators.FixedStateGenerator(state=prl.states.BaseLinearVelocityState(self.robot))
        q_init = self.robot.get_joint_configurations('home') if self.robot.has_joint_configuration('home') else \
            np.zeros(len(self.robot.joints))
        q_gen = prl.states.generators.FixedStateGenerator(state=q_state, data=q_init)
        dq_gen = prl.states.generators.FixedStateGenerator(state=dq_state, data=np.zeros(len(self.robot.joints)))

        initial_state_generator = [base_pose_gen, base_vel_gen, q_gen, dq_gen]
        if verbose:
            print("Initial state generator: {}".format(initial_state_generator))

        # create environment using composition
        super(LocomotionQuadrupedBulletEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                           terminal_conditions=terminal_condition,
                                                           initial_state_generators=initial_state_generator)


class LocomotionQuadrupedRaisimEnv(LocomotionEnv):
    r"""Locomotion Quadruped Raisim Environment

    This is based on the locomotion environment provided in `raisimGym` for the ANYmal robot in [1]. The Python version
    can be found in `raisimpy` in [2].

    Here are the various environment features:

    - simulator: Raisim
    - world: basic world with gravity enabled, a basic floor and the quadruped robot.
    - state:
        - height (1D)
        - world frame z-axis expressed in the body frame (3D)
        - joint angle positions (ND)
        - joint velocities (ND)
        - body linear velocities (3D)
        - body angular velocities (3D)
    - action: PD joint position targets
    - reward: 0.3 * v_x - 2e-5 * ||\tau||^2
        - if terminal, -10 is added to the reward.
    - initial state generator: fixed state generator for joint positions such that they are set to the home position.
    - terminal condition: if there is contact with a link that is not the foot.

    More information:
    - inner control_loop = int(control_dt / simulation_dt) where control_dt=0.01 and simulation_dt=0.001.

    References:
        - [1] RaisimGym:
          https://github.com/leggedrobotics/raisimGym/blob/master/raisim_gym/env/env/ANYmal/Environment.hpp
        - [2] Raisimpy: https://github.com/robotlearn/raisimpy/blob/master/examples/raisimpy_gym/envs/anymal/env.py
    """

    def __init__(self, simulator=None, robot='anymal', verbose=False):
        """
        Initialize the locomotion quadruped environment.

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

        # create basic world
        world = prl.worlds.BasicWorld(simulator)

        # load robot in world
        self.robot = world.load_robot(robot)
        if not isinstance(self.robot, prl.robots.LeggedRobot):  # prl.robots.QuadrupedRobot
            raise TypeError("Expecting a legged robot, but got instead {}".format(type(self.robot)))
        if verbose:
            self.robot.print_info()

        # create state
        height_state = prl.states.BaseHeightState(robot=self.robot)
        z_axis_state = prl.states.BaseAxisState(robot=self.robot, base_axis=2)  # z-axis
        q_state = prl.states.JointPositionState(robot=self.robot)
        dq_state = prl.states.JointVelocityState(robot=self.robot)
        lin_vel_state = prl.states.BaseLinearVelocityState(robot=self.robot)
        ang_vel_state = prl.states.BaseAngularVelocityState(robot=self.robot)
        state = height_state + z_axis_state + q_state + dq_state + lin_vel_state + ang_vel_state
        if verbose:
            print(state)

        # create action
        action = prl.actions.JointPositionAction(robot=self.robot, kp=self.robot.kp, kd=self.robot.kd)
        if verbose:
            print(action)

        # create terminal condition (all links that are not feet must stay out of contact)
        terminal_condition = prl.terminal_conditions.ContactCondition(robot=self.robot, link_ids=self.robot.feet,
                                                                      all=True, stay=True, out=True, complement=True)
        if verbose:
            print("Terminal condition: {}".format(terminal_condition))

        # create reward
        vel_reward = prl.rewards.BaseLinearVelocityReward(state=lin_vel_state, axis=0)
        torque_cost = prl.rewards.JointTorqueCost(state=self.robot)
        terminal_reward = prl.rewards.TerminalReward(terminal_conditions=terminal_condition, final_reward=-10.)
        reward = 0.3 * vel_reward + 2e-5 * torque_cost + terminal_reward
        if verbose:
            print(reward)

        # create initial state generator
        q_init = self.robot.get_joint_configurations('home') if self.robot.has_joint_configuration('home') else \
            np.zeros(len(self.robot.joints))
        initial_state_generator = prl.states.generators.FixedStateGenerator(state=q_state, data=q_init)
        if verbose:
            print("Initial state generator: {}".format(initial_state_generator))

        super(LocomotionQuadrupedRaisimEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                           terminal_conditions=terminal_condition,
                                                           initial_state_generators=initial_state_generator)


# Test
if __name__ == "__main__":
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = LocomotionQuadrupedRaisimEnv(sim, verbose=True)
    # env = LocomotionQuadrupedBulletEnv(sim, verbose=True)

    # run simulation
    for _ in count():
        obs, reward, done, info = env.step(sleep_dt=1./240)
        # print("obs: {}".format(obs))
        print("reward: {}".format(reward))
        print("done: {}".format(done))
        print("info: {}".format(info))
        if done:
            print("End")
            break
