#!/usr/bin/env python
"""Provide the locomotion with quadruped environment.

This is based on [1,2] but generalized to other quadruped platforms.

References:
    - [1] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
    - [2] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
"""

import pyrobolearn as prl

from pyrobolearn.envs.locomotion.locomotion import LocomotionEnv

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Hwangbo et al.", "Lee et al.", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SelfRightingEnv(LocomotionEnv):
    r"""Self-righting locomotion environment

    This is based on the locomotion environment provided in [1] with the ANYmal robotic platform. As described in [1],
    "the goal is to regain upright base pose from an arbitrary configuration and re-position joints to the sitting
    configuration such that the robot has all feet on the ground for a safe stand-up maneuver".

    - simulator: Raisim
    - world: basic world with gravity enabled, a basic floor and the quadruped robot.
    - state:
        - gravity unit vector (:math:`e_g`) expressed in the base frame (3)
        - base angular velocity in body frame (3)
        - joint position and velocity states (2N)
        - history of joint position error and velocity: current joint state at t (position error + velocity) and two
          past states corresponding to t-0.01s and t-0.02s (6N)
        - previous joint position targets a_{t-1} (N)
        - additive noise for observation
            - up to 0.25 rad/s to the angular velocity
            - up to 0.5 rad/s to the joint velocities
            - up to 0.05 rad to the joint positions
    - action: PD joint position targets :math:`q_d = 0.5 o_t + q_t` where :math:`o_t` is the output of the policy and
      :math:`q_t` are the current joint positions.
    - cost: :math:`0.0005 c_{\tau} + 0.2 c_{jslim} + 0.0025 c_{ad} + 6c_o + 6c_{jp} + 6c_{bi} + 6c_{bs} + 6c_{c,in}`,
      where:
        - torque: :math:`c_{\tau} = || \tau ||^2` where :math:`\tau` are the joint torques.
        - joint speed limit: :math:`c_{jslim} = \sum_{i}^{N} \max(\dot{q}_{i,lim} - |q_i|, 0)^2` where :math:`N` is
          the number of actuated joints, :math:`q_i` is the position of the i-th joint, and :math:`\dot{q}_{i,lim}` is
          the maximum speed of the i-th joint.
        - action difference: :math:`c_{ad} = || a_t - a_{t-1} ||^2` where :math:`a_t` is the action vector.
        - orientation cost: :math:`c_o = || [0,0,-1]^\top - e_g ||` where :math:`e_g` is the unit gravity vector
          expressed in the base frame.
        - joint position: :math:`c_{jp} = \sum_{i}^N K(d(q_i, \hat{q}_i), 2.0)` where :math:`\hat{q}_i` is the desired
          target joint position which correspond in this case to the crouching pose, :math:`d(\cdot, \cdot)` is the
          minimum angle difference which maps to :math:`[0,\pi]`, and
          :math:`K(e, \alpha) = \frac{-1}{e^{\alpha e} + 2 + e^{\alpha e}}` is a kernel function that maps
          :math:`\mathcal{R}` to :math:`[-0.25, 0[`.
        - body impulse: :math:`c_{bi} = \sum_{n \in I_c \backslash I_{c,f}} || i_{c,n} || / (|I_c| - |I_{c,f}|)` where
          :math:`I_c` is the index set of the contact points, :math:`I_{c,f}` is the index set of the foot contact
          points, :math:`i_{c,n}` is the impulse of the `n`th contact.
        - body slippage: :math:`c_{bs} = \sum_{n \in I_c} ||v_{c,n}||^2 / |I_c|` where :math:`v_{c,n}` is the velocity
          of the contact point.
        - self collision: :math:`c_{c,in} = |I_{c,in}|` where :math:`I_{c,in}` is the index set of the self-collision
          points.
    - initial state generator: drop the quadruped from 0.5m about the ground with random joint positions
    - physics randomizer:
        - link masses perturbed up to 10% of the original value
        - the CoM of the base is randomly translated up to 3cm in x,y,z directions
        - the collision geometry of the robot is approximated using collision primitives (box, cylinder, sphere) with
          randomized shapes and positions.
        - the coefficient of friction is sampled from :math:`U([0.8, 2.0])`.
    - terminal condition:
        - time limit of 6sec


    Here are more information about the policy, value function, and algorithm used (with exploration strategy) in the
    paper [1]:

    - policy network: input, 128 (tanh) units, 128 (tanh) units, N output units
    - value network: input, 128 (tanh) units, 128 (tanh) units, 1 output unit
    - exploration in the continuous action space.
    - RL algorithm: TRPO (but also tested PPO)
        - KL divergence threshold (delta) = 0.01
        - GAE: discount factor (gamma) = 0.993, lambda = 0.99
        - for value function: Adam optimizer with learning rate = 0.001
    - curriculum learning: constraining cost terms (power, torque, joint speed, action difference and orientation
      costs) are scaled to 10% of the final value at the first iteration and are scaled up as the training proceeds.

    Note that the authors report that they could train the behavior policy in ~5hours on a single desktop
    machine (32 GB memory, Intel i7-8700K and Geforce GTX 1070) with a fully C++ code.


    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
        - [2] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
    """

    def __init__(self, simulator=None, robot='anymal'):
        """
        Initialize the self-righting environment.

        Args:
            simulator (Simulator, None): simulator instance. If simulator is None, it will use the PyBullet simulator.
            robot (str): robot name.
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

        # check if the robot has the crouching pose as joint configuration.
        if not self.robot.has_joint_configuration('crouching'):
            raise TypeError("Expecting the robot to have the 'crouching' joint configuration predefined.")

        # create state
        state = None

        # create action
        action = None

        # create reward
        reward = None

        # create terminal condition
        terminal_condition = None  # prl.terminal_conditions.TimeLimitCondition(time=6)

        # create initial state generator
        initial_state_generator = None

        # create environment using composition
        super(SelfRightingEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                              terminal_conditions=terminal_condition,
                                              initial_state_generators=initial_state_generator)


class StandingUpEnv(LocomotionEnv):
    r"""Standing-up locomotion environment

    This is based on the locomotion environment provided in [1] with the ANYmal robotic platform. As described in [1],
    the goal is to stand-up from an up-right position such that the robot is ready for the next phase (i.e. locomotion).

    - simulator: Raisim
    - world: basic world with gravity enabled, a basic floor and the quadruped robot.
    - state:
        - gravity unit vector (:math:`e_g`) expressed in the base frame (3)
        - base angular velocity in body frame (3)
        - base linear velocity in body frame (3)
        - joint position and velocity states (2N)
        - history of joint position error and velocity: current joint state at t (position error + velocity) and two
          past states corresponding to t-0.01s and t-0.02s (6N)
        - previous joint position targets a_{t-1} (N)
        - additive noise for observation
            - up to 0.2 m/s to the linear velocity
            - up to 0.25 rad/s to the angular velocity
            - up to 0.5 rad/s to the joint velocities
            - up to 0.05 rad to the joint positions
    - action: PD joint position targets :math:`q_d = 0.5 o_t + q_t` where :math:`o_t` is the output of the policy and
      :math:`q_t` are the current joint positions.
    - cost: :math:`0.0001 c_{\tau} + 0.6 c_{jslim} + 0.001 c_{ad} + 2.5 c_o + 5 c_h + 3 c_{jp}`, where:
        - torque: :math:`c_{\tau} = || \tau ||^2` where :math:`\tau` are the joint torques.
        - joint speed limit: :math:`c_{jslim} = \sum_{i}^{N} \max(\dot{q}_{i,lim} - |q_i|, 0)^2` where :math:`N` is
          the number of actuated joints, :math:`q_i` is the position of the i-th joint, and :math:`\dot{q}_{i,lim}` is
          the maximum speed of the i-th joint.
        - action difference: :math:`c_{ad} = || a_t - a_{t-1} ||^2` where :math:`a_t` is the action vector.
        - orientation cost: :math:`c_o = || [0,0,-1]^\top - e_g ||` where :math:`e_g` is the unit gravity vector
          expressed in the base frame.
        - height: :math:`c_h = 1.0` if base height < threshold, otherwise 0.
        - joint position: :math:`c_{jp} = \sum_{i}^N K(d(q_i, \hat{q}_i), 2.0)` where :math:`\hat{q}_i` is the desired
          target joint position which correspond in this case to the crouching pose, :math:`d(\cdot, \cdot)` is the
          minimum angle difference which maps to :math:`[0,\pi]`, and
          :math:`K(e, \alpha) = \frac{-1}{e^{\alpha e} + 2 + e^{\alpha e}}` is a kernel function that maps
          :math:`\mathcal{R}` to :math:`[-0.25, 0[`.
    - initial state generator: drop the quadruped from 0.5m about the ground with near-upright pose.
    - physics randomizer:
        - link masses perturbed up to 10% of the original value
        - the CoM of the base is randomly translated up to 3cm in x,y,z directions
        - the collision geometry of the robot is approximated using collision primitives (box, cylinder, sphere) with
          randomized shapes and positions.
        - the coefficient of friction is sampled from :math:`U([0.8, 2.0])`.
    - terminal condition:
        - time limit of 6sec


    Here are more information about the policy, value function, and algorithm used (with exploration strategy) in the
    paper [1]:

    - policy network: input, 128 (tanh) units, 128 (tanh) units, N output units
    - value network: input, 128 (tanh) units, 128 (tanh) units, 1 output unit
    - exploration in the continuous action space.
    - RL algorithm: TRPO (but also tested PPO)
        - KL divergence threshold (delta) = 0.01
        - GAE: discount factor (gamma) = 0.993, lambda = 0.99
        - for value function: Adam optimizer with learning rate = 0.001
    - curriculum learning: constraining cost terms (power, torque, joint speed, action difference and orientation
      costs) are scaled to 10% of the final value at the first iteration and are scaled up as the training proceeds.

    Note that the authors report that they could train the behavior policy in ~5hours on a single desktop
    machine (32 GB memory, Intel i7-8700K and Geforce GTX 1070) with a fully C++ code.


    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
        - [2] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
    """

    def __init__(self, simulator=None, robot='anymal'):
        """
        Initialize the standing-up environment.

        Args:
            simulator (Simulator, None): simulator instance. If simulator is None, it will use the PyBullet simulator.
            robot (str): robot name.
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

        # check if the robot has the crouching pose as joint configuration.
        if not self.robot.has_joint_configuration('standing'):
            raise TypeError("Expecting the robot to have the 'standing' joint configuration predefined.")

        # create state
        state = None

        # create action
        action = None

        # create reward
        reward = None

        # create terminal condition
        terminal_condition = None  # prl.terminal_conditions.TimeLimitCondition(time=6)

        # create initial state generator
        initial_state_generator = None

        # create environment using composition
        super(StandingUpEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                            terminal_conditions=terminal_condition,
                                            initial_state_generators=initial_state_generator)


class CommandedLocomotionEnv(LocomotionEnv):
    r"""Locomotion quadruped environment

    This is based on the locomotion environment provided in [1] with the ANYmal robotic platform. As described in [1],
    "the goal is for the robot to follow a given velocity command composed of desired forward velocity, lateral
    velocity, and yaw rate".

    - simulator: Raisim
    - world: basic world with gravity enabled, a basic floor and the quadruped robot.
    - state:
        - desired velocity commands (forward velocity, lateral velocity, yaw rate) (3)
        - estimated base height (h_e) (1)
        - gravity unit vector (:math:`e_g`) expressed in the base frame (3)
        - base angular velocity in body frame (3)
        - base linear velocity in body frame (3)
        - joint position and velocity states (2N)
        - history of joint position error and velocity: current joint state at t (position error + velocity) and two
          past states corresponding to t-0.01s and t-0.02s (6N)
        - previous joint position targets a_{t-1} (N)
        - additive noise for observation
            - up to 0.2 m/s to the linear velocity
            - up to 0.25 rad/s to the angular velocity
            - up to 0.5 rad/s to the joint velocities
            - up to 0.05 rad to the joint positions
    - action: PD joint position targets :math:`q_d = 0.5 o_t + q_n` where :math:`o_t` is the output of the policy and
      :math:`q_n` is the standing joint configuration.
    - cost: :math:`0.0005 c_{\tau} + 0.03 c_{jslim} + 0.5c_{ad} + 0.4c_o + 6c_\omega + 10 c_v + 0.1 c_{fc} + 2 c_{fs}`,
      where:
        - torque: :math:`c_{\tau} = || \tau ||^2` where :math:`\tau` are the joint torques.
        - joint speed limit: :math:`c_{jslim} = \sum_{i}^{N} \max(\dot{q}_{i,lim} - |q_i|, 0)^2` where :math:`N` is
          the number of actuated joints, :math:`q_i` is the position of the i-th joint, and :math:`\dot{q}_{i,lim}` is
          the maximum speed of the i-th joint.
        - action difference: :math:`c_{ad} = || a_t - a_{t-1} ||^2` where :math:`a_t` is the action vector.
        - orientation cost: :math:`c_o = || [0,0,-1]^\top - e_g ||` where :math:`e_g` is the unit gravity vector
          expressed in the base frame.
        - angular velocity: :math:`c_\omega = K(|\omega^B_B - \hat{\omega}^B_B|, 1.0)`, where :math:`\omega^B_B` is the
          angular velocity of the base expressed in the body frame, :math:`\hat{\omega}` is the desired angular
          velocity, and :math:`K(e, \alpha) = \frac{-1}{e^{\alpha e} + 2 + e^{\alpha e}}` is a kernel function that
          maps :math:`\mathcal{R}` to :math:`[-0.25, 0[`.
        - linear velocity: :math:`c_v = K(|v^B_B - \hat{v}^B_B|, 4.0)`, where :math:`v^B_B` is the linear velocity of
          the base expressed in the body frame and math:`\hat{v}` is the desired linear velocity.
        - foot clearance: :math:`c_{fc} = \sum (h_{f,i} - 0.07)^2 ||v_{f,i}||,  \forall i s.t. g_i > 0, i \in I_{c,f}`,
          where :math:`h_{f,i}` is the ze position of the `i`th foot, :math:`v_{f,i}` is the velocity of the `i`th foot,
          :math:`g_i` is the gap function of the `i`th contact, and :math:`I_{c,f}` is the index set of the foot
          contact points.
        - foot slippage: :math:`c_{fs} = \sum ||v_{f,i}||,  \forall i s.t. g_i=0, i \in I_{c,f}`
    - initial state generator:
        - sample the desired forward velocity, lateral velocity and yaw rate from U(-1, 1) m/s, U(-0.4, 0.4) m/s and
          U(-1.2, 1.2) rad/s respectively. Note that this depends on the joystick/game controller that is being used.
        - the initial joint states are sampled from a MVN centered at the standing configuration.
    - physics randomizer:
        - link masses perturbed up to 10% of the original value
        - the CoM of the base is randomly translated up to 3cm in x,y,z directions
        - the collision geometry of the robot is approximated using collision primitives (box, cylinder, sphere) with
          randomized shapes and positions.
        - the coefficient of friction is sampled from :math:`U([0.8, 2.0])`.
    - terminal condition:
        - time limit of 4sec
        - joint limit with terminal cost of 1.0
        - falling (base touching the ground) with the cost of 1.0


    Here are more information about the policy, value function, and algorithm used (with exploration strategy) in the
    paper [1]:

    - policy network: input, 128 (tanh) units, 256 (tanh) units, N output units
    - value network: input, 128 (tanh) units, 256 (tanh) units, 1 output unit
    - exploration in the continuous action space.
    - RL algorithm: TRPO (but also tested PPO)
        - KL divergence threshold (delta) = 0.01
        - GAE: discount factor (gamma) = 0.995, lambda = 0.99
        - for value function: Adam optimizer with learning rate = 0.001
    - curriculum learning: constraining cost terms (power, torque, joint speed, action difference and orientation
      costs) are scaled to 10% of the final value at the first iteration and are scaled up as the training proceeds.

    Note that the authors report that they could train the behavior policy in ~5hours on a single desktop
    machine (32 GB memory, Intel i7-8700K and Geforce GTX 1070) with a fully C++ code.


    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
        - [2] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
    """

    def __init__(self, simulator=None, robot='anymal'):
        """
        Initialize the standing-up environment.

        Args:
            simulator (Simulator, None): simulator instance. If simulator is None, it will use the PyBullet simulator.
            robot (str): robot name.
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

        # check if the robot has the crouching pose as joint configuration.
        if not self.robot.has_joint_configuration('standing'):
            raise TypeError("Expecting the robot to have the 'standing' joint configuration predefined.")

        # create state
        state = None

        # create action
        action = None

        # create reward
        reward = None

        # create terminal condition
        terminal_condition = None  # prl.terminal_conditions.TimeLimitCondition(time=6)

        # create initial state generator
        initial_state_generator = None

        # create environment using composition
        super(CommandedLocomotionEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                     terminal_conditions=terminal_condition,
                                                     initial_state_generators=initial_state_generator)


class BehaviorLocomotionEnv(LocomotionEnv):
    r"""Behavior Locomotion Environment

    This is based on the locomotion environment provided in [1] with the ANYmal robotic platform. As described in [1],
    "the behavior selector has to choose an appropriate behavior such that the robot returns to a nominal operating
    state (i.e. states where it can locomote) every time it loses balance." 
    
    Practically, this environment uses the following previously defined environments `SelfRightingEnv`,
    `StandingUpEnv`, and `CommandedLocomotionEnv`.

    - simulator: Raisim
    - world: basic world with gravity enabled, a basic floor and the quadruped robot.
    - state:
        - previous discrete action (represented as a real one-hot vector) (3)
        - desired velocity commands (forward velocity, lateral velocity, yaw rate) (3)
        - estimated base height (h_e) (1)
        - gravity unit vector (:math:`e_g`) expressed in the base frame (3)
        - base angular velocity in body frame (3)
        - base linear velocity in body frame (3)
        - joint position and velocity states (2N)
        - history of joint position error and velocity: current joint state at t (position error + velocity) and two
          past states corresponding to t-0.01s and t-0.02s (6N)
        - previous joint position targets a_{t-1} (N)
        - additive noise for observation
            - up to 0.2 m/s to the linear velocity
            - up to 0.25 rad/s to the angular velocity
            - up to 0.5 rad/s to the joint velocities
            - up to 0.05 rad to the joint positions
    - action: discrete action :math:`a \in \{0, 1, 2\}` represented as a real 3D vector :math:`[p_0, p_1, p_2]` (i.e.
      the vector outputted by the policy).
    - cost: :math:`0.001 c_{pw} + 0.05 c_{\tau} + 0.05 c_{jslim} + 0.05 c_{ad} + 0.5c_o + 10 c_\omega + 10 c_v + 3c_h`,
      where:
        - power: math:`c_{pw} = \sum_i^N \max(\dot{q}_i \tau_i, 0)`, where :math:`N` is the number of actuated joints,
          :math:`\dot{q}_i` and :math:`\tau_i` are the velocity and torque (respectively) of the `i`th joint.
        - torque: :math:`c_{\tau} = || \tau ||^2`, where :math:`\tau` are the joint torques.
        - joint speed limit: :math:`c_{jslim} = \sum_{i}^{N} \max(\dot{q}_{i,lim} - |q_i|, 0)^2`, where :math:`N` is
          the number of actuated joints, :math:`q_i` is the position of the i-th joint, and :math:`\dot{q}_{i,lim}` is
          the maximum speed of the i-th joint.
        - action difference: :math:`c_{ad} = || a_t - a_{t-1} ||^2`, where :math:`a_t` is the action vector.
        - orientation cost: :math:`c_o = || [0,0,-1]^\top - e_g ||`, where :math:`e_g` is the unit gravity vector
          expressed in the base frame.
        - angular velocity: :math:`c_\omega = K(|\omega^B_B - \hat{\omega}^B_B|, 1.0)`, where :math:`\omega^B_B` is the
          angular velocity of the base expressed in the body frame, :math:`\hat{\omega}` is the desired angular
          velocity, and :math:`K(e, \alpha) = \frac{-1}{e^{\alpha e} + 2 + e^{\alpha e}}` is a kernel function that
          maps :math:`\mathcal{R}` to :math:`[-0.25, 0[`.
        - linear velocity: :math:`c_v = K(|v^B_B - \hat{v}^B_B|, 4.0)`, where :math:`v^B_B` is the linear velocity of
          the base expressed in the body frame and math:`\hat{v}` is the desired linear velocity.
        - height: :math:`c_h = 1.0` if base height < threshold, otherwise 0, where the threshold depends on the average
          base height of the robot (or its maximum possible height).
    - initial state generator:
        - sample from the initial state distributions of a randomly selected behavior {self-righting, standing-up,
          locomotion}.
    - physics randomizer:
        - link masses perturbed up to 10% of the original value
        - the CoM of the base is randomly translated up to 3cm in x,y,z directions
        - the collision geometry of the robot is approximated using collision primitives (box, cylinder, sphere) with
          randomized shapes and positions.
        - the coefficient of friction is sampled from :math:`U([0.8, 2.0])`.
    - terminal condition:
        - time limit of 12sec


    Here are more information about the policy, value function, and algorithm used (with exploration strategy) in the
    paper [1]:

    - policy network: input, 128 (tanh) units, 3 output units (softmax)
    - value network: input, 128 (tanh) units, 1 output unit
    - exploration in the discrete action space.
    - RL algorithm: TRPO (but also tested PPO)
        - KL divergence threshold (delta) = 0.01
        - GAE: discount factor (gamma) = 0.99, lambda = 0.99
        - for value function: Adam optimizer with learning rate = 0.001

    Note that the authors report that they could train the behavior policy in ~5hours on a single desktop
    machine (32 GB memory, Intel i7-8700K and Geforce GTX 1070) with a fully C++ code.


    References:
        - [1] "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning", Lee et al., 2019
        - [2] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
    """

    def __init__(self, simulator=None, robot='anymal'):
        """
        Initialize the locomotion with quadruped environment.

        Args:
            simulator (Simulator, None): simulator instance. If simulator is None, it will use the PyBullet simulator.
            robot (str): robot name.
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

        # create state
        state = None

        # create action
        action = None

        # create reward
        reward = None

        # create terminal condition
        terminal_condition = None

        # create initial state generator
        initial_state_generator = None

        # create environment using composition
        super(BehaviorLocomotionEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                    terminal_conditions=terminal_condition,
                                                    initial_state_generators=initial_state_generator)


class AgileLocomotionEnv(LocomotionEnv):
    r"""Agile locomotion environment.

    This is based on the locomotion environment provided in [1] with the ANYmal robotic platform, where they introduce
    the actuator net.

    References:
        - [1] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
    """

    def __init__(self, simulator=None, robot='anymal'):
        """
        Initialize the locomotion with quadruped environment.

        Args:
            simulator (Simulator, None): simulator instance. If simulator is None, it will use the PyBullet simulator.
            robot (str): robot name.
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

        # create state
        state = None

        # create action
        action = None

        # create reward
        reward = None

        # create terminal condition
        terminal_condition = None

        # create initial state generator
        initial_state_generator = None

        # create environment using composition
        super(AgileLocomotionEnv, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                 terminal_conditions=terminal_condition,
                                                 initial_state_generators=initial_state_generator)


# Test
if __name__ == "__main__":
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # # create environment
    # env = RobustLocomotionQuadrupedEnv(sim)
    #
    # # run simulation
    # for _ in count():
    #     env.step(sleep_dt=1. / 240)
