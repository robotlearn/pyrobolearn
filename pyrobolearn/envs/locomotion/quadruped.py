#!/usr/bin/env python
"""Provide the locomotion with quadruped environment.

This is based on [1] and [2] but generalized to other quadruped platforms.

References:
    - [1] PyBullet:
      https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur_gym_env.py
    - [2] RaisimGym: https://github.com/leggedrobotics/raisimGym/blob/master/raisim_gym/env/env/ANYmal/Environment.hpp
"""

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


class LocomotionQuadrupedEnv1(LocomotionEnv):
    r"""Locomotion Quadruped Environment

    This is based on the locomotion environment provided for the minitaur robot in PyBullet [1] but generalized to
    other quadruped robotic platforms.

    Here are the various environment features:

    - world: basic world with gravity enabled, a basic floor and the quadruped robot.
    - state:
    - action:
    - reward:
    - initial state generator:
    - physics randomizer:
    - terminal condition:

    References:
        - [1] PyBullet:
          https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur_gym_env.py
    """

    def __init__(self, simulator=None, robot='minitaur'):
        """
        Initialize the locomotion with quadruped environment.

        Args:
            simulator (Simulator, None): simulator instance.
            robot (str): robot name.
        """
        # create basic world
        world = prl.worlds.BasicWorld(simulator)
        robot = world.load_robot(robot)

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
        super(LocomotionQuadrupedEnv1, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                      terminal_conditions=terminal_condition,
                                                      initial_state_generators=initial_state_generator)


class LocomotionQuadrupedEnv2(LocomotionEnv):
    r"""Locomotion Quadruped Environment

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

    References:
        - [1] RaisimGym:
          https://github.com/leggedrobotics/raisimGym/blob/master/raisim_gym/env/env/ANYmal/Environment.hpp
        - [2] Raisimpy: https://github.com/robotlearn/raisimpy/blob/master/examples/raisimpy_gym/envs/anymal/env.py
    """

    def __init__(self, simulator=None, robot='anymal'):
        """
        Initialize the locomotion quadruped environment.

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

        super(LocomotionQuadrupedEnv2, self).__init__(world=world, states=state, rewards=reward, actions=action,
                                                      terminal_conditions=terminal_condition,
                                                      initial_state_generators=initial_state_generator)


# Test
if __name__ == "__main__":
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = LocomotionQuadrupedEnv1(sim)

    # run simulation
    for _ in count():
        env.step(sleep_dt=1./240)
