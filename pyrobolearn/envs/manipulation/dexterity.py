#!/usr/bin/env python
"""Provide the manipulation dexterity environment defined in [1].

Reference:
    - [1] "Learning Dexterous In-Hand Manipulation", OpenAI et al., 2018 (https://arxiv.org/abs/1808.00177)
"""

import pyrobolearn as prl
from pyrobolearn.envs.env import Env


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["OpenAI (Paper)", "Brian Delhaisse (PRL code)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DexterityEnv(Env):
    r"""Manipulation Dexterity Environment

    This is based on the environment presented in [1] by OpenAI. The following

    Here are the various environment features:

    - simulator: MuJoCo
    - world: basic world with gravity enabled, a basic floor, the robotic hand(s), and the cube (with letters drew on
    it).
        - robotic hand: shadowhand (by default), softhand, allegrohand, schunk_hand
    - states:
        - for value network:
            - fingertip positions (5*3D)
            - object position (3D)
            - object orientation (4D=quaternion)
            - target orientation (4D=quaternion)
            - relative target orientation (4D=quaternion)
            - hand joint angles (24D)
            - hand joint velocities (24D)
            - object velocity (3D)
            - object angular velocity (3D)
        - for policy:
            - fingertip positions (5*3D)
            - object position (3D)
            - relative target orientation (4D=quaternion)
    - actions: desired joint angles of the hand relative to the current ones. The actions are discretized into 11 bins.
    - reward function:
        - `r_t = d_t - d_{t+1}`, where `d_t` and `d_{t+1}` are the rotation angles between the desired and current
          object orientations before and after the transition, respectively.
        - 5 if the goal is achieved
        - -20 if the object drop
    - terminal condition:
        - the goal is achieved
        - the object drop
    - domain randomization
        - Gaussian noise to policy observations
            - cor
        - physics randomization:
            - object dimensions: U([0.95, 1.05])
            - object and robot link masses: U([0.5, 1.5])
            - surface friction coefficients: U([0.7, 1.3])
            - robot joint damping coefficients: U([0.3, 3.0])
            - actuator force gains (P term): \log U([0.75, 1.5])
            - additive joint limits noise: N(0, 0.15) rad
            - additive gravity vector noise (each coordinate): N(0, 0.4) m/s^2
        - visual appearance randomization
            - camera positions
            - camera intrinsics
            - lighting conditions
            - pose of the hand and object
            - materials and textures for all objects in the scene (including the hand)

    Here are more information about the policy, value function, and algorithm used (with exploration strategy) in the
    paper [1]:

    - policy network: fully-connected neural network composed of a normalization layer, dense ReLU (1024), LSTM (512)
    - value network: fully-connected neural network composed of a normalization layer, dense ReLU (1024), LSTM (512)
    - vision pose estimation network:
        - Input: 3 RGB image of size 200x200x3
        - Conv2D: 32 filters, 5x5 kernel size, stride 1, no padding
        - Conv2D: 32 filters, 3x3 kernel size, stride 1, no padding
        - Max pooling: 3x3 kernel size, stride 3
        - ResNet: 1 block, 16 filters, 3x3 kernel size, stride 3
        - ResNet: 2 blocks, 32 filters, 3x3 kernel size, stride 3
        - ResNet: 2 blocks, 64 filters, 3x3 kernel size, stride 3
        - ResNet: 2 blocks, 64 filters, 3x3 kernel size, stride 3
        - Spatial Softmax
        - Flatten
        - Concatenate
        - Fully-connected: 128 units
        - Fully-connected: output dimensions (3 for position and 4 for orientation (quaternion))
    - exploration: in the action space using a categorical distribution with 11 bins for each action coordinate
    - RL algorithm: PPO
        - clip parameter = 0.2
        - entropy regularization coefficient = 0.01
        - GAE: discount factor (gamma) = 0.998, lambda = 0.95
        - optimizer: Adam with learning rate = 3e-4
        - batch size: 80k chunks x 10 transitions = 800k transitions
        - minibatch size: 25.6k transitions
        - number of minibatches per step: 60
    - SL algorithm for the vision network
        - optimizer: Adam with learning rate = 5e-4 (halved every 20,000 batches)
        - minibatch size: 64x3 = 192 RGB images
        - weight decay regularization: 0.001
        - number of training batches: 400,000

    Reference:
        - [1] "Learning Dexterous In-Hand Manipulation", OpenAI et al., 2018 (https://arxiv.org/abs/1808.00177)
    """

    def __init__(self, simulator, hand='shadowhand', num_hands=1, with_camera=False, verbose=False):
        """
        Initialize the manipulation dexterity environment.

        Args:
            simulator (Simulator): simulator instance.
            hand (str):
            num_hands (int):
            verbose (bool): if True, it will print information when creating the environment
            with_camera (bool): if True, it will add the cameras that are presented in the paper at the same positions.
        """

        # create world
        world = prl.worlds.BasicWorld(simulator)

        # load robotic hand
        if not isinstance(hand, str):
           raise TypeError("Expecting a string specifying which hand we want to load in the world, but instead got: "
                           "{}".format(type(hand)))
        if hand[-4:] != 'hand':  # 'shadowhand', 'softhand', 'allegrohand', 'schunkhand'
            raise ValueError("Expecting the given 'hand' to be ['shadowhand', 'softhand', 'allegrohand', "
                             "'schunk_hand'], but instead got: {}".format(hand))
        self.robot = world.load_robot(hand, position=(-0.2, 0, 0.5), orientation=(-0.5, 0.5, -0.5, 0.5), left=False)

        if verbose:
            self.robot.print_info()

        # load cube in hand
        path = prl.world_mesh_path + 'manipulation/cube_with_letters/cube.obj'
        self.cube = world.load_mesh(path, position=[0.1, 0, 0.57], scale=(.05, .05, .05), flags=0, return_body=True)

        # load cameras if needed
        if with_camera:
            pass

        # create states
        states = prl.states

        state_dict = dict()
        state_dict['value'] = None
        state_dict['policy'] = None
        state_dict['vision'] = None
        self.state_dict = state_dict

        # create discrete actions
        actions = prl.actions.JointPositionChangeAction(robot, joint_ids=robot.joints, discrete_values=None)

        # create terminal condition
        drop_condition = None

        terminal_conditions = [drop_condition, ]

        # create reward
        rewards = None

        # create initial state generator
        initial_state_generators = None

        # create physics randomizer
        physics_randomizers = None

        # create environment using composition
        super(DexterityEnv, self).__init__(world=world, states=states, rewards=rewards,
                                           terminal_conditions=terminal_conditions,
                                           initial_state_generators=initial_state_generators,
                                           physics_randomizers=physics_randomizers, actions=actions)


# Test
if __name__ == '__main__':
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = DexterityEnv(sim, hand='shadowhand', verbose=True)

    # run simulation
    env.reset()
    for _ in count():
        env.step(sleep_dt=1. / 240)
