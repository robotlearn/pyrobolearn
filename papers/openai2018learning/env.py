#!/usr/bin/env python
"""
Provide the environment (world, states, actions, rewards, terminal conditions, etc) defined in the paper [1] using the
PyRoboLearn framework.

Reference:
    [1] "Learning Dexterous In-Hand Manipulation", OpenAI et al., 2018 (https://arxiv.org/abs/1808.00177)
"""

# TODO: end the implementation

import os
import pyrobolearn as prl

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["OpenAI et al."]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# define environment
class Openai2018LearningEnv(prl.envs.Env):
    r"""Openai2018Learning Environment

    Create the environment (world, states, actions, rewards, terminal conditions, etc) defined in the paper [1].

    Reference:
    [1] "Learning Dexterous In-Hand Manipulation", OpenAI et al., 2018 (https://arxiv.org/abs/1808.00177)
    """

    def __init__(self, simulator=None):
        r"""
        Initialize the environment.

        Args:
            simulator (prl.simulators.Simulator): simulator instance or None. If None, it will use PyBullet by default.
        """
        # create simulator if None
        if simulator is None:
            simulator = prl.simulators.Bullet()
        if not isinstance(simulator, prl.simulators.Simulator):
            raise TypeError("Expecting the given simulator to be None or an instance of `prl.simulators.Simulator`, "
                            "instead got: {}".format(type(simulator)))

        # create world
        world = prl.worlds.BasicWorld(simulator)

        # load hand
        robot = world.load_robot('shadowhand', position=(-0.2, 0, 0.5), orientation=(-0.5, 0.5, -0.5, 0.5), left=False)
        robot.print_info()

        # load cube in hand
        cube = world.load_mesh(os.path.dirname(__file__) + '/meshes/cube.obj', position=[0.1, 0, 0.57],
                               scale=(.05, .05, .05), flags=0)
        cube = prl.robots.Body(simulator, body_id=cube)

        # create state: fingertip positions (5*3D), object position (3D), object orientation (4D=quaternion),
        # target orientation (4D=quaternion), relative target orientation (4D=quaternion), hand joint angles (24D),
        # hand joint velocities (24D), object velocity (3D), object angular velocity (3D)
        state = prl.states.PositionState(cube) + prl.states.OrientationState(cube) + \
                prl.states.JointPositionState(robot)

        # create action

        # create reward

        super(Openai2018LearningEnv, self).__init__(world, state)


# Test environment
if __name__ == '__main__':
    import time
    from itertools import count

    # create environment
    env = Openai2018LearningEnv()

    # run environment
    for t in count():
        env.step()
        time.sleep(1./240)
