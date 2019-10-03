# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the ANYmal robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import QuadrupedRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ANYmal(QuadrupedRobot):
    r"""ANYmal robot

    'ANYmal is a dog-sized quadrupedal robot weighing about 32 kg. Each leg is about 55 cm long and has three actuated
    degrees of freedom, namely, hip abduction/adduction, hip flexion/extension, and knee flexion/extension.
    ANYmal is equipped with 12 SEAs'

    References:
        - [1] "ANYmal - a Highly Mobile and Dynamic Quadrupedal Robot", Hutter et al., 2016
        - [2] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
        - [3] www.rsl.ethz.ch/robots-media/anymal.html
        - [4] www.anybotics.com/anymal
        - [5] raisimLib: https://github.com/leggedrobotics/raisimLib
        - [6] raisimOgre - Visualizer for raisim: https://github.com/leggedrobotics/raisimOgre
        - [7] raisimGym - RL examples using raisim: https://github.com/leggedrobotics/raisimGym
    """

    def __init__(self, simulator, position=(0, 0, .6), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/anymal/anymal.urdf'):
        """
        Initialize the ANYmal robot.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.6)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.6,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(ANYmal, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'anymal'

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['LF_HIP', 'LF_THIGH', 'LF_SHANK'],
                                   ['RF_HIP', 'RF_THIGH', 'RF_SHANK'],
                                   ['LH_HIP', 'LH_THIGH', 'LH_SHANK'],
                                   ['RH_HIP', 'RH_THIGH', 'RH_SHANK']]]

        # self.feet = [self.get_link_ids(link) for link in ['LF_FOOT_MOUNT', 'RF_FOOT_MOUNT', 'LH_FOOT_MOUNT',
        #                                                   'RH_FOOT_MOUNT'] if link in self.link_names]
        self.feet = [self.get_link_ids(link) for link in ['LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT']
                     if link in self.link_names]

        # taken from "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
        # nominal range of torque (30 N·m) / nominal range of motion (0.6 rad)
        self.kp = 50. * np.ones(12)
        self.kd = 0.1 * np.ones(12)

        # taken from "raisimGym/raisim_gym/env/env/ANYmal/Environment.hpp"
        self.kp = 40. * np.ones(12)
        self.kd = 1. * np.ones(12)

        # init configuration
        self.reset_joint_states(q=[0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
        # self.reset_joint_states(q=[0.052, 1.66, -2.8, -0.052, 1.66, -2.8, 0.052, -1.66, 2.8, -0.052, -1.66, 2.8])

        # some values are taken from "raisimGym/raisim_gym/env/env/ANYmal/Environment.hpp"
        self.base_height = 0.54
        self.avg_height = 0.44

        self._joint_configuration = {'home': np.array([0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03,
                                                       -0.4, 0.8]),
                                     'standing': 'home',
                                     'init': 'home',
                                     'crouching': np.array([0.052, 1.66, -2.8, -0.052, 1.66, -2.8, 0.052, -1.66, 2.8,
                                                            -0.052, -1.66, 2.8]),
                                     'lying': 'crouching'}

    def get_home_joint_positions(self):
        """Return the joint positions for the home position."""
        return self._joint_configuration['home']


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = ANYmal(sim)

    # print information about the robot
    robot.print_info()
    print("BASE HEIGHT: {}".format(robot.base_height))
    # Position control using sliders
    # robot.add_joint_slider(robot.left_front_leg)

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        robot.step()
        print("BASE HEIGHT: {}".format(robot.get_base_position()[2]))
        print(robot.get_joint_positions())
        world.step(sleep_dt=1./240)
