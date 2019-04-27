#!/usr/bin/env python
"""Provide the ANYmal robotic platform.
"""

import os
import numpy as np

from legged_robot import QuadrupedRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "MIT"
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
        [1] "ANYmal - a Highly Mobile and Dynamic Quadrupedal Robot", Hutter et al., 2016
        [2] "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
        [3] www.rsl.ethz.ch/robots-media/anymal.html
        [4] www.anybotics.com/anymal
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, .6),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scaling=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/anymal/anymal.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.6)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.6,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(ANYmal, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
        self.name = 'anymal'

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['LF_HIP', 'LF_THIGH', 'LF_SHANK'],
                                   ['RF_HIP', 'RF_THIGH', 'RF_SHANK'],
                                   ['LH_HIP', 'LH_THIGH', 'LH_SHANK'],
                                   ['RH_HIP', 'RH_THIGH', 'RH_SHANK']]]

        self.feet = [self.get_link_ids(link) for link in ['LF_FOOT_MOUNT', 'RF_FOOT_MOUNT', 'LH_FOOT_MOUNT',
                                                          'RH_FOOT_MOUNT'] if link in self.link_names]

        # taken from "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
        self.kp = 50. * np.ones(12)
        self.kd = 0.1 * np.ones(12)


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = ANYmal(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider(robot.left_front_leg)

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
