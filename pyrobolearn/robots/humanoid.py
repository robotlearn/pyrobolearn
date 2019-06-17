#!/usr/bin/env python
"""Provide the Humanoid Mujoco model.
"""

import os

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Humanoid(BipedRobot, BiManipulatorRobot):
    r"""Humanoid Mujoco Model

    References:
        [1] description: https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/mjcf
    """

    def __init__(self,
                 simulator,
                 position=(-0.5, 0, 1.),
                 orientation=(0, 0.707, 0, 0.707),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/mjcfs/humanoid.xml'):  # humanoid_symmetric.xml
        # check parameters
        if position is None:
            position = (-0.5, 0., 1.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (1.,)
        if orientation is None:
            orientation = (0, 0.707, 0, 0.707)
        if fixed_base is None:
            fixed_base = False

        super(Humanoid, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'humanoid'

        # self.waist = self.get_link_ids('DWL') if 'DWL' in self.link_names else None
        # self.torso = self.get_link_ids('DWYTorso') if 'DWYTorso' in self.link_names else None
        #
        # self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
        #              for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
        #                            ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]

        self.feet = [self.get_link_ids(link) for link in ['left_foot', 'right_foot'] if link in self.link_names]

        # self.arms = [[self.get_link_ids(link) for link in links if link in self.link_names]
        #              for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
        #                            ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]

        self.hands = [self.get_link_ids(link) for link in ['left_lower_arm', 'right_lower_arm']
                      if link in self.link_names]


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
    robot = Humanoid(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulation
    for i in count():
        # robot.update_joint_slider()
        # step in simulation
        world.step(sleep_dt=1./240)
