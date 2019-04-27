#!/usr/bin/env python
"""Provide the Hubo robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot
from pyrobolearn.robots.hand import TwoHand

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Hubo(BipedRobot, BiManipulatorRobot, TwoHand):
    r"""Hubo robot

    "The HUBO series of biped robots was developed by the Humanoid Robot Research Center at KAIST (Korea Advanced
    Institute of Science & Technology), and is manufactured by spin-off company Rainbow Inc." [1]

    It is fully humanoid platform with 58 DoFs; 3 for the neck, 6 for each arm, 6 for each leg, and 3 for each finger,
    and 1 for the waist.

    References:
        [1] ROS wiki: http://wiki.ros.org/Robots/HUBO
        [2] Hubo Lab: hubolab.kaist.ac.kr
        [3] Rainbow Robotics: http://www.rainbow-robotics.com/new/
        [4] URDF: https://github.com/robEllenberg/hubo-urdf
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 1),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scaling=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/hubo/hubo.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 1.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (1.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Hubo, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
        self.name = 'hubo'

        self.neck = [self.get_link_ids(link) for link in ['Body_Neck', 'Body_Head_Empty', 'Body_Head']
                     if link in self.link_names]
        self.head = self.get_link_ids('Body_Head') if 'Body_Head' in self.link_names else None
        self.waist = self.get_link_ids('Body_Hip') if 'Body_Hip' in self.link_names else None

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['Body_LHY', 'Body_LHR', 'Body_LHP', 'Body_LKN', 'Body_LAP', 'Body_LAR'],
                                   ['Body_RHY', 'Body_RHR', 'Body_RHP', 'Body_RKN', 'Body_RAP', 'Body_RAR']]]

        self.feet = [self.get_link_ids(link) for link in ['Body_LAR', 'Body_RAR'] if link in self.link_names]

        self.arms = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['Body_LSP', 'Body_LSR', 'Body_LSY', 'Body_LEB', 'Body_LWY', 'Body_LWP'],
                                   ['Body_RSP', 'Body_RSR', 'Body_RSY', 'Body_REB', 'Body_RWY', 'Body_RWP']]]

        self.hands = [self.get_link_ids(link) for link in ['Body_LWP', 'Body_RWP'] if link in self.link_names]

        self.fingers = [[self.get_link_ids(link) for link in links if link in self.link_names]
                        for links in [['leftThumbProximal', 'leftThumbMedial', 'leftThumbDistal'],
                                      ['leftIndexProximal', 'leftIndexMedial', 'leftIndexDistal'],
                                      ['leftMiddleProximal', 'leftMiddleMedial', 'leftMiddleDistal'],
                                      ['leftRingProximal', 'leftRingMedial', 'leftRingDistal'],
                                      ['leftPinkyProximal', 'leftPinkyMedial', 'leftPinkyDistal'],
                                      ['rightThumbProximal', 'rightThumbMedial', 'rightThumbDistal'],
                                      ['rightIndexProximal', 'rightIndexMedial', 'rightIndexDistal'],
                                      ['rightMiddleProximal', 'rightMiddleMedial', 'rightMiddleDistal'],
                                      ['rightRingProximal', 'rightRingMedial', 'rightRingDistal'],
                                      ['rightPinkyProximal', 'rightPinkyMedial', 'rightPinkyDistal']]]

        self.left_fingers = range(5)
        self.right_fingers = range(5, 10)


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
    robot = Hubo(sim)

    # print information about the robot
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
