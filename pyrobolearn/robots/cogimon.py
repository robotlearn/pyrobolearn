#!/usr/bin/env python
"""Provide the Cogimon robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import BipedRobot
from pyrobolearn.robots.manipulator import BiManipulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Cogimon(BipedRobot, BiManipulator):
    r"""Cogimon humanoid robot.

    References:
        - [1] https://github.com/ADVRHumanoids/iit-cogimon-ros-pkg
    """

    def __init__(self, simulator, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 lower_body=False, urdf=os.path.dirname(__file__) + '/urdfs/cogimon/cogimon.urdf'):
        # cogimon_lower_body.urdf
        """
        Initialize the Cogimon robot.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            lower_body (bool): if we should only use the lower body of the robot.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 1.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (1.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False
        if lower_body:
            urdf = os.path.dirname(__file__) + '/urdfs/cogimon/cogimon_lower_body.urdf'

        super(Cogimon, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'cogimon'

        self.waist = self.get_link_ids('DWL') if 'DWL' in self.link_names else None
        self.torso = self.get_link_ids('DWYTorso') if 'DWYTorso' in self.link_names else None

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
                                   ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]

        self.feet = [self.get_link_ids(link) for link in ['LFoot', 'RFoot'] if link in self.link_names]

        self.arms = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
                                   ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]

        self.hands = [self.get_link_ids(link) for link in ['LSoftHand', 'RSoftHand'] if link in self.link_names]


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
    robot = Cogimon(sim, lower_body=True)

    # print information about the robot
    robot.print_info()

    # # Position control using sliders
    robot.add_joint_slider(robot.left_leg)

    # run simulator
    for _ in count():
        robot.update_joint_slider()
        world.step(sleep_dt=1./240)
