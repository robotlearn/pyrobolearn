#!/usr/bin/env python
"""Provide the Humanoid Mujoco model.
"""

import os
from legged_robot import BipedRobot
from manipulator import BiManipulatorRobot


class Humanoid(BipedRobot, BiManipulatorRobot):
    r"""Humanoid Mujoco Model
    """

    def __init__(self,
                 simulator,
                 init_pos=(-0.5, 0, 1.),
                 init_orient=(0, 0.707, 0, 0.707),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/mjcfs/humanoid.xml'):  # humanoid_symmetric.xml
        # check parameters
        if init_pos is None:
            init_pos = (-0.5, 0., 1.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (1.,)
        if init_orient is None:
            init_orient = (0, 0.707, 0, 0.707)
        if useFixedBase is None:
            useFixedBase = False

        super(Humanoid, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'humanoid'

        # self.waist = self.getLinkIds('DWL') if 'DWL' in self.link_names else None
        # self.torso = self.getLinkIds('DWYTorso') if 'DWYTorso' in self.link_names else None
        #
        # self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
        #              for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
        #                            ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]

        self.feet = [self.getLinkIds(link) for link in ['left_foot', 'right_foot'] if link in self.link_names]

        # self.arms = [[self.getLinkIds(link) for link in links if link in self.link_names]
        #              for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
        #                            ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]

        self.hands = [self.getLinkIds(link) for link in ['left_lower_arm', 'right_lower_arm']
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
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulation
    for i in count():
        # robot.updateJointSlider()
        # step in simulation
        world.step(sleep_dt=1./240)
