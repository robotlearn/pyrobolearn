#!/usr/bin/env python
"""Provide the Cogimon robotic platform.
"""

import os
from legged_robot import BipedRobot
from manipulator import BiManipulatorRobot


class Cogimon(BipedRobot, BiManipulatorRobot):
    r"""Cogimon humanoid robot.

    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 1.),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/cogimon/cogimon.urdf',
                 lower_body=False):  # cogimon_lower_body.urdf

        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 1.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (1.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False
        if lower_body:
            urdf_path = os.path.dirname(__file__) + '/urdfs/cogimon/cogimon_lower_body.urdf'

        super(Cogimon, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'cogimon'

        self.waist = self.getLinkIds('DWL') if 'DWL' in self.link_names else None
        self.torso = self.getLinkIds('DWYTorso') if 'DWYTorso' in self.link_names else None

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
                                   ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]

        self.feet = [self.getLinkIds(link) for link in ['LFoot', 'RFoot'] if link in self.link_names]

        self.arms = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
                                   ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]

        self.hands = [self.getLinkIds(link) for link in ['LSoftHand', 'RSoftHand'] if link in self.link_names]


def CogimonLowerBody(simulator, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False, scaling=1.,
                     urdf_path=os.path.dirname(__file__) + '/urdfs/cogimon/cogimon_lower_body.urdf'):
    """Load Cogimon Lower Body"""
    return Cogimon(simulator=simulator, init_pos=init_pos, init_orient=init_orient, useFixedBase=useFixedBase,
                   scaling=scaling, urdf_path=urdf_path)


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
    robot = Cogimon(sim, lower_body=False)

    # print information about the robot
    robot.printRobotInfo()

    # # Position control using sliders
    robot.addJointSlider(robot.left_leg)

    # run simulator
    for _ in count():
        robot.updateJointSlider()
        world.step(sleep_dt=1./240)
