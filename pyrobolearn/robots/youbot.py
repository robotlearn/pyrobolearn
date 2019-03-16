#!/usr/bin/env python
"""Provide the Youbot robotic platforms.

These include: YoubotBase, KukaYoubotArm, Youbot, YoubotDualArm
"""

import os
import numpy as np

from manipulator import ManipulatorRobot, BiManipulatorRobot
from wheeled_robot import DifferentialWheeledRobot


class YoubotBase(DifferentialWheeledRobot):
    r"""Youbot Base robot

    References:
        [1] https://github.com/youbot
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.085),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/youbot/youbot_base_only.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.085)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.085,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(YoubotBase, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'youbot_base'

        # self.wheels = [self.getLinkIds(link) for link in ['left_wheel', 'right_wheel']
        #                if link in self.link_names]
        # self.wheel_directions = np.ones(len(self.wheels))


class KukaYoubotArm(ManipulatorRobot):
    r"""Kuka Youbot arm robot

    References:
        [1] https://github.com/youbot
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.03),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=True,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/youbot/youbot_arm_only.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.03)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos += (0.03,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        super(KukaYoubotArm, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'kuka_youbot_arm'


class Youbot(ManipulatorRobot, DifferentialWheeledRobot):
    r"""Youbot robot

    References:
        [1] https://github.com/youbot
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.085),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/youbot/youbot.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.085)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos += (0.085,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Youbot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'youbot'

        # self.wheels = [self.getLinkIds(link) for link in ['left_wheel', 'right_wheel']
        #                if link in self.link_names]
        # self.wheel_directions = np.ones(len(self.wheels))


class YoubotDualArm(BiManipulatorRobot, DifferentialWheeledRobot):
    r"""Youbot dual arm robot

    References:
        [1] https://github.com/youbot
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.085),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/youbot/youbot_dual_arm.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.085)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos += (0.085,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(YoubotDualArm, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'youbot_dual_arm'

        # self.wheels = [self.getLinkIds(link) for link in ['left_wheel', 'right_wheel']
        #                if link in self.link_names]
        # self.wheel_directions = np.ones(len(self.wheels))


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
    youbot_base = YoubotBase(sim, init_pos=(0, -0.75))
    kuka_arm = KukaYoubotArm(sim, init_pos=(0, -0.25))
    youbot = Youbot(sim, init_pos=(0, 0.25))
    youbot_dual_arm = YoubotDualArm(sim, init_pos=(0, 0.75))

    robots = [youbot_base, kuka_arm, youbot, youbot_dual_arm]

    # print information about the robot
    for robot in robots:
        robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for _ in count():
        # robots[0].updateJointSlider()
        # for robot in robots:
        #     robot.drive(5)
        world.step(sleep_dt=1./240)
