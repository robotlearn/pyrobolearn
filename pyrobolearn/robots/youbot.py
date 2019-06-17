#!/usr/bin/env python
"""Provide the Youbot robotic platforms.

These include: YoubotBase, KukaYoubotArm, Youbot, YoubotDualArm
"""

import os
import numpy as np

from pyrobolearn.robots.manipulator import ManipulatorRobot, BiManipulatorRobot
from pyrobolearn.robots.wheeled_robot import DifferentialWheeledRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class YoubotBase(DifferentialWheeledRobot):
    r"""Youbot Base robot

    References:
        [1] https://github.com/youbot
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.085),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/youbot/youbot_base_only.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.085)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.085,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(YoubotBase, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'youbot_base'

        # self.wheels = [self.get_link_ids(link) for link in ['left_wheel', 'right_wheel']
        #                if link in self.link_names]
        # self.wheel_directions = np.ones(len(self.wheels))


class KukaYoubotArm(ManipulatorRobot):
    r"""Kuka Youbot arm robot

    References:
        [1] https://github.com/youbot
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.03),
                 orientation=(0, 0, 0, 1),
                 fixed_base=True,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/youbot/youbot_arm_only.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.03)
        if len(position) == 2:  # assume x, y are given
            position += (0.03,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(KukaYoubotArm, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'kuka_youbot_arm'


class Youbot(ManipulatorRobot, DifferentialWheeledRobot):
    r"""Youbot robot

    References:
        [1] https://github.com/youbot
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.085),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/youbot/youbot.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.085)
        if len(position) == 2:  # assume x, y are given
            position += (0.085,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Youbot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'youbot'

        # self.wheels = [self.get_link_ids(link) for link in ['left_wheel', 'right_wheel']
        #                if link in self.link_names]
        # self.wheel_directions = np.ones(len(self.wheels))


class YoubotDualArm(BiManipulatorRobot, DifferentialWheeledRobot):
    r"""Youbot dual arm robot

    References:
        [1] https://github.com/youbot
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.085),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/youbot/youbot_dual_arm.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.085)
        if len(position) == 2:  # assume x, y are given
            position += (0.085,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(YoubotDualArm, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'youbot_dual_arm'

        # self.wheels = [self.get_link_ids(link) for link in ['left_wheel', 'right_wheel']
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
    youbot_base = YoubotBase(sim, position=(0, -0.75))
    kuka_arm = KukaYoubotArm(sim, position=(0, -0.25))
    youbot = Youbot(sim, position=(0, 0.25))
    youbot_dual_arm = YoubotDualArm(sim, position=(0, 0.75))

    robots = [youbot_base, kuka_arm, youbot, youbot_dual_arm]

    # print information about the robot
    for robot in robots:
        robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider()

    # run simulator
    for _ in count():
        # robots[0].update_joint_slider()
        # for robot in robots:
        #     robot.drive(5)
        world.step(sleep_dt=1./240)
