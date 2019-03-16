#!/usr/bin/env python
"""Provide the Minitaur robotic platform.
"""

import os
from legged_robot import QuadrupedRobot


class Minitaur(QuadrupedRobot):
    r"""Minitaur robot

    Minitaur robot from Ghost Robotics (https://www.ghostrobotics.io/)

    References:
        [1] pybullet_envs/bullet/minitaur.py
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, .2),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/minitaur/minitaur.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.2)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.2,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Minitaur, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'minitaur'

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['motor_front_leftL_link', 'motor_front_leftR_link',
                                    'lower_leg_front_leftL_link', 'lower_leg_front_leftR_link'],
                                   ['motor_front_rightL_link', 'motor_front_rightR_link',
                                    'lower_leg_front_rightL_link', 'lower_leg_front_rightR_link'],
                                   ['motor_back_leftL_link', 'motor_back_leftR_link',
                                    'lower_leg_back_leftL_link', 'lower_leg_back_leftR_link'],
                                   ['motor_back_rightL_link', 'motor_back_rightR_link',
                                    'lower_leg_back_rightL_link', 'lower_leg_back_rightR_link']]]

        self.feet = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['lower_leg_front_leftL_link', 'lower_leg_front_leftR_link'],
                                   ['lower_leg_front_rightL_link', 'lower_leg_front_rightR_link'],
                                   ['lower_leg_back_leftL_link', 'lower_leg_back_leftR_link'],
                                   ['lower_leg_back_rightL_link', 'lower_leg_back_rightR_link']]]


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
    robot = Minitaur(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    robot.addJointSlider(robot.getLeftFrontLegIds())

    # run simulator
    for _ in count():
        robot.updateJointSlider()
        # robot.computeAndDrawCoMPosition()
        # robot.computeAndDrawProjectedCoMPosition()
        world.step(sleep_dt=1./240)
