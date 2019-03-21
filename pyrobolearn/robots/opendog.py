#!/usr/bin/env python
"""Provide the OpenDog robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import QuadrupedRobot


class OpenDog(QuadrupedRobot):
    r""" OpenDog robot

    References:
        [1] https://github.com/XRobots/openDog
        [2] https://github.com/wiccopruebas/opendog_project
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, .6),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/opendog/opendog.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.6)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.6,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(OpenDog, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'opendog'

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['lf_hip', 'lf_upperleg', 'lf_lowerleg'],
                                   ['rf_hip', 'rf_upperleg', 'rf_lowerleg'],
                                   ['lb_hip', 'lb_upperleg', 'lb_lowerleg'],
                                   ['rb_hip', 'rb_upperleg', 'rb_lowerleg']]]

        self.feet = [self.getLinkIds(link) for link in ['lf_lowerleg', 'rf_lowerleg', 'lb_lowerleg', 'rb_lowerleg']
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
    robot = OpenDog(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider(robot.getLeftFrontLegIds())

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        world.step(sleep_dt=1./240)
