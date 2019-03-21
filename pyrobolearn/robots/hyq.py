#!/usr/bin/env python
"""Provide the HyQ robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import QuadrupedRobot


class HyQ(QuadrupedRobot):
    r"""HyQ robot

    HyQ robot created by IIT.

    References:
        [1]
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, .9),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/hyq/hyq.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.9)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.9,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(HyQ, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'hyq'

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['lf_hipassembly', 'lf_upperleg', 'lf_lowerleg'],
                                   ['rf_hipassembly', 'rf_upperleg', 'rf_lowerleg'],
                                   ['lh_hipassembly', 'lh_upperleg', 'lh_lowerleg'],
                                   ['rh_hipassembly', 'rh_upperleg', 'rh_lowerleg']]]

        self.feet = [self.getLinkIds(link) for link in ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
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
    robot = HyQ(sim)

    # print information about the robot
    robot.printRobotInfo()

    # # Position control using sliders
    robot.addJointSlider(robot.getLeftFrontLegIds())

    # run simulator
    for _ in count():
        robot.updateJointSlider()
        robot.computeAndDrawCoMPosition()
        robot.computeAndDrawProjectedCoMPosition()
        world.step(sleep_dt=1./240)
