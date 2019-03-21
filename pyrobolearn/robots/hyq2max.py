#!/usr/bin/env python
"""Provide the HyQ2Max robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.legged_robot import QuadrupedRobot


class HyQ2Max(QuadrupedRobot):
    r"""HyQ2Max

    HyQ2Max robot created by IIT.

    References:
        [1] "Design of the Hydraulically-Actuated,Torque-Controlled Quadruped Robot HyQ2Max", Semini et al., 2016
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.8),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/hyq2max/hyq2max.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.8)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.8,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(HyQ2Max, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'hyq2max'
        self.height = 0.9

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['lf_hipassembly', 'lf_upperleg', 'lf_lowerleg'],
                                   ['rf_hipassembly', 'rf_upperleg', 'rf_lowerleg'],
                                   ['lh_hipassembly', 'lh_upperleg', 'lh_lowerleg'],
                                   ['rh_hipassembly', 'rh_upperleg', 'rh_lowerleg']]]

        self.feet = [self.getLinkIds(link) for link in ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
                     if link in self.link_names]

        for foot in self.feet:
            self.sim.changeDynamics(self.id, foot, lateralFriction=.9, spinningFriction=1., rollingFriction=1.)
            self.sim.changeDynamics(self.id, foot, restitution=0.)

        # taken from "Learning agile and dynamic motor skills for legged robots", Hwangbo et al., 2019
        self.kp = 50. * np.ones(12)
        self.kd = 0.1 * np.ones(12)


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)
    world.loadJapaneseMonastery()

    # create robot
    robot = HyQ2Max(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider(robot.getLeftFrontLegIds())

    # run simulator
    for _ in count():
        # robot.updateJointSlider()
        robot.computeAndDrawCoMPosition()
        robot.computeAndDrawProjectedCoMPosition()

        world.step(sleep_dt=1./240)
