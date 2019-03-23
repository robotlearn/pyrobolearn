#!/usr/bin/env python
"""Provide the Aibo robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import QuadrupedRobot


class Aibo(QuadrupedRobot):
    r"""Aibo

    This is the Aibo quadruped dog robot developed by Sony.

    WARNINGS: THE INERTIA MATRICES AND THE POSITION OF COLLISIONS MESHES IN THE URDF NEED TO BE CORRECTED!!
    """

    def __init__(self,
                 simulator,
                 position=(0, 0, 0.02),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scaling=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/aibo/aibo.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.02)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.02,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Aibo, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
        self.name = 'aibo'

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['LF_up_leg_rot_y', 'LF_up_leg_rot_x', 'LF_down_leg'],
                                   ['RF_up_leg_rot_y', 'RF_up_leg_rot_x', 'RF_down_leg'],
                                   ['LB_up_leg_rot_y', 'LB_up_leg_rot_x', 'LB_down_leg'],
                                   ['RB_up_leg_rot_y', 'RB_up_leg_rot_x', 'RB_down_leg']]]

        self.feet = [self.get_link_ids(link) for link in ['LF_paw', 'RF_paw', 'LB_paw', 'RB_paw']
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
    robot = Aibo(sim)  # , useFixedBase=True)

    # print information about the robot
    robot.print_info()

    # # Position control using sliders
    # robot.add_joint_slider(robot.getLeftFrontLegIds() + robot.getRightFrontLegIds())

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
