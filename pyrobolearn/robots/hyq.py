#!/usr/bin/env python
"""Provide the HyQ robotic platform.
"""

import os

from pyrobolearn.robots.legged_robot import QuadrupedRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class HyQ(QuadrupedRobot):
    r"""HyQ robot

    HyQ robot created by IIT.

    References:
        - [1] https://dls.iit.it/robots/hyq-robot
        - [2] https://github.com/iit-DLSLab/hyq-description
    """

    def __init__(self, simulator, position=(0, 0, .9), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/hyq/hyq.urdf'):
        """
        Initialize the HyQ robot.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.9)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.9,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(HyQ, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'hyq'

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['lf_hipassembly', 'lf_upperleg', 'lf_lowerleg'],
                                   ['rf_hipassembly', 'rf_upperleg', 'rf_lowerleg'],
                                   ['lh_hipassembly', 'lh_upperleg', 'lh_lowerleg'],
                                   ['rh_hipassembly', 'rh_upperleg', 'rh_lowerleg']]]

        self.feet = [self.get_link_ids(link) for link in ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
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
    robot.print_info()

    # # Position control using sliders
    robot.add_joint_slider(robot.left_front_leg)

    # run simulator
    for _ in count():
        robot.update_joint_slider()
        robot.compute_and_draw_com_position()
        robot.compute_and_draw_projected_com_position()
        world.step(sleep_dt=1./240)
