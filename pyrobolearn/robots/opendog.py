#!/usr/bin/env python
"""Provide the OpenDog robotic platform.
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


class OpenDog(QuadrupedRobot):
    r""" OpenDog robot

    References:
        - [1] https://github.com/XRobots/openDog
        - [2] https://github.com/wiccopruebas/opendog_project
    """

    def __init__(self, simulator, position=(0, 0, .6), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/opendog/opendog.urdf'):
        """
        Initialize the OpenDog robot.

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
            position = (0., 0., 0.6)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.6,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(OpenDog, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'opendog'

        self.legs = [[self.get_link_ids(link) for link in links if link in self.link_names]
                     for links in [['lf_hip', 'lf_upperleg', 'lf_lowerleg'],
                                   ['rf_hip', 'rf_upperleg', 'rf_lowerleg'],
                                   ['lb_hip', 'lb_upperleg', 'lb_lowerleg'],
                                   ['rb_hip', 'rb_upperleg', 'rb_lowerleg']]]

        self.feet = [self.get_link_ids(link) for link in ['lf_lowerleg', 'rf_lowerleg', 'lb_lowerleg', 'rb_lowerleg']
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
    robot.print_info()

    # Position control using sliders
    # robot.add_joint_slider(robot.left_front_leg)

    # run simulator
    for _ in count():
        # robot.update_joint_slider()
        world.step(sleep_dt=1./240)
