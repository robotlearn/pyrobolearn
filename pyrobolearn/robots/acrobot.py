#!/usr/bin/env python
"""Provide the acrobot robotic platform.
"""

import os
import numpy as np

from pyrobolearn.robots.robot import Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Acrobot(Robot):  # TODO: create the acrobot dynamically instead of loading from the URDF
    r"""Acrobot

    Note that in the URDF, the continuous joints were replace by revolute joints. Be careful, that the limit values
    for these joints are probably not correct.
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=True, scale=1,
                 urdf=os.path.dirname(__file__) + '/urdfs/rrbot/acrobot.urdf'):
        """
        Initialize the acrobot robot.

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
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = True

        super(Acrobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'acrobot'

        self.height = 2. * scale

        # set initial joint positions
        self.reset_joint_states(q=[np.pi, 0.], joint_ids=self.joints)

    def get_force_torque_sensor(self, idx=0):
        return np.array(self.sim.getJointState(self.id, 2)[2])


# Test
if __name__ == "__main__":
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create world
    world = prl.worlds.BasicWorld(sim)

    # load robot
    robot = Acrobot(sim)
    robot.disable_motor()
    robot.print_info()
    # robot.add_joint_slider()

    # run simulation
    for _ in count():
        world.step(sleep_dt=1./240)
