#!/usr/bin/env python
"""Provide the Shadow Hand robotic platform.
"""

import os

from pyrobolearn.robots.hand import Hand

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ShadowHand(Hand):
    r"""Shadow Hand

    References:
        - [1] https://www.shadowrobot.com/products/dexterous-hand/
        - [2] Shadow hand description: https://github.com/shadow-robot/sr_common
        - [3] Documentation: https://dexterous-hand.readthedocs.io/en/latest/index.html
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0.707, 0.707), fixed_base=True, scale=1.,
                 left=True):
        """
        Initialize the Shadow hand robot.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[3]): Cartesian world position.
            orientation (np.array[4]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
            left (bool): if we should create a left hand, or right hand.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.,)
        if fixed_base is None:
            fixed_base = True

        if left:
            if orientation is None:
                orientation = (0, 0, 0.707, 0.707)
            urdf_path = os.path.dirname(__file__) + '/urdfs/shadowhand/left_hand.urdf'
        else:
            if orientation is None:
                orientation = (0, 0, 1, 0)
            urdf_path = os.path.dirname(__file__) + '/urdfs/shadowhand/right_hand.urdf'

        super(ShadowHand, self).__init__(simulator, urdf_path, position, orientation, fixed_base, scale)
        self.name = 'shadow_hand'


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
    left_hand = ShadowHand(sim, position=(-0.15, 0, 0), left=True)
    right_hand = ShadowHand(sim, position=(0.15, 0., 0.), orientation=(0, 0, 0.707, -0.707), left=False)

    # print information about the robot
    left_hand.print_info()
    # H = left_hand.get_mass_matrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    # Position control using sliders
    # left_hand.add_joint_slider()

    left_hand.set_joint_positions([0.] * left_hand.num_dofs)
    right_hand.set_joint_positions([0.] * right_hand.num_dofs)

    for i in count():
        # left_hand.update_joint_slider()
        # left_hand.set_joint_positions([0.] * left_hand.num_dofs)
        # right_hand.set_joint_positions([0.] * right_hand.num_dofs)

        world.step(sleep_dt=1./240)
