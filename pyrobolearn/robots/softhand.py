#!/usr/bin/env python
"""Provide the Soft Hand robotic platform.
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


# TODO: correct the inertia matrices and masses; they are too big!
class SoftHand(Hand):
    r"""Pisa-IIT Soft Hand

    References:
        - [1] https://github.com/CentroEPiaggio/pisa-iit-soft-hand
        - [2] "Adaptive Synergies for the Design and Control of the Pisa/IIT SoftHand", Catalano et al., 2014
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed_base=True, scale=1., left=True):
        """
        Initialize the Soft hand robot.

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
                orientation = (0, 0, 0, 1)
            urdf_path = os.path.dirname(__file__) + '/urdfs/softhand/left_hand.urdf'
        else:
            if orientation is None:
                orientation = (0, 0, 1, 0)
            urdf_path = os.path.dirname(__file__) + '/urdfs/softhand/right_hand.urdf'

        super(SoftHand, self).__init__(simulator, urdf_path, position, orientation, fixed_base, scale)
        self.name = 'soft_hand'


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
    left_hand = SoftHand(sim, position=(-0.15, 0, 0), left=True)
    right_hand = SoftHand(sim, position=(0.15, 0., 0.), orientation=(0, 0, 1, 0), left=False)

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
