#!/usr/bin/env python
"""Provide the Allegro hand robotic platform.
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
class AllegroHand(Hand):
    r"""Allegro Hand

    References:
        - [1] http://www.simlab.co.kr/Allegro-Hand.htm
        - [2] https://github.com/simlabrobotics/allegro_hand_ros
    """

    def __init__(self, simulator, position=(0, 0, 0), orientation=(0, 0, 0, 1), scale=1., fixed_base=True):
        # left=False
        """
        Initialize the Allegro hand.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[3]): Cartesian world position.
            orientation (np.array[4]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the hand base will be fixed in the world.
            scale (float): scaling factor that is used to scale the hand.
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

        # if left:
        #     urdf_path = '../robots/urdfs/allegrohand/allegro_left_hand.urdf'
        # else:
        urdf_path = os.path.dirname(__file__) + '/urdfs/allegrohand/allegro_right_hand.urdf'

        super(AllegroHand, self).__init__(simulator, urdf_path, position, orientation, fixed_base, scale)
        self.name = 'allegro_hand'


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
    right_hand = AllegroHand(sim)  # , position=(0.,0.,0.), orientation=(0,0,1,0))

    # print information about the robot
    right_hand.print_info()
    # H = right_hand.get_mass_matrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    # Position control using sliders
    right_hand.add_joint_slider()

    for i in count():
        right_hand.update_joint_slider()
        # right_hand.set_joint_positions([0.] * right_hand.num_dofs)

        # step in simulation
        world.step(sleep_dt=1./240)
