#!/usr/bin/env python
"""Provide the Allegro hand robotic platform.
"""

import os
from hand import Hand


# TODO: correct the inertia matrices and masses; they are too big!
class AllegroHand(Hand):
    r"""Allegro Hand

    References:
        [1] http://www.simlab.co.kr/Allegro-Hand.htm
        [2] https://github.com/simlabrobotics/allegro_hand_ros
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 scaling=1.,
                 left=False,
                 useFixedBase=True):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = True

        # if left:
        #     urdf_path = '../robots/urdfs/allegrohand/allegro_left_hand.urdf'
        # else:
        urdf_path = os.path.dirname(__file__) + '/urdfs/allegrohand/allegro_right_hand.urdf'

        super(AllegroHand, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
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
    right_hand = AllegroHand(sim)  # , init_pos=(0.,0.,0.), init_orient=(0,0,1,0))

    # print information about the robot
    right_hand.printRobotInfo()
    # H = right_hand.calculateMassMatrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    # Position control using sliders
    right_hand.addJointSlider()

    for i in count():
        right_hand.updateJointSlider()
        # right_hand.setJointPositions([0.] * right_hand.getNumberOfDoFs())

        # step in simulation
        world.step(sleep_dt=1./240)
