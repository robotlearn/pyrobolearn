#!/usr/bin/env python
"""Provide the Soft Hand robotic platform.
"""

import os
from hand import Hand


# TODO: correct the inertia matrices and masses; they are too big!
class SoftHand(Hand):
    r"""Pisa-IIT Soft Hand

    References:
        [1] https://github.com/CentroEPiaggio/pisa-iit-soft-hand
        [2] "Adaptive Synergies for the Design and Control of the Pisa/IIT SoftHand", Catalano et al., 2014
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0),
                 init_orient=(0, 0, 0, 1),
                 scaling=1.,
                 left=True,
                 useFixedBase=True):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.,)
        if useFixedBase is None:
            useFixedBase = True

        if left:
            if init_orient is None:
                init_orient = (0, 0, 0, 1)
            urdf_path = os.path.dirname(__file__) + '/urdfs/softhand/left_hand.urdf'
        else:
            if init_orient is None:
                init_orient = (0, 0, 1, 0)
            urdf_path = os.path.dirname(__file__) + '/urdfs/softhand/right_hand.urdf'

        super(SoftHand, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
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
    left_hand = SoftHand(sim, init_pos=(-0.15, 0, 0), left=True)
    right_hand = SoftHand(sim, init_pos=(0.15, 0., 0.), init_orient=(0, 0, 1, 0), left=False)

    # print information about the robot
    left_hand.printRobotInfo()
    # H = left_hand.calculateMassMatrix()
    # print("Inertia matrix: H(q) = {}".format(H))

    # Position control using sliders
    # left_hand.addJointSlider()

    left_hand.setJointPositions([0.] * left_hand.getNumberOfDoFs())
    right_hand.setJointPositions([0.] * right_hand.getNumberOfDoFs())

    for i in count():
        # left_hand.updateJointSlider()
        # left_hand.setJointPositions([0.] * left_hand.getNumberOfDoFs())
        # right_hand.setJointPositions([0.] * right_hand.getNumberOfDoFs())

        world.step(sleep_dt=1./240)
