#!/usr/bin/env python
"""Provide the Hand abstract classes.
"""

from pyrobolearn.robots.robot import Robot


class Hand(Robot):
    r"""Hand end-effector

    """

    def __init__(self,
                 simulator,
                 urdf,
                 position=(0, 0, 1.),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scaling=1.):
        super(Hand, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.fingers = []  # list of fingers where each finger is a list of links/joints

    @property
    def num_fingers(self):
        """Return the number of fingers on the hand"""
        return len(self.fingers)

    def getFinger(self, fingerId=None):
        """Return the list of joint/link ids for the specified finger"""
        if fingerId:
            return self.fingers[fingerId]
        return self.fingers


class TwoHand(Hand):
    r"""Two hand end-effectors

    """
    def __init__(self,
                 simulator,
                 urdf,
                 position=(0, 0, 1.),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scaling=1.):
        super(TwoHand, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)

        self.left_fingers = []  # list of ids in self.fingers
        self.right_fingers = []     # list of ids in self.fingers

    @property
    def num_fingers_left_hand(self):
        """Return the number of fingers on the left hand"""
        return len(self.left_fingers)

    @property
    def num_fingers_right_hand(self):
        """Return the number of fingers on the right hand"""
        return len(self.right_fingers)

    def getLeftFingers(self, fingerId=None):
        """Return the specified left fingers"""
        if fingerId:
            if isinstance(fingerId, int):
                return self.fingers[self.left_fingers[fingerId]]
            elif isinstance(fingerId, (tuple, list)):
                return [self.fingers[self.left_fingers[finger]] for finger in fingerId]
        return [self.fingers[finger] for finger in self.left_fingers]

    def getRightFingers(self, fingerId=None):
        """Return the specified right fingers"""
        if fingerId:
            if isinstance(fingerId, int):
                return self.fingers[self.right_fingers[fingerId]]
            elif isinstance(fingerId, (tuple, list)):
                return [self.fingers[self.right_fingers[finger]] for finger in fingerId]
        return [self.fingers[finger] for finger in self.right_fingers]
