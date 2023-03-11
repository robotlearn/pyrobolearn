#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Hand abstract classes.
"""

from pyrobolearn.robots.gripper import AngularGripper


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Hand(AngularGripper):
    r"""Hand end-effector
    """

    def __init__(self, simulator, urdf, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.):
        """
        Initialize the hand robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(Hand, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

    def get_fist_configuration(self):
        """Return the joint configuration for the hand (and fingers) to form a fist."""
        pass

    def get_open_configuration(self):
        """Return the joint configuration for the hand (and fingers) to form an open hand (where the fingers are
        attached/close to each other)."""
        pass

    def get_fully_open_configuration(self):
        """Return the joint configuration for the hand (and fingers) to form a fully open hand (where the distance
        between the fingers is maximal)."""
        pass

    def get_thumb_up_configuration(self, factor=1.):
        """
        Return the joint configuration for the fingers to have the thumb up.

        Args:
            factor (float): 1 = completely up, 0 = completely down
        """
        pass

    def get_thumb_down_configuration(self, factor=1.):
        """
        Return the joint configuration for the fingers to have the thumb up.

        Args:
            factor (float): 1 = completely down, 0 = completely up
        """
        return self.get_thumb_up_configuration(factor=1.-factor)

    def get_index_up_configuration(self, factor=1.):
        """
        Return the joint configuration for the fingers to have the index finger up.

        Args:
            factor (float): 1 = completely up, 0 = completely down
        """
        pass

    def get_index_down_configuration(self, factor=1.):
        """
        Return the joint configuration for the fingers to have the index finger up.

        Args:
            factor (float): 1 = completely down, 0 = completely up
        """
        return self.get_index_up_configuration(factor=1.-factor)

    def get_middle_up_configuration(self, factor=1.):
        """
        Return the joint configuration for the fingers to have the middle finger up.

        Args:
            factor (float): 1 = completely up, 0 = completely down
        """
        pass

    def get_middle_down_configuration(self, factor=1.):
        """
        Return the joint configuration for the fingers to have the middle finger up.

        Args:
            factor (float): 1 = completely down, 0 = completely up
        """
        return self.get_middle_up_configuration(factor=1.-factor)

    def get_ring_up_configuration(self, factor=1.):
        """
        Return the joint configuration for the fingers to have the ring finger up.

        Args:
            factor (float): 1 = completely up, 0 = completely down
        """
        pass

    def get_ring_down_configuration(self, factor=1.):
        """
        Return the joint configuration for the fingers to have the ring finger up.

        Args:
            factor (float): 1 = completely down, 0 = completely up
        """
        return self.get_ring_up_configuration(factor=1.-factor)

    def get_pinky_up_configuration(self,  factor=1.):
        """
        Return the joint configuration for the fingers to have the pinky finger up.

        Args:
            factor (float): 1 = completely up, 0 = completely down
        """
        pass

    def get_pinky_down_configuration(self,  factor=1.):
        """
        Return the joint configuration for the fingers to have the pinky finger up.

        Args:
            factor (float): 1 = completely down, 0 = completely up
        """
        return self.get_pinky_up_configuration(factor=1.-factor)

    def get_ok_configuration(self):
        """Return the joint configuration for the fingers to perform the OK or ring gesture.

        References:
            - https://en.wikipedia.org/wiki/OK_gesture
        """
        pass


class TwoHand(Hand):
    r"""Two hand end-effectors

    """

    def __init__(self, simulator, urdf, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.):
        """
        Initialize the two hands robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(TwoHand, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

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

    def get_left_fingers(self, finger_id=None):
        """Return the specified left fingers"""
        if finger_id:
            if isinstance(finger_id, int):
                return self.fingers[self.left_fingers[finger_id]]
            elif isinstance(finger_id, (tuple, list)):
                return [self.fingers[self.left_fingers[finger]] for finger in finger_id]
        return [self.fingers[finger] for finger in self.left_fingers]

    def get_right_fingers(self, finger_id=None):
        """Return the specified right fingers"""
        if finger_id:
            if isinstance(finger_id, int):
                return self.fingers[self.right_fingers[finger_id]]
            elif isinstance(finger_id, (tuple, list)):
                return [self.fingers[self.right_fingers[finger]] for finger in finger_id]
        return [self.fingers[finger] for finger in self.right_fingers]
