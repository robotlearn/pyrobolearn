#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Leap Motion hand tracking sensor input interface.

You can run the Leap control panel by typing in the terminal:
$ LeapControlPanel

If you need to restart the daemon (if necessary), just run:
$ sudo service leapd restart

References:
    - Leap Motion: https://www.leapmotion.com/
    - Installation (Ubuntu): https://www.leapmotion.com/setup/desktop/linux/
    - V2 tracking toolkit (SDK): https://developer.leapmotion.com/sdk/v2
    - Python API documentation: https://developer-archive.leapmotion.com/documentation/python/index.html
"""

import time
import numpy as np

try:
    import Leap
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `Leap`, see the `install_leapmotion_ubuntu.txt` file.')


from pyrobolearn.tools.interfaces.sensors import SensorInterface
from pyrobolearn.utils.transformation import get_quaternion_from_matrix, get_rpy_from_matrix


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LeapMotionInterface(SensorInterface):
    r"""Leap Motion Interface

    This class implements the Leap motion interface. Note that the right-handed Cartesian coordinate system used by
    the Leap Motion is different from the one used in robotics (see [5]).

    In robotics, the x-axis points forward, the y-axis points to the left, and the z-axis points upward.
    In the Leap's coordinate system (when the Leap motion is placed in front of the user on a table), the x-axis
    points to the right, the y-axis points upward, and the z-axis points toward the user.

    Also, the physical quantities measured by the Leap motion are in the following units:

    - Distance: millimeter
    - Time: 	microseconds (unless otherwise noted)
    - Speed: 	millimeter/second
    - Angle: 	radians

    In this class, the returned physical quantities accessed through the given methods, are converted to the standard
    units (meter, second, radian), and to the coordinate system used in robotics.

    Part of the documentation for the methods has been copied-pasted from [4] for completeness purposes.

    References:
        - [1] Leap Motion: https://www.leapmotion.com/
        - [2] Installation (Ubuntu): https://www.leapmotion.com/setup/desktop/linux/
        - [3] V2 tracking toolkit (SDK): https://developer.leapmotion.com/sdk/v2
        - [4] Python API documentation: https://developer-archive.leapmotion.com/documentation/python/index.html
        - [5] Leap motion - Coordinate systems:
          https://developer-archive.leapmotion.com/documentation/python/devguide/Leap_Coordinate_Mapping.html
    """

    def __init__(self, bounding_box=None, use_thread=False, sleep_dt=0., verbose=False):
        """
        Initialize the Leap motion input interface.

        Args:
            bounding_box (None, np.array[float[2,3]]): bounding box limits.
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring or
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """

        # create Leap controller
        self.controller = Leap.Controller()

        # wait until the controller connects to the daemon
        while not self.controller.is_connected:
            if verbose:
                print("Waiting the Leap controller to connect to the Leap Daemon...")
            time.sleep(0.01)

        if verbose:
            print("The Leap controller is now connected.")

        # get the first frame
        self._frame = self.controller.frame()
        self._interaction_box = self._frame.interaction_box
        self._left_hand = None
        self._right_hand = None

        super(LeapMotionInterface, self).__init__(use_thread, sleep_dt, verbose)

    ##############
    # Properties #
    ##############

    @property
    def frame(self):
        """Return the last frame."""
        return self._frame

    @property
    def interaction_box(self):
        """Return the last interaction box."""
        return self._interaction_box

    @property
    def hands(self):
        """Return the list of hands."""
        return self.frame.hands

    @property
    def left_hand(self):
        """Return the left hand."""
        return self._left_hand

    @property
    def leftmost_hand(self):
        """Return the left most hand."""
        return self.hands.leftmost

    @property
    def right_hand(self):
        """Return the right hand."""
        return self._right_hand

    @property
    def rightmost_hand(self):
        """Return the right most hand."""
        return self.hands.rightmost

    ###########
    # Methods #
    ###########

    @staticmethod
    def get_hand_direction(hand):
        """
        Get the direction vector; the direction from the palm position toward the fingers.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: hand direction (unit vector)
        """
        d = hand.direction
        return np.array([-d.z, -d.x, d.y])

    @staticmethod
    def get_hand_palm_normal(hand):
        """
        Get the normal vector to the palm. If your hand is flat, this vector will point downward, or 'out' of the
        front surface of your palm.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: palm normal (unit vector)
        """
        d = hand.palm_normal
        return np.array([-d.z, -d.x, d.y])

    @staticmethod
    def get_hand_rotation_matrix(hand):
        """
        Get the hand orientation as a rotation matrix.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3,3]]: rotation matrix.
        """
        basis = hand.basis
        x_basis = basis.x_basis.to_float_array()
        y_basis = basis.y_basis.to_float_array()
        z_basis = basis.z_basis.to_float_array()
        return np.array([-z_basis, -x_basis, y_basis])

    @staticmethod
    def get_hand_quaternion(hand):
        """
        Get the hand orientation as a quaternion [x,y,z,w].

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[4]]: quaternion [x,y,z,w]
        """
        return get_quaternion_from_matrix(LeapMotionInterface.get_hand_rotation_matrix(hand))

    @staticmethod
    def get_hand_rpy(hand):
        """
        Get the hand orientation as roll-pitch-yaw angles (in radians).

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: roll-pitch-yaw angles (in radians)
        """
        return get_rpy_from_matrix(LeapMotionInterface.get_hand_rotation_matrix(hand))

    @staticmethod
    def get_hand_position(hand):
        """
        Get the hand position (center position of the palm) in meter from the Leap Motion Controller origin.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: position (in meter)
        """
        d = hand.palm_position
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_hand_stable_position(hand):
        """
        Get the stabilized hand position (center position of the palm) in meter.

        Smoothing and stabilization is performed in order to make this value more suitable for interaction with 2D
        content. The stabilized position lags behind the palm position by a variable amount, depending primarily on
        the speed of movement.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: stabilized hand position (in meter)
        """
        d = hand.stabilized_palm_position
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_hand_homogeneous_transform(hand):
        """
        Get the homogeneous transformation matrix of the hand.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[4,4]: homogeneous transformation matrix
        """
        rotation = LeapMotionInterface.get_hand_rotation_matrix(hand)
        position = LeapMotionInterface.get_hand_position(hand)
        return np.vstack((np.hstack((rotation, position.reshape(-1, 1))),
                          np.array([0., 0., 0., 1.])))

    @staticmethod
    def get_wrist_position(hand):
        """
        Get the wrist position (in meter) associated with the given hand.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: wrist position (in meter)
        """
        d = hand.wrist_position
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_hand_velocity(hand):
        """
        Get the hand velocity in meter.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: linear velocity (in meter/second)
        """
        d = hand.palm_velocity
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def is_left_hand(hand):
        """
        Return True if the given hand is a left hand.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            bool: True if left hand
        """
        return hand.is_left

    @staticmethod
    def is_right_hand(hand):
        """
        Return True if the given hand is a right hand.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            bool: True if right hand
        """
        return hand.is_right

    @staticmethod
    def get_hand_confidence(hand):
        """
        Return the hand confidence; how well the internal hand model fits the observed data.
        A low value indicates that there are significant discrepancies; finger positions, even hand identification
        could be incorrect. The significance of the confidence value to your application can vary with context.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            float: hand confidence
        """
        return hand.confidence

    @staticmethod
    def get_hand_grab_strength(hand):
        """
        Get the strength of a grab.

        The strength of a grab hand pose as a value in the range [0..1]. An open hand has a grab strength of zero. As
        a hand closes into a fist, its grab strength increases to one.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            float: value in [0..1].
        """
        return hand.grab_strength

    @staticmethod
    def get_hand_pinch_strength(hand):
        """
        Get the strength of a pinch.

        The strength of a pinch pose between the thumb and the closest finger tip as a value in the range [0..1]. An
        open, flat hand has a grab strength of zero. As the tip of the thumb approaches the tip of a finger, the pinch
        strength increases to one.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            float: value in [0..1]
        """
        return hand.pinch_strength

    @staticmethod
    def get_hand_sphere_center(hand):
        """
        Get the center of a sphere fit to the curvature of the given hand. The sphere is placed roughly as if the hand
        were holding a ball. Thus the size of the sphere decreases as the fingers are curled into a fist.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: position of the center of the "hold" sphere (in meter)
        """
        d = hand.sphere_center
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_hand_sphere_radius(hand):
        """
        Get the radius of a sphere fit to the curvature of the given hand. This sphere is placed roughly as if the
        hand were holding a ball. Thus the size of the sphere decreases as the fingers are curled into a fist.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            float: sphere radius.
        """
        return hand.sphere_radius / 1000.

    @staticmethod
    def get_hand_palm_width(hand):
        """
        Get the hand palm width; the average width of the hand (not including fingers or thumb)

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            float: palm width (in meter)
        """
        return hand.palm_width / 1000.

    @staticmethod
    def get_fingers(hand):
        """
        Get the list of fingers associated with the given hand.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            Leap.FingerList: list of finger objects (given in arbitrary order).
        """
        return hand.fingers

    @staticmethod
    def get_finger(hand, idx=0):
        """
        Get the `idx`th finger from the hand.

        Args:
            hand (Leap.Hand): hand instance.
            idx (int): index of the finger. 0 is for the thumb, 1 for the index, 2 for the middle, 3 for the ring, and
                4 for the pinky.

        Returns:
            Leap.Finger: finger instance.
        """
        f = Leap.Finger
        f = {0: f.TYPE_THUMB, 1: f.TYPE_INDEX, 2: f.TYPE_MIDDLE, 3: f.TYPE_RING, 4: f.TYPE_PINKY}
        for finger in hand.fingers:
            if finger.type == f[idx]:
                return finger

    @staticmethod
    def get_finger_type(finger):
        """
        Get the finger type which is between {'thumb', 'index', 'middle', 'ring', 'pinky'}.

        Args:
            finger (Leap.Finger): finger instance.

        Returns:
            str: finger type
        """
        if finger.type == finger.TYPE_THUMB:
            return 'thumb'
        if finger.type == finger.TYPE_INDEX:
            return 'index'
        if finger.type == finger.TYPE_MIDDLE:
            return 'middle'
        if finger.type == finger.TYPE_RING:
            return 'ring'
        if finger.type == finger.TYPE_PINKY:
            return 'pinky'

    @staticmethod
    def get_arm(hand):
        """
        Get the arm associated with the hand.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            Arm: arm instance.
        """
        return hand.arm

    @staticmethod
    def get_elbow_position(hand_or_arm):
        """
        Get the elbow position (in meter) associated to the given hand or arm.

        Args:
            hand_or_arm (Leap.Hand, Leap.Arm): hand or arm instance.

        Returns:
            np.array[float[3]]: elbow position (in meter)
        """
        if isinstance(hand_or_arm, Leap.Hand):
            d = hand_or_arm.arm.elbow_position
        elif isinstance(hand_or_arm, Leap.Arm):
            d = hand_or_arm.elbow_position
        else:
            raise TypeError("Expecting the given parameter to be an instance of `Leap.Hand` or `Leap.Arm`, instead "
                            "got: {}".format(type(hand_or_arm)))
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_finger_tip_position(finger):
        """
        Get the finger tip position (in meter).

        Args:
            finger (Leap.Finger): finger instance.

        Returns:
            np.array[float[3]]: finger tip position (in meter)
        """
        d = finger.tip_position
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_finger_tip_stable_position(finger):
        """
        Get the finger filtered and stabilized position (in meter) using velocity and past positions.

        Args:
            finger (Leap.Finger): finger instance.

        Returns:
            np.array[float[3]]: finger tip stabilized position
        """
        d = finger.stabilized_tip_position
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_finger_tip_velocity(finger):
        """
        Get the finger tip velocity (in meter/second).
        
        Args:
            finger (Leap.Finger): finger instance.

        Returns:
            np.array[float[3]]: finger tip velocity (in meter/second)
        """
        d = finger.tip_velocity
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_finger_direction(finger):
        """
        Get the finger direction.

        Args:
            finger (Leap.Finger): finger instance.

        Returns:
            np.array[float[3]]: finger direction (unit vector)
        """
        d = finger.direction
        return np.array([-d.z, -d.x, d.y])

    @staticmethod
    def get_finger_length(finger):
        """
        Get the finger length (in meter).

        Args:
            finger (Leap.Finger): finger instance.

        Returns:
            np.array[float[3]]: finger length (in meter)
        """
        return finger.length / 1000.

    @staticmethod
    def get_finger_width(finger):
        """
        Get the finger width (in meter).

        Args:
            finger (Leap.Finger): finger instance.

        Returns:
            np.array[float[3]]: finger witdh (in meter)
        """
        return finger.width / 1000.

    @staticmethod
    def get_interaction_box(frame):
        """
        Get the interaction box from the specified frame.

        Args:
            frame (Leap.Frame): frame instance.

        Returns:
            Leap.InteractionBox: interaction box
        """
        return frame.interaction_box

    @staticmethod
    def get_interaction_box_center(box):
        """
        Get the interaction box center position (in meter).

        Args:
            box (Leap.InteractionBox): interaction box instance.

        Returns:
            np.array[float[3]]: interaction box center (in meter).
        """
        d = box.center
        return np.array([-d.z, -d.x, d.y]) / 1000.

    @staticmethod
    def get_interaction_box_depth(box):
        """
        Get the interaction box depth (in meter) measured along the x-axis.

        Args:
            box (Leap.InteractionBox): interaction box instance.

        Returns:
            float: depth (in meter)
        """
        return box.depth

    @staticmethod
    def get_interaction_box_height(box):
        """
        Get the interaction box height (in meter) measured along the z-axis.

        Args:
            box (Leap.InteractionBox): interaction box instance.

        Returns:
            float: height (in meter)
        """
        return box.height

    @staticmethod
    def get_interaction_box_width(box):
        """
        Get the interaction box width (in meter) measured along the y-axis.

        Args:
            box (Leap.InteractionBox): interaction box instance.

        Returns:
            float: width (in meter)
        """
        return box.width

    @staticmethod
    def get_interaction_box_dimensions(box):
        """
        Get the interaction box dimensions (depth, width, height) in meter along the (x, y, z) axis.

        Args:
            box (Leap.InteractionBox): interaction box instance.

        Returns:
            np.array[float[3]]: box dimensions (depth, width, height)
        """
        return np.array([box.depth, box.width, box.height])

    def normalize_point(self, position, clamp=True):
        """
        Normalize the given position such that the position is in the range of [0..1].

        Args:
            position (np.array[float[3]]): position to normalized.
            clamp (bool): Whether or not to limit the output value to the range [0,1] when the input position is
                outside the InteractionBox. Defaults to True.

        Returns:
            np.array[float[3]]: normalized position
        """
        pass

    def transform_normalized_position(self, position, clamp=True, ranges=None):
        """
        Transform the normalized position.

        Args:
            position (np.array[float[3]]): normalized position.
            clamp (bool): Whether or not to limit the output value to the range [0,1] when the input position is
                outside the InteractionBox. Defaults to True.
            ranges (None, np.array[float[2,3]]):

        Returns:
            np.array[float[3]]: transformed position.
        """
        # normalize the position
        position = self.normalize_point(position, clamp=clamp)

    def transform_position(self, position, clamp=True, ranges=None):
        """
        Transform the position

        Args:
            position (np.array[float[3]]): normalized position.
            clamp (bool): Whether or not to limit the output value to the range [0,1] when the input position is
                outside the InteractionBox. Defaults to True.
            ranges (None, np.array[float[2,3]]):

        Returns:
            np.array[float[3]]: transformed position.
        """
        pass

    def get_hand_transformed_position(self, hand):
        """
        Get the hand transformed position (in meter) in the application coordinate system.

        Args:
            hand (Leap.Hand): hand instance.

        Returns:
            np.array[float[3]]: hand transformed position (in meter)
        """
        pass

    @staticmethod
    def get_left_image(frame):
        """
        Get the left image.

        Args:
            frame (Leap.Frame): frame instance.

        Returns:
            np.array[uint8[H,W]]: image.
        """
        image = frame.images[0]
        height, width = image.height, image.width
        return np.frombuffer(image.data, dtype=np.uint8).reshape(height, width)

    @staticmethod
    def get_right_image(frame):
        """
        Get the right image.

        Args:
            frame (Leap.Frame): frame instance.

        Returns:
            np.array[uint8[H,W]]: image
        """
        image = frame.images[1]
        height, width = image.height, image.width
        return np.frombuffer(image.data, dtype=np.uint8).reshape(height, width)

    def run(self):
        """Run the interface."""
        if self.controller.is_connected:
            # get the last frame from the Leap motion controller
            self._frame = self.controller.frame()

            # get the last interaction box
            # self._interaction_box = self._frame.interaction_box

            # get the left and right hand
            for hand in self.frame.hands:
                if hand.is_left:
                    self._left_hand = hand
                elif hand.is_right:
                    self._right_hand = hand


# Test the interface
if __name__ == '__main__':

    import time

    # create the myo interface
    leap = LeapMotionInterface(verbose=True)

    try:
        while True:
            leap.step()
            print("")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        leap.close()
        print("Bye!")
