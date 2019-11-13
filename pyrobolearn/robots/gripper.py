#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Gripper abstract classes.
"""

from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.robots.robot import Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Gripper(Robot):
    r"""Gripper end-effector
    """

    def __init__(self, simulator, urdf, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.):
        """
        Initialize the gripper.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the URDF file.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the gripper will be fixed in the world.
            scale (float): scaling factor that is used to scale the gripper.
        """
        super(Gripper, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

        self.fingers = []  # list of fingers where each finger is a list of links/joints

    @property
    def num_fingers(self):
        """Return the number of fingers on the hand"""
        return len(self.fingers)

    def get_finger(self, finger_id=None):
        """Return the list of joint/link ids for the specified finger"""
        if finger_id:
            return self.fingers[finger_id]
        return self.fingers

    def open(self, factor):
        """
        Open the gripper.

        Args:
            factor (float): float representing how much the gripper is open (1=completely open, 0=completely close).
        """
        self.close(factor=1.-factor)

    def close(self, factor):
        """
        Close the gripper. This has to be implemented in the child class.

        Args:
            factor (float): float representing how much the gripper is closed (1=completely close, 0=completely open).
        """
        pass

    def grasp(self, strength, point=None, frame=Simulator.LINK_FRAME):
        """
        Grasp an object. Compared to :func:`~open` or :func:`~close` which uses position control, this uses force
        (impedance) control using attractor points. The strength factor increases the stiffness of the grasping; i.e.
        it increases the applied torques in the gripper joints.

        Args:
            strength (float): scalar describing how much to increases the stiffness (the torques that are applied on
              the joint fingers). If positive, it closes the gripper fingers. If negative, it opens the gripper
              fingers.
            point (np.array[float[3]], list of np.array[float[3]], None): attractor point(s) described in the specified
              frame. If multiple points are specified, they have to match the number of fingers and will be used in the
              same order. If None, it will grasp in a "natural" way (which is let to the user that has implemented
              this method).
            frame (int): integer describing if the above given point is described in the world frame
              (``Simulator.WORLD_FRAME``), or in the link frame of the gripper base (``Simulator.LINK_FRAME``).
        """
        pass


class ParallelGripper(Gripper):
    r"""Parallel Gripper

    When the fingers are closing toward each other, they remain parallel. Most of the time, these types of grippers
    have 2 fingers. This type is also known as the slider type.
    """

    def __init__(self, simulator, urdf, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.):
        """
        Initialize the gripper.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the URDF file.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the gripper will be fixed in the world.
            scale (float): scaling factor that is used to scale the gripper.
        """
        super(ParallelGripper, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)


class AngularGripper(Gripper):
    r"""Angular Gripper

    When the fingers are closing toward each other, each finger rotates around an axis. They can have 2, 3, or even
    4 grippers. This type is also known as the lever type.
    """

    def __init__(self, simulator, urdf, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.):
        """
        Initialize the gripper.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the URDF file.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the gripper will be fixed in the world.
            scale (float): scaling factor that is used to scale the gripper.
        """
        super(AngularGripper, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)


class VacuumGripper(Gripper):
    r"""Vacuum Gripper

    A vacuum gripper doesn't have any fingers but use suction in order to grasp different objects. Note that this
    required the simulator to be able to simulate soft bodies and the interaction of these ones with rigid bodies.
    """

    def __init__(self, simulator, urdf, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.):
        """
        Initialize the gripper.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the URDF file.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the gripper will be fixed in the world.
            scale (float): scaling factor that is used to scale the gripper.
        """
        super(VacuumGripper, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
