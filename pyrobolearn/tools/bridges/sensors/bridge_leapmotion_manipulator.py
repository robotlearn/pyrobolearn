#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the bridge between the LeapMotion and a manipulator's end-effector. Note that this uses a QP controller
behind the scene to move the robot. You can use position or impedance control.
"""

from enum import Enum
import numpy as np

from pyrobolearn.tools.bridges import Bridge
from pyrobolearn.robots import Manipulator
from pyrobolearn.tools.interfaces.sensors.leapmotion import LeapMotionInterface
from pyrobolearn.priorities.models.robot_model import RobotModelInterface
from pyrobolearn.priorities.tasks.velocity import CartesianTask
from pyrobolearn.priorities.tasks.torque import CartesianImpedanceControlTask
from pyrobolearn.priorities.solvers import QPTaskSolver


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ControlType(Enum):
    POSITION = 1
    IMPEDANCE = 2


class Hand(Enum):
    LEFT = 1
    RIGHT = 2


class Position(Enum):
    ABSOLUTE = 1
    RELATIVE = 2


class BridgeLeapMotionManipulator(Bridge):
    r"""Bridge between LeapMotion and a manipulator's end-effector.

    Just move your hand in front of the LeapMotion camera to move the end-effector.
    """

    def __init__(self, manipulator, interface=None, base_link=None, end_effector_link=None, control_type='position',
                 gripper=None, bounding_box=None, use_orientation=True, hand='right', position='relative',
                 priority=None, verbose=False):
        """

        Args:
            manipulator (Manipulator): manipulator robot instance.
            interface (None, LeapMotionInterface): LeapMotion interface. If None, it will create one.
            base_link (int, str): base link id or name.
            end_effector_link (int, str): end effector link id or name.
            control_type (str): type of control to use, select between {'position', 'impedance'}.
            gripper (None, Gripper): gripper robot instance.
            bounding_box (None, np.array[float[2,3]]): bounding box limits.
            use_orientation (bool): if we should account for the orientation of the end-effector as well using the
              interface.
            hand (str): the hand that we will use to move the end-effector, select between {'right', 'left'}.
            position (str): if the position of the hand should be absolute or relative when moving the end-effector.
              If relative, by moving the hand from the center it will compute the distance from it and move the
              end-effector accordingly. Note that the orientation is always absolute here.
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        # set manipulator
        self.manipulator = manipulator
        self.gripper = gripper
        self.base_link = -1 if base_link is None else base_link
        self.distal_link = manipulator.get_end_effector_ids(end_effector=0) if end_effector_link is None \
            else end_effector_link
        self.use_orientation = use_orientation
        self.hand_type = Hand.RIGHT if hand == 'right' else Hand.LEFT
        self.position_type = Position.RELATIVE if position == 'relative' else Position.ABSOLUTE
        self.grasp_strength = 5

        # if interface not defined, create one.
        if interface is None:
            interface = LeapMotionInterface(bounding_box=bounding_box, verbose=verbose)
        if not isinstance(interface, LeapMotionInterface):
            raise TypeError("Expecting the given 'interface' to be an instance of `LeapMotionInterface`, but got "
                            "instead: {}".format(type(interface)))

        # create controller
        model = RobotModelInterface(manipulator)
        x_des = self.manipulator.get_link_positions(link_ids=end_effector_link, wrt_link_id=base_link)
        if control_type == 'position':
            task = CartesianTask(model, distal_link=end_effector_link, base_link=base_link, desired_position=x_des,
                                 kp_position=50.)
            control_type = ControlType.POSITION
        elif control_type == 'impedance':
            task = CartesianImpedanceControlTask(model, distal_link=end_effector_link, base_link=base_link,
                                                 desired_position=x_des, kp_position=100, kd_linear=60)
            control_type = ControlType.IMPEDANCE
        else:
            raise NotImplementedError("Please select between {'position', 'impedance'} for the control_type.")

        self._task = task
        self._control_type = control_type
        self._solver = QPTaskSolver(task=task)

        # call superclass
        super(BridgeLeapMotionManipulator, self).__init__(interface, priority, verbose=verbose)

    ##############
    # Properties #
    ##############

    @property
    def manipulator(self):
        """Return the manipulator instance."""
        return self._manipulator

    @manipulator.setter
    def manipulator(self, manipulator):
        """Set the manipulator instance."""
        if not isinstance(manipulator, Manipulator):
            raise TypeError("Expecting the given 'manipulator' to be an instance of `Manipulator`, instead got: "
                            "{}".format(type(manipulator)))
        self._manipulator = manipulator

    @property
    def gripper(self):
        """Return the gripper instance."""
        return self._gripper

    @gripper.setter
    def gripper(self, gripper):
        """Set the gripper instance."""
        if not isinstance(gripper, Gripper):
            raise TypeError("Expecting the given 'gripper' to be an instance of `Gripper`, instead got: "
                            "{}".format(type(gripper)))
        self._gripper = gripper

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self._manipulator.simulator

    @property
    def solver(self):
        """Return the QP solver."""
        return self._solver

    @property
    def task(self):
        """Return the priority task."""
        return self._task

    ###########
    # Methods #
    ###########

    def step(self, update_interface=False):
        """Perform a step: map the LeapMotion interface to the end-effector."""
        # update interface
        if update_interface:
            self.interface()

        hand = None
        if self.hand_type == Hand.RIGHT:
            hand = self.interface.right_hand
        elif self.hand_type == Hand.LEFT:
            hand = self.interface.left_hand

        if hand is not None:
            # update the QP task desired position
            # position = self.manipulator.get_link_positions(link_ids=self.distal_link, wrt_link_id=self.base_link)
            # position += self.interface.translation
            self.task.desired_position = self.interface.get_hand_stable_position(hand)

            # update the QP task desired orientation (if specified)
            if self.use_orientation:
                # orientation = self.manipulator.get_link_orientations(link_ids=self.distal_link,
                #                                                      wrt_link_id=self.base_link)
                # orientation = get_rpy_from_quaternion(orientation)
                self.task.desired_orientation = self.interface.get_hand_quaternion(hand)

            # update the QP task
            self.task.update(update_model=True)

            # solve QP and set the joint variables
            if self._control_type == ControlType.POSITION:  # Position control
                # solve QP task
                dq = self.solver.solve()

                # set joint positions
                q = self.manipulator.get_joint_positions()
                q = q + dq * self.simulator.dt
                self.manipulator.set_joint_positions(q)

            else:  # Impedance control
                # solve QP task
                torques = self.solver.solve()

                # set joint torques
                self.manipulator.set_joint_torques(torques)

            # check if gripper
            if self.gripper is not None:
                # check if button is pressed
                if self.interface.left_button_pressed:  # open the gripper
                    if self._control_type == ControlType.POSITION:  # position control
                        self.gripper.open(factor=1)
                    else:  # impedance control
                        self.gripper.grasp(strength=self.grasp_strength)
                elif self.interface.right_button_pressed:  # close the gripper
                    if self._control_type == ControlType.POSITION:  # position control
                        self.gripper.close(factor=1)
                    else:  # impedance control
                        self.gripper.grasp(strength=-self.grasp_strength)

