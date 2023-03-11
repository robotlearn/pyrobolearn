# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the bridge between the mouse+keyboard interface and a manipulator's end-effector. Note that this uses a QP
controller behind the scene to move the robot. You can use position or impedance control.
"""

from enum import Enum
import numpy as np

from pyrobolearn.tools.bridges import Bridge
from pyrobolearn.robots import Manipulator, Gripper
from pyrobolearn.utils.transformation import get_rpy_from_quaternion, get_quaternion_from_rpy
from pyrobolearn.tools.interfaces.mouse_keyboard.mousekeyboard import MouseKeyboardInterface
from pyrobolearn.priorities.models.robot_model import RobotModelInterface
from pyrobolearn.priorities.tasks.velocity import CartesianTask
from pyrobolearn.priorities.tasks.torque import CartesianImpedanceControlTask
from pyrobolearn.priorities.solvers import QPTaskSolver


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ControlType(Enum):
    POSITION = 1
    IMPEDANCE = 2


class BridgeMouseKeyboardManipulator(Bridge):
    r"""Bridge between a mouse+keyboard and a manipulator's end-effector.

    Here is the mapping between the interface and the end-effector:
    - left and right arrows: move on the y axis in the world.
    - up and down arrows: move on the x axis in the world.
    - shift and enter: move on the z axis in the world.
    - ctrl + left/right arrow: move around the z axis (yaw)
    - ctrl + up/down arrow: move around the y axis (pitch)
    - ctrl + shift/enter: move around the x axis (roll)
    - numpad 0: open the gripper.
    - numpad 1: close the gripper.
    """

    def __init__(self, manipulator, interface=None, base_link=None, end_effector_link=None, control_type='position',
                 gripper=None, translation_step=0.001, rotation_step=0.001, use_orientation=False,
                 priority=None, verbose=False):
        """
        Initialize the bridge between the mouse+keyboard interface and a manipulator's end-effector.

        Args:
            manipulator (Manipulator): manipulator robot instance.
            interface (None, SpaceMouseInterface): SpaceMouse interface. If None, it will create one.
            base_link (int, str): base link id or name. If None, it will be set to -1 (the base)
            end_effector_link (int, str): end effector link id or name. If None, it will take the first end-effector.
            control_type (str): type of control to use, select between {'position', 'impedance'}.
            gripper (None, Gripper): gripper robot instance.
            translation_step (float): translation step value. As long as we keep pushing on the corresponding buttons,
              it will increment the positions by that step value.
            rotation_step (float): rotation step value. As long as we keep pushing on the corresponding buttons, it
              will increment the angles by that step value.
            use_orientation (bool): if we should account for the orientation of the end-effector as well using the
              interface.
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
        self.grasp_strength = 0

        self.translation_step = translation_step
        self.rotation_step = rotation_step

        # if interface not defined, create one.
        if not isinstance(interface, MouseKeyboardInterface):
            interface = MouseKeyboardInterface(self.simulator)

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
        super(BridgeMouseKeyboardManipulator, self).__init__(interface, priority, verbose=verbose)

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
        """Perform a step: map the SpaceMouse interface to the end-effector."""
        # update interface
        if update_interface:
            self.interface()

        # get interface values
        key, pressed, down = self.interface.key, self.interface.key_pressed, self.interface.key_down

        # update the QP task desired position
        translation = np.zeros(3)
        if key.ctrl not in pressed:
            if key.top_arrow in down:
                translation[0] = self.translation_step
            elif key.bottom_arrow in down:
                translation[0] = -self.translation_step
            elif key.left_arrow in down:
                translation[1] = self.translation_step
            elif key.right_arrow in down:
                translation[1] = -self.translation_step
            elif key.enter in down:
                translation[2] = self.translation_step
            elif key.shift in down:
                translation[2] = -self.translation_step

        position = self.manipulator.get_link_positions(link_ids=self.distal_link, wrt_link_id=self.base_link)
        position += translation
        self.task.desired_position = position

        # update the QP task desired orientation (if specified)
        if self.use_orientation:
            drpy = np.zeros(3)
            if key.ctrl in pressed:
                if key.top_arrow in down:
                    drpy[1] = self.rotation_step
                elif key.bottom_arrow in down:
                    drpy[1] = -self.rotation_step
                elif key.left_arrow in down:
                    drpy[2] = self.rotation_step
                elif key.right_arrow in down:
                    drpy[2] = -self.rotation_step
                elif key.enter in down:
                    drpy[0] = self.rotation_step
                elif key.shift in down:
                    drpy[0] = -self.rotation_step
            orientation = self.manipulator.get_link_orientations(link_ids=self.distal_link, wrt_link_id=self.base_link)
            orientation = get_rpy_from_quaternion(orientation)
            orientation += drpy
            self.task.desired_orientation = get_quaternion_from_rpy(orientation)

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
            if key.n0 in down:  # open the gripper
                if self._control_type == ControlType.POSITION:  # position control
                    self.gripper.open(factor=1)
                else:  # impedance control
                    self.grasp_strength += 0.1
                    self.gripper.grasp(strength=self.grasp_strength)
            elif key.n1 in down:  # close the gripper
                if self._control_type == ControlType.POSITION:  # position control
                    self.gripper.close(factor=1)
                else:  # impedance control
                    self.grasp_strength -= 0.1
                    self.gripper.grasp(strength=self.grasp_strength)
