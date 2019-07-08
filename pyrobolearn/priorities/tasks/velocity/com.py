#!/usr/bin/env python
r"""Provide the center of mass velocity task.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CoMTask(JointVelocityTask):
    r"""Center of Mass Velocity Task

    The CoM task tries to impose a desired position of the CoM with respect to the world frame.

    .. math:: ||J_{CoM} \dot{q} - (K_p (x_d - x) + \dot{x}_d)||^2

    where :math:`J_{CoM}` is the CoM Jacobian, :math:`\dot{q}` are the joint velocities being optimized, :math:`K_p`
    is the stiffness gain, :math:`x_d` and :math:`x` are the desired and current cartesian CoM position
    respectively, and :math:`\dot{x}_d` is the desired linear velocity of the CoM.

    This is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=J_{CoM}`,
    :math:`x=\dot{q}`, and :math:`b = K_p (x_d - x) + \dot{x}_d`.
    """

    def __init__(self, model, x_desired=None, dx_desired=None, kp=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            x_desired (np.array[3], None): desired CoM position. If None, it will be set to 0.
            dx_desired (np.array[3], None): desired CoM linear velocity. If None, it will be set to 0.
            kp (float, np.array[3,3]): stiffness gain.
            weight (float, np.array[3,3]): weight scalar or matrix associated to the task.
            constraints (list of Constraint): list of constraints associated with the task.
        """
        super(CoMTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define variable
        self.kp = kp

        # define desired references
        self.x_desired = x_desired
        self.dx_desired = dx_desired

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def x_desired(self):
        """Get the desired CoM position."""
        return self._x_d

    @x_desired.setter
    def x_desired(self, x_d):
        """Set the desired CoM position."""
        if x_d is None:
            x_d = np.zeros(3)
        if not isinstance(x_d, np.ndarray):
            raise TypeError("Expecting the given desired CoM position to be a np.array, instead got: "
                            "{}".format(type(x_d)))
        if len(x_d) != 3:
            raise ValueError("Expecting the given desired CoM position array to be of length 3, instead got a length "
                             "of: {}".format(len(x_d)))
        self._x_d = x_d

    @property
    def dx_desired(self):
        """Get the desired CoM linear velocity."""
        return self._dx_d

    @dx_desired.setter
    def dx_desired(self, dx_d):
        """Set the desired CoM linear velocity."""
        if dx_d is None:
            dx_d = np.zeros(3)
        if not isinstance(dx_d, np.ndarray):
            raise TypeError("Expecting the given desired CoM linear velocity to be a np.array, instead got: "
                            "{}".format(type(dx_d)))
        if len(dx_d) != 3:
            raise ValueError("Expecting the given desired CoM linear velocity array to be of length 3, instead got a "
                             "length of: {}".format(len(dx_d)))
        self._dx_d = dx_d

    @property
    def kp(self):
        """Return the stiffness gain."""
        return self._kp

    @kp.setter
    def kp(self, kp):
        """Set the stiffness gain."""
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given stiffness gain kp to be an int, float, np.array, instead got: "
                            "{}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (3, 3):
            raise ValueError("Expecting the given stiffness gain matrix kp to be of shape {}, but instead got "
                             "shape: {}".format((self.x_size, self.x_size), kp.shape))
        self._kp = kp

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, dx_des=None, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[3], None): desired CoM position. If None, it will be set to 0.
            dx_des (np.array[3], None): desired CoM linear velocity. If None, it will be set to 0.
        """
        self.x_desired = x_des
        self.dx_desired = dx_des

    def get_desired_references(self):
        """Return the desired references.

        Returns:
            np.array[3]: desired CoM position.
            np.array[3]: desired CoM linear velocity.
        """
        return self.x_desired, self.dx_desired

    def _update(self):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        x = self.model.get_com_position()
        self._A = self.model.get_com_jacobian()  # shape: (3, N)
        self._b = np.dot(self.kp, (self._x_d - x)) + self._dx_d  # shape: (3,)
