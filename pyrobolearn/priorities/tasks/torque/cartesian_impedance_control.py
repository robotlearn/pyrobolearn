#!/usr/bin/env python
r"""Provide the cartesian impedance control task.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointTorqueTask
from pyrobolearn.utils.transformation import quaternion_error


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CartesianImpedanceControlTask(JointTorqueTask):
    r"""Cartesian Impedance Control Task

    The cartesian impedance control task optimizes the joint torques such that it applies the necessary torques to
    move a distal link with respect to a bas e

    .. math:: || J(q) H(q)^{-1} \tau - J(q) H(q)^{-1} J(q)^\top f ||^2 = || J(q) H(q)^{-1} (\tau - J(q)^\top f) ||^2

    where :math:`J(q) \in \mathbb{R}^{6 \times N}` is the Jacobian matrix, :math:`H(q) \in \mathbb{R}^{N \times N}` is
    the joint inertia matrix, :math:`\tau \in \mathbb{R}^N` are the torques being optimized, and
    :math:`f \in \mathbb{R}^6` is the desired wrench computed from:

    .. math:: f = K_p e + K_d (\dot{x}_d - \dot{x})

    where :math:`K_p` and :math:`K_d` are the stiffness and damping gains, :math:`e \in \mathbb{R}^{6}` is the error
    which is the concatenation of the position error given by :math:`e_{p} = (x_d - x)` (with :math:`x_d` being the
    desired pose, and :math:`x` the current pose), and the orientation error given by (if expressed as quaternions
    :math:`o = {s, v}` where :math:`s` is the real scalar part, and :math:`v` is the vector part)
    :math:`e_{o} = s v_d - s_d v - v_d \cross v`, and :math:`\dot{x}_d \in \mathbb{R}^{6}` is the desired cartesian
    velocity for the distal link with respect to the base link.

    The above optimization problem is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
    :math:`A = J(q) H(q)^{-1}`, :math:`x = \tau`, and :math:`b = J(q) H(q)^{-1} J(q)^\top f`.

    Note that :math:`||J(q) H(q)^{-1} (\tau - J(q)^\top f)||^2  \leq  ||J(q) H(q)^{-1}|| ||\tau - J(q)^\top f||^2`.

    .. seealso:: `tasks/velocity/cartesian.py`
    """

    def __init__(self, model, distal_link, base_link=None, local_position=(0, 0, 0), x_desired=None,
                 dx_desired=None, kp=1., kd=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the world.
            local_position (np.array[3]): local position on the distal link.
            x_desired (np.array[7], None): desired cartesian pose of distal link wrt the base.
            dx_desired (np.array[6], None): desired cartesian velocity of distal link wrt the base.
            kp (float, np.array[6,6]): stiffness gain.
            kd (float, np.array[6,6]): damping gain.
            weight (float, np.array[6,6]): weight scalar or matrix associated to the task.
            constraints (list of Constraint): list of constraints associated with the task.
        """
        super(CartesianImpedanceControlTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define variables
        self.distal_link = self.model.get_link_id(distal_link)
        self.base_link = self.model.get_link_id(base_link) if base_link is not None else base_link
        self.local_position = local_position
        self.kp = kp
        self.kd = kd

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
        """Get the desired cartesian pose for the distal link wrt to the base."""
        return self._x_d

    @x_desired.setter
    def x_desired(self, x_d):
        """Get the desired cartesian pose for the distal link wrt to the base."""
        if x_d is None:
            x_d = np.array([0.] * 6 + [1.])
        if not isinstance(x_d, np.ndarray):
            raise TypeError("Expecting the given desired pose to be a np.array, instead got: {}".format(type(x_d)))
        if len(x_d) != 7:
            raise ValueError("Expecting the given desired pose array to be of length 7 (3 for the position, and 4 "
                             "for the orientation expressed as a quaternion [x,y,z,w]), instead got a length of: "
                             "{}".format(len(x_d)))
        self._x_d = x_d

    @property
    def dx_desired(self):
        """Get the desired cartesian velocity for the distal link wrt to the base."""
        return self._dx_d

    @dx_desired.setter
    def dx_desired(self, dx_d):
        """Set the desired cartesian velocity for the distal link wrt to the base."""
        if dx_d is None:
            dx_d = np.zeros(6)
        if not isinstance(dx_d, np.ndarray):
            raise TypeError("Expecting the given desired velocity to be a np.array, instead got: {}".format(type(dx_d)))
        if len(dx_d) != 7:
            raise ValueError("Expecting the given desired velocity array to be of length 6 (3 for the linear and 3 "
                             "for the angular part), instead got a length of: {}".format(len(dx_d)))
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
        if isinstance(kp, np.ndarray) and kp.shape != (6, 6):
            raise ValueError("Expecting the given stiffness gain matrix kp to be of shape {}, but instead got "
                             "shape: {}".format((self.x_size, self.x_size), kp.shape))
        self._kp = kp

    @property
    def kd(self):
        """Return the damping gain."""
        return self._kd

    @kd.setter
    def kd(self, kd):
        """Set the damping gain."""
        if not isinstance(kd, (float, int, np.ndarray)):
            raise TypeError("Expecting the given damping gain kd to be an int, float, np.array, instead got: "
                            "{}".format(type(kd)))
        if isinstance(kd, np.ndarray) and kd.shape != (6, 6):
            raise ValueError("Expecting the given damping gain matrix kd to be of shape {}, but instead got "
                             "shape: {}".format((6, 6), kd.shape))
        self._kd = kd

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, dx_des=None, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[7], None): desired cartesian pose (position and quaternion [x,y,z,w]) of distal link wrt
                the base.
            dx_des (np.array[6], None): desired cartesian velocity of distal link wrt the base.
        """
        self.x_desired = x_des
        self.dx_desired = dx_des

    def get_desired_references(self):
        """Return the desired references.

        Returns:
            np.array[7]: desired cartesian pose (position and quaternion [x,y,z,w]) of distal link wrt the base.
            np.array[6]: desired cartesian velocity of distal link wrt the base.
        """
        return self.x_desired, self.dx_desired

    def _update(self):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        # get useful variables
        x = self.model.get_link_pose_wrt(self.distal_link, self.base_link)
        dx = self.model.get_link_velocity(self.distal_link, self.base_link)
        J = self.model.get_jacobian(self.distal_link, self.base_link, self.local_position)  # shape: (6,N)
        H = self.model.get_inertia_matrix()  # shape: (N,N)

        # compute A matrix
        self._A = J.dot(np.linalg.inv(H))  # shape: (6,N)

        # compute position/orientation error
        position_error = (self._x_d[:3] - x[:3])
        orientation_error = quaternion_error(quat_des=self._x_d[3:], quat_cur=x[3:])
        error = np.concatenate((position_error, orientation_error))

        # compute wrench
        f = np.dot(self.kp, error) + np.dot(self.kd, (self._dx_d - dx))  # shape: (6,)

        # compute b vector
        self._b = self._A.dot(J.T).dot(f)
