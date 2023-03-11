#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the postural (velocity) task.

The postural task tries to bring the robot to a reference posture; that is, it minimizes the joint velocities such
that it gets close to the specified posture (given by the desired joint positions and velocities):

.. math:: || \dot{q} - (K_p (q_d - q) + \dot{q}_d) ||^2,

which is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=I`, :math:`x=\dot{q}`,
and :math:`b = K_p (q_d - q) + \dot{q}_d`, where :math:`K_p` is the stiffness gain and the subscript :math:`d`
means "desired".

Note that specifying the joint positions is optional, you can only specify the joint velocities if you wish.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointVelocityTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Alessio Rocchi (C++)", "Enrico Mingo Hoffman (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PosturalTask(JointVelocityTask):
    r"""Postural Task

    The postural task tries to bring the robot to a reference posture; that is, it minimizes the joint velocities such
    that it gets close to the specified posture (given by the desired joint positions and velocities):

    .. math:: || \dot{q} - (K_p (q_d - q) + \dot{q}_d) ||^2,

    which is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting :math:`A=I`, :math:`x=\dot{q}`,
    and :math:`b = K_p (q_d - q) + \dot{q}_d`, where :math:`K_p` is the stiffness gain and the subscript :math:`d`
    means "desired".

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, q_desired=None, dq_desired=None, kp=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            q_desired (np.array[float[N]], None): desired joint positions, where :math:`N` is the number of DoFs. If
              None, it will not be considered.
            dq_desired (np.array[float[N]], None): desired joint velocities, where :math:`N` is the number of DoFs.
              If None, it will be set to 0.
            kp (float, np.array[float[N,N]]): stiffness gain.
            weight (float, np.array[float[N,N]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(PosturalTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define variables
        self.kp = kp

        # define desired references
        self.q_desired = q_desired
        self.dq_desired = dq_desired

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def q_desired(self):
        """Get the desired joint positions."""
        return self._q_d

    @q_desired.setter
    def q_desired(self, q_d):
        """Set the desired joint positions."""
        if q_d is not None:
            if not isinstance(q_d, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given desired joint positions to be an instance of np.array, instead "
                                "got: {}".format(type(q_d)))
            q_d = np.asarray(q_d)
            if len(q_d) != self.x_size:
                raise ValueError("Expecting the length of the given desired joint positions (={}) to be the same as "
                                 "the number of DoFs (={})".format(len(q_d), self.x_size))
        self._q_d = q_d

    @property
    def dq_desired(self):
        """Get the desired joint velocities."""
        return self._dq_d

    @dq_desired.setter
    def dq_desired(self, dq_d):
        """Set the desired joint velocities."""
        if dq_d is None:
            dq_d = np.zeros(self.x_size)
        if not isinstance(dq_d, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired joint velocities to be an instance of np.array, instead "
                            "got: {}".format(type(dq_d)))
        dq_d = np.asarray(dq_d)
        if len(dq_d) != self.x_size:
            raise ValueError("Expecting the length of the given desired joint velocities (={}) to be the same as the "
                             "number of DoFs (={})".format(len(dq_d), self.x_size))
        self._dq_d = dq_d

    @property
    def x_desired(self):
        """Get the desired joint positions."""
        return self._q_d

    @x_desired.setter
    def x_desired(self, q_d):
        """Set the desired joint positions."""
        self.q_desired = q_d

    @property
    def dx_desired(self):
        """Get the desired joint velocities."""
        return self._dq_d

    @dx_desired.setter
    def dx_desired(self, dq_d):
        """Set the desired joint velocities."""
        self.dq_desired = dq_d

    @property
    def kp(self):
        """Return the stiffness gain."""
        return self._kp

    @kp.setter
    def kp(self, kp):
        """Set the stiffness gain."""
        if kp is None:
            kp = 1.
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given stiffness gain kp to be an int, float, np.array, instead got: "
                            "{}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (self.x_size, self.x_size):
            raise ValueError("Expecting the given stiffness gain matrix kp to be of shape {}, but instead got "
                             "shape: {}".format((self.x_size, self.x_size), kp.shape))
        self._kp = kp

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, dx_des=None, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[float[N]], None): desired joint positions, where :math:`N` is the number of DoFs. If None,
              it will be set to 0.
            dx_des (np.array[float[N]], None): desired joint velocities, where :math:`N` is the number of DoFs. If
              None, it will be set to 0.
        """
        self.x_desired = x_des
        self.dx_desired = dx_des

    def get_desired_references(self):
        """Return the desired references.

        Returns:
            np.array[float[N]]: desired joint positions.
            np.array[float[N]]: desired joint velocities.
        """
        return self.x_desired, self.dx_desired

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        q = self.model.get_joint_positions()

        # update b vector
        if self._q_d is None:
            self._b = self._dq_d
        else:
            self._b = np.dot(self.kp, (self._q_d - q)) + self._dq_d  # shape: (N,)
