# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the joint impedance control task.

The joint impedance control task minimizes the specified torques given as a PD control from the desired joint
positions and velocities. That it, it minimizes:

.. math:: || \tau - (K_p (q_d - q) + K_d (\dot{q}_d - \dot{q})) ||^2

where :math:`\tau` are the torques being optimized, :math:`K_p` and :math:`K_d` are the stiffness and damping
gains respectively, :math:`q` and :math:`\dot{q}` are the joint positions and velocities, and the subscript
:math:`d` means 'desired'.

The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
:math:`A = I` (where :math:`I` is the identity matrix), :math:`x = \tau`, and
:math:`b = K_p (q_d - q) + K_d (\dot{q}_d - \dot{q})`.

From [1], "note that "if used in the null-space, it realizes the null-space stiffness as described in [2]".

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "Cartesian Impedance Control of Redundant and Flexible-Joint Robots", Ott, 2008
"""

import numpy as np

from pyrobolearn.priorities.tasks import JointTorqueTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointImpedanceControlTask(JointTorqueTask):
    r"""Joint Impedance Control Task

    The joint impedance control task minimizes the specified torques given as a PD control from the desired joint
    positions and velocities. That it, it minimizes:

    .. math:: || \tau - (K_p (q_d - q) + K_d (\dot{q}_d - \dot{q})) ||^2

    where :math:`\tau` are the torques being optimized, :math:`K_p` and :math:`K_d` are the stiffness and damping
    gains respectively, :math:`q` and :math:`\dot{q}` are the joint positions and velocities, and the subscript
    :math:`d` means 'desired'.

    The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
    :math:`A = I` (where :math:`I` is the identity matrix), :math:`x = \tau`, and
    :math:`b = K_p (q_d - q) + K_d (\dot{q}_d - \dot{q})`.

    From [1], "if used in the null-space, it realizes the null-space stiffness as described in [3]".

    .. seealso:: `tasks/velocity/postural.py`

    References:
        - [1] OpenSoT framework
        - [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
        - [3] "Cartesian Impedance Control of Redundant and Flexible-Joint Robots", Ott, 2008
    """

    def __init__(self, model, q_desired=None, dq_desired=None, kp=1., kd=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            q_desired (np.array[float[N]], None): desired joint positions, where :math:`N` is the number of DoFs. If 
              None, it will be set to 0.
            dq_desired (np.array[float[N]], None): desired joint velocities, where :math:`N` is the number of DoFs. If 
              None, it will be set to 0.
            kp (float, np.array[float[N,N]]): stiffness gain.
            kd (float, np.array[float[N,N]]): damping gain.
            weight (float, np.array[float[N,N]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(JointImpedanceControlTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # define variables
        self.kp = kp
        self.kd = kd

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
        if q_d is None:
            q_d = np.zeros(self.x_size)
        elif not isinstance(q_d, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired joint positions to be an instance of np.array, instead got: "
                            "{}".format(type(q_d)))
        q_d = np.asarray(q_d)
        if len(q_d) != self.x_size:
            raise ValueError("Expecting the length of the given desired joint positions (={}) to be the same as the "
                             "number of DoFs (={})".format(len(q_d), self.x_size))
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
        elif not isinstance(dq_d, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired joint velocities to be an instance of np.array, instead got: "
                            "{}".format(type(dq_d)))
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
        if not isinstance(kp, (float, int, np.ndarray)):
            raise TypeError("Expecting the given stiffness gain kp to be an int, float, np.array, instead got: "
                            "{}".format(type(kp)))
        if isinstance(kp, np.ndarray) and kp.shape != (self.x_size, self.x_size):
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
        if isinstance(kd, np.ndarray) and kd.shape != (self.x_size, self.x_size):
            raise ValueError("Expecting the given damping gain matrix kd to be of shape {}, but instead got "
                             "shape: {}".format((self.x_size, self.x_size), kd.shape))
        self._kd = kd

    ###########
    # Methods #
    ###########

    def set_desired_references(self, x_des, dx_des=None, *args, **kwargs):
        """Set the desired references.

        Args:
            x_des (np.array[float[N]], None): desired joint positions, where :math:`N` is the number of DoFs. If None,
              it will be set to 0.
            dx_des (np.array[float[N]], None): desired joint velocities, where :math:`N` is the number of DoFs. If None,
              it will be set to 0.
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
        dq = self.model.get_joint_velocities()

        # update b vector
        self._b = np.dot(self.kp, self._q_d - q) + np.dot(self.kd, self._dq_d - dq)  # shape: (N,)
