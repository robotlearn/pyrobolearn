#!/usr/bin/env python
r"""Provide the center of mass force task.

From the documentation of the framework of [1]: "The CoM task computes the wrenches at the contact, in world frame,
in order to realize a certain acceleration and variation of angular momentum at the CoM considering the Centroidal
Dynamics":

.. math::

    m * \ddot{r} = \sum_i f_i + mg \\
    \dot{L} = \sum_i p_i \times f_i + \tau_i,

where :math:`w = [f \tau] \in \mathbb{R}^6` is the wrench vector composed of a force vector
:math:`f \in \mathbb{R}^3` and a torque vector :math:`\tau \in \mathbb{R}^3`, :math:`m` is the mass, :math:`r` is
the CoM position, :math:`g` is the gravity vector, :math:`L` is the angular momentum around the CoM, :math:`p` is
the position vector of where the wrench is applied (with respect to the CoM), and the subscript :math:`i` is to
denote each link where a wrench is applied to it (by contact).

The task can be mathematically described as:

.. math:: || A w - [m (\ddot{r}_{ref} - g)^\top, \dot{L}_{ref}^\top]^\top ||^2

where :math:`A \in \mathbb{R}^{6 \times 6N_c}` (described below in more details), :math:`N_c` is the number of
contact links, :math:`w = [f_1 \tau1 \cdot f_{N_c} \tau_{N_c}] \in \mathbb{R}^{6N_c}` is the wrench vector being
optimized, :math:`m \in \mathbb{R}` is the total mass of the robot, :math:`g \in \mathbb{R}^3` is the gravity
vector, :math:`\ddot{r}_{ref} \in \mathbb{R}^3` is the reference CoM acceleration vector (see below), and
:math:`\dot{L}_{ref} \in \mathbb{R}^3` is the reference variation of the angular momentum around the CoM (see
below).

The reference vectors are given by:

.. math::

    \ddot{r}_{ref} = \ddot{r}_{des} + k_d (\dot{r}_{des} - \dot{r}) + k_p (r_{des} - r) \\
    \dot{L}_{ref} = \dot{L}_{des} + k_d (L_{des} - L)

The matrix :math:`A` is given:

.. math::

    A = \left[ \begin{array}{cc}
        I_{3 \times 3} & 0_{3 \times 3} \\
        S(p_1) & I_{3 \times 3}
        \end{array} \cdot \begin{array}{cc}
        I_{3 \times 3} & 0_{3 \times 3} \\
        S(p_{N_c}) & I_{3 \times 3}
        \end{array} \right]

where :math:`S(p_i)` is the skew-symmetric matrix built from the contact position vector
:math:`p_i \in \mathbb{R}^3`.

The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
:math:`A = A`, :math:`x = w`, and :math:`b = [m (\ddot{r}_{ref} - g)^\top, \dot{L}_{ref}^\top]^\top`.

The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

import numpy as np

from pyrobolearn.priorities.tasks import ForceTask
from pyrobolearn.utils.transformation import skew_matrix


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Enrico Mingo Hoffman (C++)", "Alessio Rocchi (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CoMForceTask(ForceTask):
    r"""CoM Force Task

    From the documentation of the framework of [1]: "The CoM task computes the wrenches at the contact, in world frame,
    in order to realize a certain acceleration and variation of angular momentum at the CoM considering the Centroidal
    Dynamics":

    .. math::

        m * \ddot{r} = \sum_i f_i + mg \\
        \dot{L} = \sum_i p_i \times f_i + \tau_i,

    where :math:`w = [f \tau] \in \mathbb{R}^6` is the wrench vector composed of a force vector
    :math:`f \in \mathbb{R}^3` and a torque vector :math:`\tau \in \mathbb{R}^3`, :math:`m` is the mass, :math:`r` is
    the CoM position, :math:`g` is the gravity vector, :math:`L` is the angular momentum around the CoM, :math:`p` is
    the position vector of where the wrench is applied (with respect to the CoM), and the subscript :math:`i` is to
    denote each link where a wrench is applied to it (by contact).

    The task can be mathematically described as:

    .. math:: || A w - [m (\ddot{r}_{ref} - g)^\top, \dot{L}_{ref}^\top]^\top ||^2

    where :math:`A \in \mathbb{R}^{6 \times 6N_c}` (described below in more details), :math:`N_c` is the number of
    contact links, :math:`w = [f_1 \tau1 \cdot f_{N_c} \tau_{N_c}] \in \mathbb{R}^{6N_c}` is the wrench vector being
    optimized, :math:`m \in \mathbb{R}` is the total mass of the robot, :math:`g \in \mathbb{R}^3` is the gravity
    vector, :math:`\ddot{r}_{ref} \in \mathbb{R}^3` is the reference CoM acceleration vector (see below), and
    :math:`\dot{L}_{ref} \in \mathbb{R}^3` is the reference variation of the angular momentum around the CoM (see
    below).

    The reference vectors are given by:

    .. math::

        \ddot{r}_{ref} = \ddot{r}_{des} + k_d (\dot{r}_{des} - \dot{r}) + k_p (r_{des} - r) \\
        \dot{L}_{ref} = \dot{L}_{des} + k_d (L_{des} - L)

    The matrix :math:`A` is given:

    .. math::

        A = \left[ \begin{array}{cc}
            I_{3 \times 3} & 0_{3 \times 3} \\
            S(p_1) & I_{3 \times 3}
            \end{array} \cdot \begin{array}{cc}
            I_{3 \times 3} & 0_{3 \times 3} \\
            S(p_{N_c}) & I_{3 \times 3}
            \end{array} \right]

    where :math:`S(p_i)` is the skew-symmetric matrix built from the contact position vector
    :math:`p_i \in \mathbb{R}^3`.

    The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
    :math:`A = A`, :math:`x = w`, and :math:`b = [m (\ddot{r}_{ref} - g)^\top, \dot{L}_{ref}^\top]^\top`.

    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model, contact_links=[], desired_acceleration=None, desired_velocity=None,
                 desired_position=None, desired_variation_angular_momentum=None, desired_angular_momentum=None,
                 k_velocity=1., k_position=1., k_angular_momentum=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            contact_links (list[str], list[int]): list of unique contact link names or ids.
            desired_acceleration (np.array[float[3]], None): desired CoM linear acceleration. If None, it will be set
              to 0.
            desired_velocity (np.array[float[3]], None): desired CoM linear velocity. If None, it will be set to 0.
            desired_position (np.array[float[3]], None): desired CoM position. If None, it will not be considered.
            desired_variation_angular_momentum (np.array[float[3]], None): desired CoM variation angular momentum. If
              None, it will be set to 0.
            desired_angular_momentum (np.array[float[3]], None): desired CoM angular momentum. If None, it will be set
              to 0.
            k_velocity (float, np.array[float[3,3]]): CoM velocity gain.
            k_position (float, np.array[float[3,3]]): CoM position gain.
            k_angular_momentum (float, np.array[float[3,3]]): CoM angular momentum.
            weight (float, np.array[float[6,6]]): weight scalar or matrix associated to the task.
            constraints (list[Constraint]): list of constraints associated with the task.
        """
        super(CoMForceTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # set variables
        self.contact_links = contact_links
        self.desired_acceleration = desired_acceleration
        self.desired_velocity = desired_velocity
        self.desired_position = desired_position
        self.desired_variation_angular_momentum = desired_variation_angular_momentum
        self.desired_angular_momentum = desired_angular_momentum

        # set gains
        self.k_velocity = k_velocity
        self.k_position = k_position
        self.k_angular_momentum = k_angular_momentum

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def contact_links(self):
        """Get the contact links."""
        return self._contact_links

    @contact_links.setter
    def contact_links(self, contacts):
        """Set the contact links."""
        if contacts is None:
            contacts = []
        elif not isinstance(contacts, (list, tuple, np.ndarray)):
            raise TypeError("Expecting the given 'contact_links' to be a list of names/ids, but got instead: "
                            "{}".format(type(contacts)))
        self._contact_links = contacts

        # enable / disable the tasks based on the number of contact links
        if len(contacts) == 0:
            self.disable()
        else:
            self.enable()

    @property
    def desired_acceleration(self):
        """Get the desired CoM linear acceleration."""
        return self._a_des

    @desired_acceleration.setter
    def desired_acceleration(self, acceleration):
        """Set the desired CoM linear acceleration."""
        if acceleration is None:
            acceleration = np.zeros(3)
        elif not isinstance(acceleration, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired linear acceleration to be a np.array, instead got: "
                            "{}".format(type(acceleration)))
        acceleration = np.asarray(acceleration)
        if len(acceleration) != 3:
            raise ValueError("Expecting the given desired linear acceleration array to be of length 3, but instead "
                             "got: {}".format(len(acceleration)))
        self._a_des = acceleration

    @property
    def desired_velocity(self):
        """Get the desired CoM linear velocity."""
        return self._v_des

    @desired_velocity.setter
    def desired_velocity(self, velocity):
        """Set the desired CoM linear velocity."""
        if velocity is None:
            velocity = np.zeros(3)
        elif not isinstance(velocity, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired linear velocity to be a np.array, instead got: "
                            "{}".format(type(velocity)))
        velocity = np.asarray(velocity)
        if len(velocity) != 3:
            raise ValueError("Expecting the given desired linear velocity array to be of length 3, but instead "
                             "got: {}".format(len(velocity)))
        self._v_des = velocity

    @property
    def desired_position(self):
        """Get the desired CoM position."""
        return self._x_des

    @desired_position.setter
    def desired_position(self, position):
        """Set the desired CoM position."""
        if position is not None:
            if not isinstance(position, (np.ndarray, list, tuple)):
                raise TypeError("Expecting the given desired position to be a np.array, instead got: "
                                "{}".format(type(position)))
            position = np.asarray(position)
            if len(position) != 3:
                raise ValueError("Expecting the given desired position array to be of length 3, but instead "
                                 "got: {}".format(len(position)))
        self._x_des = position

    @property
    def desired_angular_momentum(self):
        """Get the desired CoM angular momentum."""
        return self._l_des

    @desired_angular_momentum.setter
    def desired_angular_momentum(self, angular_momentum):
        """Set the desired CoM angular momentum."""
        if angular_momentum is None:
            angular_momentum = np.zeros(3)
        elif not isinstance(angular_momentum, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired angular momentum to be a np.array, instead got: "
                            "{}".format(type(angular_momentum)))
        angular_momentum = np.asarray(angular_momentum)
        if len(angular_momentum) != 3:
            raise ValueError("Expecting the given desired angular momentum array to be of length 3, but instead "
                             "got: {}".format(len(angular_momentum)))
        self._l_des = angular_momentum

    @property
    def desired_variation_angular_momentum(self):
        """Get the desired CoM variation angular momentum."""
        return self._dl_des

    @desired_variation_angular_momentum.setter
    def desired_variation_angular_momentum(self, variation_angular_momentum):
        """Set the desired CoM variation angular momentum."""
        if variation_angular_momentum is None:
            variation_angular_momentum = np.zeros(3)
        elif not isinstance(variation_angular_momentum, (np.ndarray, list, tuple)):
            raise TypeError("Expecting the given desired variation angular momentum to be a np.array, instead got: "
                            "{}".format(type(variation_angular_momentum)))
        variation_angular_momentum = np.asarray(variation_angular_momentum)
        if len(variation_angular_momentum) != 3:
            raise ValueError("Expecting the given desired variation angular momentum array to be of length 3, but "
                             "instead got: {}".format(len(variation_angular_momentum)))
        self._dl_des = variation_angular_momentum

    @property
    def k_position(self):
        """Return the position gain."""
        return self._kp

    @k_position.setter
    def k_position(self, k):
        """Set the position gain."""
        if k is None:
            k = 1.
        if not isinstance(k, (float, int, np.ndarray)):
            raise TypeError("Expecting the given position gain to be an int, float, np.array, instead "
                            "got: {}".format(type(k)))
        if isinstance(k, np.ndarray) and k.shape != (3, 3):
            raise ValueError("Expecting the given position gain matrix to be of shape {}, but instead "
                             "got shape: {}".format((3, 3), k.shape))
        self._kp = k

    @property
    def k_velocity(self):
        """Return the velocity gain."""
        return self._kv

    @k_velocity.setter
    def k_velocity(self, k):
        """Set the velocity gain."""
        if k is None:
            k = 1.
        if not isinstance(k, (float, int, np.ndarray)):
            raise TypeError("Expecting the given velocity gain to be an int, float, np.array, instead "
                            "got: {}".format(type(k)))
        if isinstance(k, np.ndarray) and k.shape != (3, 3):
            raise ValueError("Expecting the given velocity gain matrix to be of shape {}, but instead "
                             "got shape: {}".format((3, 3), k.shape))
        self._kv = k

    @property
    def k_angular_momentum(self):
        """Return the angular momentum gain."""
        return self._kl

    @k_angular_momentum.setter
    def k_angular_momentum(self, k):
        """Set the angular momentum gain."""
        if k is None:
            k = 1.
        if not isinstance(k, (float, int, np.ndarray)):
            raise TypeError("Expecting the given angular momentum gain to be an int, float, np.array, instead "
                            "got: {}".format(type(k)))
        if isinstance(k, np.ndarray) and k.shape != (3, 3):
            raise ValueError("Expecting the given angular momentum gain matrix to be of shape {}, but instead "
                             "got shape: {}".format((3, 3), k.shape))
        self._kl = k

    ###########
    # Methods #
    ###########

    def _update(self, x=None):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        x = self.model.get_com_position()  # shape: (3,)
        dx = self.model.get_com_velocity()  # shape: (3,)
        l = self.model.get_centroidal_momentum()[:3]  # shape: (3,)

        # compute reference acceleration
        a_ref = self._a_des + np.dot(self._kv, (self._v_des - dx))  # shape: (3,)
        if self._x_des is not None:
            a_ref += np.dot(self._kp, (self._x_des - x))  # shape: (3,)

        # compute reference variation of the angular momentum
        l_ref = self._dl_des + np.dot(self._kl, (self._l_des - l))  # shape: (3,)

        # compute A matrix
        As = []
        for link in self.contact_links:
            link = self.model.get_link_id(link)
            contact_pos = self.model.get_position(link)
            A = np.vstack((np.hstack((np.identity(3), np.zeros(3))),
                           np.hstack((skew_matrix(contact_pos - x), np.identity(3)))))
            As.append(A)

        # compute A matrix and b vector
        self._A = np.concatenate(As, axis=1)  # (6, 6*Nc)
        self._b = np.concatenate((a_ref, l_ref))  # (6,)
