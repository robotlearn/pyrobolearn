#!/usr/bin/env python
r"""RBDL model interface used in priority tasks.

This is based on the implementation in `https://github.com/ADVRHumanoids/ModelInterfaceRBDL`, which is licensed under
the LPGLv3.

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
    - [3] "Rigid Body Dynamics Algorithms", Featherstone, 2008
"""

import numpy as np
import rbdl

from pyrobolearn.priorities.models import ModelInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Arturo Laurenzi (C++)", "Luca Muratore (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RBDLModelInterface(ModelInterface):
    r"""RBDL Model interface.

    """

    def __init__(self, urdf, floating_base=False, verbose=False):
        """
        Initialize the RBDL model interface.

        Args:
            urdf (str): path to the URDF file.
            floating_base (bool): set this variable to True, if we have a floating-based robot.
            verbose (bool): if True, it will print information when loading the URDF.
        """
        # load the RBDL model
        model = rbdl.loadModel(filename=urdf, floating_base=floating_base, verbose=verbose)

        # call parent constructor
        super(RBDLModelInterface, self).__init__(model)

        # define joint attributes
        self.q = np.zeros(self.model.q_size)
        self.dq = np.zeros(self.model.qdot_size)
        self.ddq = np.zeros(self.model.qdot_size)

        self.mass = 0
        self.com = np.zeros(3)
        self.com_vel = np.zeros(3)
        self.com_acc = np.zeros(3)
        self.angular_momentum_com = np.zeros(3)
        self.change_angular_momentum_com = np.zeros(3)

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        """Return the model instance."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the model instance."""
        if not isinstance(model, rbdl.Model):
            raise TypeError("Expectig the given 'model' to be an instance of `rbdl.Model`, instead got: "
                            "{}".format(type(model)))
        self._model = model

    @property
    def num_dof(self):
        """Return the number of degrees of freedom."""
        return self.model.dof_count

    ###########
    # Methods #
    ###########

    def get_com_position(self):
        """
        Get the position of the center of mass (CoM).

        Returns:
            np.array[3]: position of the center of mass
        """
        return rbdl.CalcCenterOfMass(self.model, self.q, self.dq, self.ddq, self.com, self.com_vel, self.com_acc,
                                     self.angular_momentum_com, self.change_angular_momentum_com,
                                     update_kinematics=True)

    def get_com_jacobian(self):
        """
        Get the CoM Jacobian.

        Returns:
            np.array[N,N]: CoM Jacobian (where N is the number of DoFs)
        """
        pass

    def get_com_velocity(self):
        """
        Get the linear CoM velocity.

        Returns:
            np.array[3]: CoM velocity.
        """
        pass

    def get_com_acceleration(self):
        """
        Get the linear CoM acceleration.

        Returns:
            np.array[3]: CoM acceleration.
        """
        pass

    def get_gravity(self):
        """
        Get the gravity vector applied on the model.

        Returns:
            np.array[3]: gravity vector expressed in the world frame.
        """
        pass

    def set_gravity(self, gravity):
        """
        Set the gravity vector applied on the model.

        Args:
            gravity (np.array[3]): gravity vector expressed in the world frame.
        """
        pass

    def get_model_ordered_joint_names(self):
        """
        Get the model ordered joint names.

        Returns:
            list of str: list of joint names.
        """
        pass

    def get_jacobian(self, link_id, point):
        r"""
        Get the 6D Jacobian for a point on a link, that when multiplied with :math:`\dot{q}` gives a 6D vector that
        has the angular velocity as the first three entries and the linear velocity as the last three entries.

        .. math:: v = [\omega, \dot{p}] = J(q) \dot{q}

        where :math:`J(q)` is the concatenation of the angular and linear Jacobian.

        Args:
            link_id (int): unique link id.
            point (np.array[3]): position of the point in link's local frame

        Returns:
            np.array[6, N]: 6D Jacobian (=concatenation of the angular and linear Jacobian).
        """
        pass

    def get_pose(self, link_id):
        """
        Return the pose of the specified link.

        Args:
            link_id (int): link id

        Returns:
            np.array[7]: pose (position and quaternion expressed as [x,y,z,w])
        """
        pass

    def get_velocity_twist(self, link_id):
        r"""
        Compute the angular and linear velocity of a link, given by :math:`v = [\omega, \dot{p}]`.

        Args:
            link_id (int): link id.

        Returns:
            np.array[6]: The resulting 6D spatial velocity vector where the first three elements are the angular
                velocity and the last three are the linear velocity expressed in the global world reference frame.
        """
        pass

    def get_acceleration_twist(self, link_id):
        r"""
        Compute the angular and linear acceleration of a link, given by :math:`\dot{v} = [dot{\omega}, \ddot{p}]`.

        Args:
            link_id (int): link id.

        Returns:
            np.array[6]: The resulting 6D spatial acceleration vector where the first three elements are the
                angular acceleration and the last three are the linear acceleration expressed in the global world
                reference frame.
        """
        pass

    def get_relative_acceleration_twist(self, target_link_id, base_link_id):
        r"""
        Compute the relative angular and linear acceleration of a target link with respect to a base link. The
        acceleration is given by :math:`\dot{v} = [dot{\omega}, \ddot{p}]`.

        Returns:
            np.array[6]: The resulting 6D spatial acceleration vector where the first three elements are the
                angular acceleration and the last three are the linear acceleration expressed in the local frame of
                the base link.
        """
        pass

    def set_floating_base_pose(self, pose):
        """
        Set the floating base pose. Given the desired pose (=position + orientation), the corresponding joint position
        values for the 6 virtual joints (attached to the floating base) are computed.

        Args:
            pose (np.array[7]): the desired pose (position and orientation given as a quaternion [x,y,z,w]) of the
                floating base.
        """
        pass

    def set_floating_base_velocity(self, velocity):
        """
        Set the floating base velocity. This computes the corresponding joint velocity values for the 6 virtual joints
        (that are attached to the floating base).

        Args:
            np.array[3], np.array[6]: desired linear (and angular) velocity of the floating base.
        """
        pass

    def compute_gravity_compensation(self):
        """
        Return the torques to perform gravity compensation.

        Returns:
            np.array[N]: torques to perform gravity compensation.
        """
        pass

    def compute_nonlinear_term(self):
        r"""
        Computes the non-linear terms :math:`C(q, \dot{q})` in the dynamic equation of motion for a rigid-body system,
        given by:

        .. math:: \tau = H(q) \ddot{q} + C(q, \dot{q})

        where ":math:`\tau` is the vector of applied forces, :math:`H` is the joint space inertia matrix,
        :math:`C(q, \dot{q})` is the vector of force terms that account for the Coriolis and centrifugal forces,
        gravity, and any other forces acting on the system other than those in :math:`\tau`." [1]

        Returns:
            np.array[N]: non-linear force terms.

        References:
            - [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008
        """
        pass

    def compute_JdotQdot(self):
        r"""
        Compute :math:`\dot{J}(q) \dot{q}`, which appears in :math:`\dot{v} = J(q) \ddot{q} + \dot{J}(q) \dot{q}`,
        which is the first time derivative of :math:`v = J(q) \dot{q}`.

        Returns:
            np.array[6]: the matrix multiplication of the first derivative of the Jacobian with the joint velocities.
        """
        pass

    def compute_relative_JdotQdot(self, target_link_id, base_link_id):
        r"""
        Compute the relative :math:`\dot{J}(q) \dot{q}`, which appears in
        :math:`\dot{v} = J(q) \ddot{q} + \dot{J}(q) \dot{q}`, which is the first time derivative of
        :math:`v = J(q) \dot{q}`. The Jacobian is taken from the specified base link to the target link.

        Args:
            target_link_id (int): target link id
            base_link_id (int): base link id

        Returns:
            np.array[6]: relative :math:`\dot{J}(q) \dot{q}`
        """
        pass

    def get_inertia_matrix(self):
        """
        Computes the joint space inertia matrix.

        Returns:
            np.array[N, N]: joint space inertia matrix (where `N` is the number of DoFs).
        """
        pass

    def get_inertia_inverse_times_vector(self, vector):
        r"""
        Computes the effect of multiplying the inverse of the joint space inertia matrix :math:`H(q)` with a vector
        in linear time.

        Args:
            np.array[N]: vector to be multiplied with the inverse joint space inertia matrix.

        Returns:
            np.array[N]: resulting vector
        """
        pass

    def get_point_acceleration(self, link_id, point):
        """
        Computes the linear acceleration of a point on a link.

        Args:
            link_id (int): unique link id.
            point (np.array[3]): position of the point in link's local frame

        Returns:
            np.array[3]: The cartesian acceleration of the point in global frame
        """
        pass

    def get_link_id(self, name):
        """
        Return the link id associated with the given name.

        Args:
            name (str): name of the link

        Returns:
            int: unique link id
        """
        pass

    def get_mass(self):
        """
        Return the total mass of the model.

        Returns:
            float: total mass
        """
        pass

    def get_floating_base_link(self):
        """
        Return the floating base link.

        Returns:
            int: floating base link
        """
        pass

    def update(self, q=None, dq=None, ddq=None):
        if q is None:
            q = self.q
        if dq is None:
            dq = self.dq
        if ddq is None:
            ddq = self.ddq
        rbdl.UpdateKinematics(self.model, q, dq, ddq)
