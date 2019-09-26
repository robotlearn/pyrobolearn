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
from pyrobolearn.utils.transformation import get_quaternion_from_matrix


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
        model = rbdl.loadModel(filename=urdf.encode(), floating_base=floating_base, verbose=verbose)

        # call parent constructor
        super(RBDLModelInterface, self).__init__(model)

        # define joint attributes
        self.zeros = np.zeros(self.model.q_size)
        self._q = np.zeros(self.model.q_size)
        self._dq = np.zeros(self.model.qdot_size)
        self._ddq = np.zeros(self.model.qdot_size)

        self.mass = 0
        for body_id in range(len(self.model.mBodies)):
            body = self.model.mBodies[body_id]
            self.mass += body.mMass

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
            raise TypeError("Expecting the given 'model' to be an instance of `rbdl.Model`, instead got: "
                            "{}".format(type(model)))
        self._model = model

    @property
    def num_dofs(self):
        """Return the number of degrees of freedom."""
        return self.model.dof_count

    ###########
    # Methods #
    ###########

    def get_link_id(self, link):
        """
        Return the link id associated with the given name.

        Args:
            link (str, int): unique link name (or id). If id, it will just return the argument.

        Returns:
            int: unique link id
        """
        if isinstance(link, (str, bytes)):
            if link == 'world' or link == 'ROOT':
                link = self.model.GetBodyId('ROOT'.encode())
            else:
                link = self.model.GetBodyId(link.encode())
        if link >= 4294967295:
            raise ValueError("The given link doesn't exist in the RBDL model.")
        return link

    def get_mass(self):
        """
        Return the total mass of the model.

        Returns:
            float: total mass
        """
        return self.mass

    def has_floating_base(self):
        """
        Return True if we have a floating base.

        Returns:
            bool: True if floating base.
        """
        if len(self.model.mBodies) - 1 != self.model.dof_count:  # mBodies contains 'ROOT' as well
            return True
        return False

    def get_floating_base_link(self):
        """
        Return the floating base link.

        Returns:
            int: floating base link
        """
        pass

    def get_joint_positions(self):
        """
        Get the joint positions.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        return self._q

    def get_joint_velocities(self):
        """
        Get the joint velocities.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        return self._dq

    def get_joint_accelerations(self):
        """
        Get the joint accelerations.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        return self._ddq

    def get_com_position(self):
        """
        Get the position of the center of mass (CoM).

        Returns:
            np.array[float[3]]: position of the center of mass
        """
        # return rbdl.CalcCenterOfMass(self.model, self._q, self._dq, self._ddq, self.com, self.com_vel, self.com_acc,
        #                              self.angular_momentum_com, self.change_angular_momentum_com,
        #                              update_kinematics=True)
        rbdl.CalcCenterOfMass(self.model, self._q, self._dq, self._ddq, self.com)  # TODO: update library
        return self.com

    def get_com_velocity(self):
        """
        Get the linear CoM velocity.

        Returns:
            np.array[float[3]]: CoM velocity.
        """
        rbdl.CalcCenterOfMass(self.model, self._q, self._dq, self._ddq, self.com, com_velocity=self.com_vel)
        return self.com_vel

    def get_com_acceleration(self):
        """
        Get the linear CoM acceleration.

        Returns:
            np.array[float[3]]: CoM acceleration.
        """
        rbdl.CalcCenterOfMass(self.model, self._q, self._dq, self._ddq, self.com, com_acceleration=self.com_acc)
        return self.com_acc

    def get_com_jacobian(self, full=False):
        """
        Get the CoM Jacobian.

        Args:
            full (bool): if True, it will return the jacobian as the concatenation of the angular and linear Jacobian.
              Otherwise, it will just return the linear Jacobian.

        Returns:
            if full:
                np.array[float[6,N]]: CoM Jacobian (concatenation of the angular and linear Jacobian, where N is the
                  number of DoFs)
            else:
                np.array[float[3,N]]: CoM Jacobian (only the linear part)
        """
        jacobian = np.zeros((6, self.num_dofs)) if full else np.zeros((3, self.num_dofs))
        mass = 0

        for body_id in range(len(self.model.mBodies)):
            body = self.model.mBodies[body_id]
            jac = np.zeros(3, self.num_dofs)
            rbdl.CalcPointJacobian(self.model, self._q, body_id, body.mCenterOfMass, jac, update_kinematics=False)
            if full:
                jacobian[3:] += body.mMass * jac
            else:
                jacobian += body.mMass * jac
            mass += body.mMass

        return jacobian / mass

    def get_gravity(self):
        """
        Get the gravity vector applied on the model.

        Returns:
            np.array[float[3]]: gravity vector expressed in the world frame.
        """
        return m.gravity

    def set_gravity(self, gravity):
        """
        Set the gravity vector applied on the model.

        Args:
            gravity (np.array[float[3]]): gravity vector expressed in the world frame.
        """
        m.gravity = gravity

    def get_model_ordered_joint_names(self):
        """
        Get the model ordered joint names.

        Returns:
            list[str]: list of joint names.
        """
        pass

    def get_jacobian(self, link, wrt_link=None, point=(0., 0., 0.)):  # TODO: wrt_link
        r"""
        Get the 6D Jacobian for a point on a link, that when multiplied with :math:`\dot{q}` gives a 6D vector that
        has the angular velocity as the first three entries and the linear velocity as the last three entries.

        .. math:: v = [\omega, \dot{p}] = J(q) \dot{q}

        where :math:`J(q)` is the concatenation of the angular and linear Jacobian.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): unique link id, or name. If specified, it will take the relative jacobian. If
              None, the jacobian will be taken with respect to the world frame.
            point (np.array[float[3]]): position of the point in link's local frame.

        Returns:
            np.array[float[6,N]]: 6D Jacobian (=concatenation of the angular and linear Jacobian).
        """
        link = self.get_link_id(link)
        jacobian = np.zeros((6, self.num_dofs))
        point = np.asarray(point)
        rbdl.CalcPointJacobian6D(self.model, self._q, link, point, jacobian, update_kinematics=False)
        return jacobian

    def get_pose(self, link, wrt_link=None, point=(0., 0., 0.)):
        """
        Return the pose of the specified link with respect to the other given link.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): the other link id, or name. If None, returns the position wrt to the world, and
              if -1 wrt to the base.
            point (np.array[float[3]]): position of the point in link's local frame.

        Returns:
            np.array[float[7]]: pose (position and quaternion expressed as [x,y,z,w])
        """
        link = self.get_link_id(link)
        point = np.asarray(point)
        position = rbdl.CalcBodyToBaseCoordinates(self.model, self._q, link, point, update_kinematics=False)
        orientation = rbdl.CalcBodyWorldOrientation(self.model, self._q, link, update_kinematics=False)
        orientation = get_quaternion_from_matrix(orientation)
        return np.concatenate((position, orientation))

    def get_velocity_twist(self, link, point=(0., 0., 0.)):
        r"""
        Compute the angular and linear velocity of a link, given by :math:`v = [\omega, \dot{p}]`.

        Args:
            link (int, str): unique link id, or name.
            point (np.array[float[3]]): position of the point in link's local frame.

        Returns:
            np.array[float[6]]: The resulting 6D spatial velocity vector where the first three elements are the angular
                velocity and the last three are the linear velocity expressed in the global world reference frame.
        """
        link = self.get_link_id(link)
        point = np.asarray(point)
        velocity = rbdl.CalcPointVelocity6D(self.model, self._q, self._dq, link, point, update_kinematics=False)
        return velocity

    def get_acceleration_twist(self, link, point=(0., 0., 0.)):
        r"""
        Compute the angular and linear acceleration of a link, given by :math:`\dot{v} = [dot{\omega}, \ddot{p}]`.

        Args:
            link (int, str): unique link id, or name.
            point (np.array[float[3]]): position of the point in link's local frame.

        Returns:
            np.array[float[6]]: The resulting 6D spatial acceleration vector where the first three elements are the
                angular acceleration and the last three are the linear acceleration expressed in the global world
                reference frame.
        """
        link = self.get_link_id(link)
        point = np.asarray(point)
        acceleration = rbdl.CalcPointVelocity6D(self.model, self._q, self._dq, self._ddq, link, point,
                                                update_kinematics=False)
        return acceleration

    def get_relative_acceleration_twist(self, target_link, base_link):
        r"""
        Compute the relative angular and linear acceleration of a target link with respect to a base link. The
        acceleration is given by :math:`\dot{v} = [dot{\omega}, \ddot{p}]`.

        Args:
            target_link (int): target link id, or name.
            base_link (int): base link id, or name.

        Returns:
            np.array[float[6]]: The resulting 6D spatial acceleration vector where the first three elements are the
                angular acceleration and the last three are the linear acceleration expressed in the local frame of
                the base link.
        """
        pass

    def set_floating_base_pose(self, pose):
        """
        Set the floating base pose. Given the desired pose (=position + orientation), the corresponding joint position
        values for the 6 virtual joints (attached to the floating base) are computed.

        Args:
            pose (np.array[float[7]]): the desired pose (position and orientation given as a quaternion [x,y,z,w]) of
              the floating base.
        """
        pass

    def set_floating_base_velocity(self, velocity):
        """
        Set the floating base velocity. This computes the corresponding joint velocity values for the 6 virtual joints
        (that are attached to the floating base).

        Args:
            np.array[float[3]], np.array[float[6]]: desired linear (and angular) velocity of the floating base.
        """
        pass

    def compute_gravity_compensation(self):
        """
        Return the torques to perform gravity compensation.

        Returns:
            np.array[float[N]]: torques to perform gravity compensation.
        """
        tau = np.zeros(self.num_dofs)
        rbdl.NonlinearEffects(self.model, self._q, self.zeros, tau)
        return tau

    def compute_nonlinear_term(self):
        r"""
        Computes the non-linear terms :math:`N(q, \dot{q})` in the dynamic equation of motion for a rigid-body system,
        given by:

        .. math:: \tau = H(q) \ddot{q} + N(q, \dot{q})

        where ":math:`\tau` is the vector of applied forces, :math:`H` is the joint space inertia matrix,
        :math:`N(q, \dot{q})` is the vector of force terms that account for the Coriolis and centrifugal forces,
        gravity, and any other forces acting on the system other than those in :math:`\tau`." [1]

        Returns:
            np.array[float[N]]: non-linear force terms.

        References:
            - [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008
        """
        tau = np.zeros(self.num_dofs)
        rbdl.NonlinearEffects(self.model, self._q, self._dq, tau)
        return tau

    def compute_JdotQdot(self, link, point=(0., 0., 0.)):
        r"""
        Compute :math:`\dot{J}(q) \dot{q}`, which appears in :math:`\dot{v} = J(q) \ddot{q} + \dot{J}(q) \dot{q}`,
        which is the first time derivative of :math:`v = J(q) \dot{q}`.

        Args:
            link (int, str): unique link id, or name.
            point (np.array[float[3]]): position of the point in link's local frame.

        Returns:
            np.array[float[6]]: the matrix multiplication of the first derivative of the Jacobian with the joint
              velocities.
        """
        link = self.get_link_id(link)
        point = np.asarray(point)
        return rbdl.CalcPointAcceleration6D(self.model, self._q, self._dq, self._ddq * 0, link, point,
                                            update_kinematics=True)

    def compute_relative_JdotQdot(self, target_link, base_link):
        r"""
        Compute the relative :math:`\dot{J}(q) \dot{q}`, which appears in
        :math:`\dot{v} = J(q) \ddot{q} + \dot{J}(q) \dot{q}`, which is the first time derivative of
        :math:`v = J(q) \dot{q}`. The Jacobian is taken from the specified base link to the target link.

        Args:
            target_link (int): target link id, or name.
            base_link (int): base link id, or name.

        Returns:
            np.array[float[6]]: relative :math:`\dot{J}(q) \dot{q}`
        """
        pass

    def get_inertia_matrix(self):
        """
        Computes the joint space inertia matrix.

        Returns:
            np.array[float[N,N]]: joint space inertia matrix (where `N` is the number of DoFs).
        """
        H = np.zeros(self.num_dofs, self.num_dofs)
        rbdl.CompositeRigidBodyAlgorithm(self.model, self._q, H, update_kinematics=False)
        return H

    def get_inertia_inverse_times_vector(self, vector):
        r"""
        Computes the effect of multiplying the inverse of the joint space inertia matrix :math:`H(q)` with a vector
        in linear time.

        Args:
            vector (np.array[float[N]]): vector to be multiplied with the inverse joint space inertia matrix.

        Returns:
            np.array[float[N]]: resulting vector
        """
        # TODO: wrap CalcMInvTimesTau
        H = self.get_inertia_matrix()
        return H.dot(vector)

    def get_point_acceleration(self, link, point=(0., 0., 0.)):
        """
        Computes the linear acceleration of a point on a link.

        Args:
            link (int, str): unique link id, or name.
            point (np.array[float[3]]): position of the point in link's local frame

        Returns:
            np.array[float[3]]: The cartesian acceleration of the point in global frame
        """
        link = self.get_link_id(link)
        point = np.asarray(point)
        return rbdl.CalcPointAcceleration(self.model, self._q, self._dq, self._ddq, link, point, update_kinematics=True)

    def update(self, q=None, dq=None, ddq=None):
        """Update: move to the next step."""
        if q is not None:
            self._q = q
        if dq is not None:
            self._dq = dq
        if ddq is not None:
            self._ddq = ddq
        rbdl.UpdateKinematics(self.model, self._q, self._dq, self._ddq)
