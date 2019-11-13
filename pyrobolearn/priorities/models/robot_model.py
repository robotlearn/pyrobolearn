#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Robot model interface used in priority tasks.

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
    - [3] "Rigid Body Dynamics Algorithms", Featherstone, 2008
"""

import numpy as np

from pyrobolearn.priorities.models import ModelInterface
from pyrobolearn.robots import Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotModelInterface(ModelInterface):
    r"""Robot Model interface.

    Robot model interface that accepts as input a robot that inherited from `pyrobolearn.robots.Robot`.
    """

    def __init__(self, model):
        """
        Initialize the robot model interface.

        Args:
            model (Robot): a robot instance.
        """
        super(RobotModelInterface, self).__init__(model)

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
        if not isinstance(model, Robot):
            raise TypeError("Expecting the given 'model' to be an instance of `Robot`, instead got: "
                            "{}".format(type(model)))
        self._model = model

    # alias
    @property
    def robot(self):
        """Return the robot instance."""
        return self._model

    @property
    def num_dofs(self):
        """Return the number of degrees of freedom.

        - If fixed base, this is equal to the number of actuated joints.
        - If floating base, this is equal to the number of actuated joints + 6 DoFs for the base.
        """
        return self.model.num_dofs
    
    @property
    def num_actuated_joints(self):
        """Return the number of actuated joints."""
        return self.model.num_actuated_joints

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
        if isinstance(link, str):
            return self.model.get_link_ids(link)
        return link

    def get_mass(self):
        """
        Return the total mass of the model.

        Returns:
            float: total mass
        """
        return self.model.mass

    def has_floating_base(self):
        """
        Return True if we have a floating base.

        Returns:
            bool: True if floating base.
        """
        return self.model.has_floating_base()

    def get_floating_base_link(self):
        """
        Return the floating base link.

        Returns:
            int: floating base link
        """
        return -1

    def get_joint_limits(self):
        r"""
        Return the joint limits.

        Returns:
            np.array[float[2, N]]: lower and upper joint position limits.
        """
        return self.model.get_joint_limits()
    
    def get_joint_velocity_limits(self):
        r"""
        Return the joint velocity limits.

        Returns:
            np.array[float[2, N]]: lower and upper joint velocity limits.
        """
        dq = self.model.get_joint_max_velocities()
        return -dq, dq

    def get_joint_positions(self):
        """
        Get the joint positions.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        return self.model.get_joint_positions()

    def get_joint_velocities(self):
        """
        Get the joint velocities.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        return self.model.get_joint_velocities()

    def get_joint_accelerations(self):
        """
        Get the joint accelerations.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        return self.model.get_joint_accelerations()

    def get_com_position(self):
        """
        Get the position of the center of mass (CoM).

        Returns:
            np.array[float[3]]: position of the center of mass
        """
        return self.model.get_center_of_mass_position()

    def get_com_velocity(self):
        """
        Get the linear CoM velocity.

        Returns:
            np.array[float[3]]: CoM velocity.
        """
        return self.model.get_center_of_mass_velocity()

    def get_com_acceleration(self):
        """
        Get the linear CoM acceleration.

        Returns:
            np.array[float[3]]: CoM acceleration.
        """
        return self.model.get_center_of_mass_acceleration()

    def get_com_jacobian(self, full=False):
        """
        Get the CoM Jacobian.

        Args:
            full (bool): if True, it will return the jacobian as the concatenation of the angular and linear Jacobian.
              Otherwise, it will just return the linear Jacobian.

        Returns:
            if full:
                np.array[float[6,N]]: CoM Jacobian (concatenation of the linear and angular Jacobian, where N is the
                  number of DoFs)
            else:
                np.array[float[3,N]]: CoM Jacobian (only the linear part)
        """
        jac = self.model.get_center_of_mass_jacobian()
        if full:
            return jac
        return jac[:3]

    def get_gravity(self):
        """
        Get the gravity vector applied on the model.

        Returns:
            np.array[float[3]]: gravity vector expressed in the world frame.
        """
        return self.model.simulator.gravity

    def set_gravity(self, gravity):
        """
        Set the gravity vector applied on the model.

        Args:
            gravity (np.array[float[3]]): gravity vector expressed in the world frame.
        """
        self.model.simulator.gravity = gravity

    def get_model_ordered_joint_names(self):
        """
        Get the model ordered (non-fixed) joint names.

        Returns:
            list[str]: list of joint names.
        """
        return self.model.get_link_names(self.model.joints)

    def get_jacobian(self, link, wrt_link=None, frame=None, point=(0., 0., 0.)):
        r"""
        Get the 6D Jacobian for a point on a link, that when multiplied with :math:`\dot{q}` gives a 6D vector that
        has the angular velocity as the first three entries and the linear velocity as the last three entries.

        .. math:: v = [\omega, \dot{p}] = J(q) \dot{q}

        where :math:`J(q)` is the concatenation of the angular and linear Jacobian.

        Warnings: this is different from the `robot.get_jacobian()` which returns the concatenation of the linear
        and angular Jacobian. Here, we return the concatenation of the angular followed by the linear Jacobian.

        Args:
            link (int): unique link id.
            wrt_link (int, str, None): unique link id, or name. If specified, it will take the relative jacobian. If
              None, the jacobian will be taken with respect to the world frame.
            frame (int, str, None): unique link id, or name. If specified, it will express the final jacobian in that
              specified frame.
            point (np.array[float[3]], None): the point on the specified link to compute the Jacobian (in link local
              coordinates around its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.array[float[6,N]]: 6D Jacobian (=concatenation of the linear and angular Jacobian).
        """
        link = self.get_link_id(link)
        wrt_link = None if wrt_link is None else self.get_link_id(wrt_link)
        frame = None if frame is None else self.get_link_id(frame)

        # if the jacobian is already cached, return it
        if ('J', link, wrt_link, frame, tuple(point)) in self._states:
            return self._states[('J', link, wrt_link, frame, tuple(point))]

        # get the jacobian, cache it, and return it
        if wrt_link is None:
            jacobian = self.model.get_jacobian(link_id=link, local_position=point)
        else:
            jacobian = self.model.get_relative_jacobian(link_id=link, wrt_link_id=wrt_link, local_position=point)
        if frame is not None:
            jacobian = self.model.express_jacobian_in_frame(jacobian, link_id=frame)

        self._states[('J', link, wrt_link, frame, tuple(point))] = jacobian
        return jacobian

    def get_pose(self, link, wrt_link=None, point=(0., 0., 0.)):  # TODO: use point
        """
        Return the pose of the specified link with respect to another link.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): the other link id, or name. If None, returns the pose wrt to the world, and
              if -1 wrt to the base.
            point (np.array[float[3]]): position of the point in link's local frame.

        Returns:
            np.array[float[7]]: pose (position and quaternion expressed as [x,y,z,w])
        """
        link = self.get_link_id(link)
        if wrt_link is None:
            return self.model.get_link_world_poses(link)
        return self.model.get_link_poses(link, self.get_link_id(wrt_link))

    def get_position(self, link, wrt_link=None):  # TODO: use point
        """
        Return the position of the specified link with respect to another link.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): the other link id, or name. If None, returns the position wrt to the world,
              and if -1 wrt to the base.

        Returns:
            np.array[float[3]]: position
        """
        link = self.get_link_id(link)
        if wrt_link is None:
            return self.model.get_link_world_positions(link)
        return self.model.get_link_positions(link, wrt_link_id=self.get_link_id(wrt_link))

    def get_orientation(self, link, wrt_link=None):  # TODO: use point
        """
        Return the orientation of the specified link with respect to another link.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): the other link id, or name. If None, returns the orientation wrt to the world,
              and if -1 wrt to the base.

        Returns:
            np.array[float[4]]: orientation (expressed as a quaternion [x,y,z,w])
        """
        link = self.get_link_id(link)
        if wrt_link is None:
            return self.model.get_link_world_orientations(link)
        return self.model.get_link_orientations(link, wrt_link_id=self.get_link_id(wrt_link))

    def get_velocity(self, link, wrt_link=None, point=(0., 0., 0.)):  # TODO: use point
        r"""
        Compute the linear and angular velocity of a link, given by :math:`v = [\dot{p}, \omega]`.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): the other link id, or name. If None, returns the velocities wrt to the world,
              and if -1 wrt to the base.
            point (np.array[float[3]]): position of the point in link's local frame.

        Returns:
            np.array[float[6]]: The resulting 6D velocity vector where the first three elements are the linear
                velocity and the last three are the angular velocity expressed in the global world reference frame.
        """
        link = self.get_link_id(link)
        if wrt_link is None:
            return self.model.get_link_world_velocities(link)
        return self.model.get_link_velocities(link, wrt_link_id=self.get_link_id(wrt_link))

    def get_velocity_twist(self, link, point=(0., 0., 0.)):  # TODO: use point
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
        return self.model.get_link_velocity_twists(link)

    def get_acceleration_twist(self, link, point=(0., 0., 0.)):  # TODO: use point
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
        return self.model.get_link_acceleration_twists(link)

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
        target_link = self.get_link_id(target_link)
        base_link = self.get_link_id(base_link)
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
            velocity (np.array[float[3]], np.array[float[6]]): desired linear (and angular) velocity of the floating
              base.
        """
        pass

    def compute_gravity_compensation(self):
        """
        Return the torques to perform gravity compensation.

        Returns:
            np.array[float[N]]: torques to perform gravity compensation.
        """
        return self.model.get_gravity_compensation_torques()

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
        # if already cached, return it
        if 'N' in self._states:
            return self._states['N']

        # otherwise, compute, cache, and return it
        tau = self.model.get_coriolis_and_gravity_compensation_torques()
        self._states['N'] = tau
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
        # if already cached, return it
        if ('Jdotqdot', link, tuple(point)) in self._states:
            return self._states[('Jdotqdot', link, point)]

        # compute, cache, and return it
        # \dot{v} = J(q)\ddot{q} + \dot{J}(q) \dot{q}
        # if ddq = 0; then \dot{v} = \dot{J}(q) \dot{q}
        # Otherwise: \dot{J}(q) \dot{q} = \dot{v} - J(q) \ddot{q}
        acc = self.get_acceleration_twist(link, point)
        jacobian = self.get_jacobian(link, point=point)
        ddq = self.get_joint_accelerations()

        # return \dot{J}(q) \dot{q} = \dot{v} - J(q) \ddot{q}
        return acc - jacobian.dot(ddq)

    def compute_com_JdotQdot(self):
        r"""
        Compute :math:`\dot{J}_{CoM}(q) \dot{q}` from the centroidal momentum matrix.

        Returns:
            np.array[float[6]]: the matrix multiplication of the first derivative of the CoM Jacobian with the joint
              velocities.
        """
        A_G, dA_G_dq = self.get_centroidal_dynamics()
        return dA_G_dq / self.get_mass()

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
        target_link = self.get_link_id(target_link)
        base_link = self.get_link_id(base_link)
        pass

    def get_inertia_matrix(self):
        """
        Computes the joint space inertia matrix.

        Returns:
            np.array[float[N,N]]: joint space inertia matrix (where `N` is the number of DoFs).
        """
        # if already cached, return it
        if 'H' in self._states:
            return self._states['H']

        # compute, cache, and return it
        inertia = self.model.get_inertia_matrix()
        self._states['H'] = inertia
        return inertia

    def get_inertia_inverse_times_vector(self, vector):
        r"""
        Computes the effect of multiplying the inverse of the joint space inertia matrix :math:`H(q)` with a vector
        in linear time.

        Args:
            vector (np.array[float[N]]): vector to be multiplied with the inverse joint space inertia matrix.

        Returns:
            np.array[float[N]]: resulting vector
        """
        # TODO: improve this
        # get inverse of inertia matrix
        Hinv = self._states.get('Hinv', None)

        # if None, compute and cache it
        if Hinv is None:
            H = self.get_inertia_matrix()
            Hinv = np.linalg.inv(H)
            self._states['Hinv'] = Hinv

        return np.dot(Hinv, vector)

    def get_point_acceleration(self, link, point=(0., 0., 0.)):  # TODO: use point
        """
        Computes the linear acceleration of a point on a link.

        Args:
            link (int): unique link id.
            point (np.array[float[3]]): position of the point in link's local frame

        Returns:
            np.array[float[3]]: The cartesian acceleration of the point in global frame
        """
        link = self.get_link_id(link)
        return self.model.get_link_world_accelerations(link)

    def get_centroidal_momentum_matrix(self):
        r"""
        Return the centroidal momentum matrix.

        Returns:
            np.array[float[6,6+N]]: the centroidal momentum matrix :math:`A_G`
        """
        return self.model.get_centroidal_momentum_matrix()

    def get_centroidal_momentum(self):
        r"""
        Return the centroidal momentum vector :math:`h_G = A_G \dot{q} \in \mathbb{R}^6`

        Returns:
            np.array[float[6]]: centroidal momentum vector.
        """
        return self.model.get_centroidal_momentum()

    def get_centroidal_dynamics(self):
        r"""
        Return the centroidal momentum matrix :math:`A_G` and its derivative multiplied by the joint velocities
        :math:`\dot{A}_G \dot{q}`.

        Returns:
            np.array[float[6,6+N]]: the centroidal momentum matrix :math:`A_G`
            np.array[float[6]]: the centroidal dynamics velocity-dependent bias vector :math:`\dot{A}_G \dot{q}`
        """
        return self.model.get_centroidal_dynamics()

    def update(self, q=None, dq=None, ddq=None, update_model=False):
        """Update: move to the next step."""
        self._states = dict()
        if update_model:
            self.model.step()
