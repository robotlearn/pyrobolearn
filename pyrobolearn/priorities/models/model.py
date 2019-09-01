#!/usr/bin/env python
r"""Model interface used in priority tasks.

A model interface is an abstraction layer that provides a common interface and remove the direct coupling between the
``Robot`` and the tasks, constraints, and solvers. It also serves as a container to different quantities (joint
positions, velocities, torques, etc), which avoids the need to recompute them for each task / constraint that used
them.

This is based on the implementation in `https://github.com/ADVRHumanoids/ModelInterfaceRBDL` (distributed under the
LGPLv3).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
    - [3] "Rigid Body Dynamics Algorithms", Featherstone, 2008
"""


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Arturo Laurenzi (C++)", "Luca Muratore (C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ModelInterface(object):
    r"""Model interface.

    A model interface is an abstraction layer that provides a common interface and remove the direct coupling between
    the ``Robot`` and the tasks, constraints, and solvers. It also serves as a container to different quantities (joint
    positions, velocities, torques, etc), which avoids the need to recompute them for each task / constraint that used
    them.

    This is based on the implementation in `https://github.com/ADVRHumanoids/ModelInterfaceRBDL` (distributed under
    the LGPLv3).
    """

    def __init__(self, model=None):
        """
        Initialize the model interface.

        Args:
            model (object): the model.
        """
        # set the model
        self.model = model

        # save the states: every time a method (which returned the result) is called, the result is cached here
        # avoiding to recompute it
        self._states = dict()

    ##############
    # Properties #
    ##############

    @property
    def gravity(self):
        """Get the gravity vector."""
        return self.get_gravity()

    # @gravity.setter
    # def gravity(self, gravity):
    #     """Set the gravity vector."""
    #     self.set_gravity(gravity)

    @property
    def num_dofs(self):
        """Return the number of degrees of freedom."""
        raise NotImplementedError

    @property
    def num_actuated_joints(self):
        """Return the number of actuated joints."""
        raise NotImplementedError

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
        pass

    def get_mass(self):
        """
        Return the total mass of the model.

        Returns:
            float: total mass
        """
        pass

    def has_floating_base(self):
        """
        Return True if we have a floating base.

        Returns:
            bool: True if floating base.
        """
        pass

    def get_floating_base_link(self):
        """
        Return the floating base link.

        Returns:
            int: floating base link
        """
        pass

    def get_joint_limits(self):
        r"""
        Return the joint limits.

        Returns:
            np.array[float[N]]: lower joint position limits.
            np.array[float[N]]: upper joint position limits.
        """
        pass

    def get_joint_velocity_limits(self):
        r"""
        Return the joint velocity limits.

        Returns:
            np.array[float[N]]: lower joint velocity limits.
            np.array[float[N]]: upper joint velocity limits.
        """
        pass

    def get_joint_positions(self):
        """
        Get the joint positions.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        pass

    def get_joint_velocities(self):
        """
        Get the joint velocities.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        pass

    def get_joint_accelerations(self):
        """
        Get the joint accelerations.

        Returns:
            np.array[float[N]]: the joint positions.
        """
        pass

    def get_com_position(self):
        """
        Get the position of the center of mass (CoM).

        Returns:
            np.array[float[3]]: position of the center of mass
        """
        pass

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
        pass

    def get_com_velocity(self):
        """
        Get the linear CoM velocity.

        Returns:
            np.array[float[3]]: CoM velocity.
        """
        pass

    def get_com_acceleration(self):
        """
        Get the linear CoM acceleration.

        Returns:
            np.array[float[3]]: CoM acceleration.
        """
        pass

    def get_gravity(self):
        """
        Get the gravity vector applied on the model.

        Returns:
            np.array[float[3]]: gravity vector expressed in the world frame.
        """
        pass

    def set_gravity(self, gravity):
        """
        Set the gravity vector applied on the model.

        Args:
            gravity (np.array[float[3]]): gravity vector expressed in the world frame.
        """
        pass

    def get_model_ordered_joint_names(self):
        """
        Get the model ordered joint names.

        Returns:
            list[str]: list of joint names.
        """
        pass

    def get_jacobian(self, link, wrt_link=None, frame=None, point=(0., 0., 0.)):
        r"""
        Get the 6D Jacobian for a point on a link, that when multiplied with :math:`\dot{q}` gives a 6D vector that
        has the angular velocity as the first three entries and the linear velocity as the last three entries.

        .. math:: v = [\omega, \dot{p}] = J(q) \dot{q}

        where :math:`J(q)` is the concatenation of the angular and linear Jacobian.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): unique link id, or name. If specified, it will take the relative jacobian. If
              None, the jacobian will be taken with respect to the world frame.
            frame (int, str, None): unique link id, or name. If specified, it will express the final jacobian in that
              specified frame.
            point (np.array[float[3]]): position of the point in link's local frame.

        Returns:
            np.array[float[6,N]]: 6D Jacobian (=concatenation of the angular and linear Jacobian).
        """
        pass

    def get_pose(self, link, wrt_link=None, point=(0., 0., 0.)):
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
        pass

    def get_position(self, link, wrt_link=None):
        """
        Return the position of the specified link with respect to another link.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): the other link id, or name. If None, returns the position wrt to the world,
              and if -1 wrt to the base.

        Returns:
            np.array[float[3]]: position
        """
        pass

    def get_orientation(self, link, wrt_link=None):
        """
        Return the orientation of the specified link with respect to another link.

        Args:
            link (int, str): unique link id, or name.
            wrt_link (int, str, None): the other link id, or name. If None, returns the orientation wrt to the world,
              and if -1 wrt to the base.

        Returns:
            np.array[float[4]]: orientation (expressed as a quaternion [x,y,z,w])
        """
        pass

    def get_velocity_twist(self, link):
        r"""
        Compute the angular and linear velocity of a link, given by :math:`v = [\omega, \dot{p}]`.

        Args:
            link (int, str): unique link id, or name..

        Returns:
            np.array[float[6]]: The resulting 6D spatial velocity vector where the first three elements are the angular
                velocity and the last three are the linear velocity expressed in the global world reference frame.
        """
        pass

    def get_acceleration_twist(self, link):
        r"""
        Compute the angular and linear acceleration of a link, given by :math:`\dot{v} = [dot{\omega}, \ddot{p}]`.

        Args:
            link (int, str): unique link id, or name..

        Returns:
            np.array[float[6]]: The resulting 6D spatial acceleration vector where the first three elements are the
                angular acceleration and the last three are the linear acceleration expressed in the global world
                reference frame.
        """
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def get_inertia_inverse_times_vector(self, vector):
        r"""
        Computes the effect of multiplying the inverse of the joint space inertia matrix :math:`H(q)` with a vector
        in linear time.

        Args:
            vector (np.array[float[N]]): vector to be multiplied with the inverse joint space inertia matrix.

        Returns:
            np.array[float[N]]: resulting vector
        """
        pass

    def get_point_acceleration(self, link, point=(0., 0., 0.)):
        """
        Computes the linear acceleration of a point on a link.

        Args:
            link (int, str): unique link id, or name.
            point (np.array[float[3]]): position of the point in link's local frame

        Returns:
            np.array[float[3]]: The cartesian acceleration of the point in global frame
        """
        pass

    def get_centroidal_momentum_matrix(self):
        r"""
        Return the centroidal momentum matrix.

        Returns:
            np.array[float[6,6+N]]: the centroidal momentum matrix :math:`A_G`
        """
        pass

    def update(self):
        """
        This is to notify the model interface that we moved to the next time step :math:`t \rightarrow t+1`.
        Practically, it frees every variables that has been cached in this instance (in self._states).
        """
        self._states = dict()
        self.get_joint_positions()
        self.get_joint_velocities()
        self.get_joint_accelerations()
