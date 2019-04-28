#!/usr/bin/env python
"""Define the robot class from which all the other robots inherit from.

The robot can access to the simulator as the `World`. If the `Robot` class defined here is the first layer in
the inheritance hierarchy/tree then in the second layer, you have `Manipulator`, `LeggedRobot`, `WheeledRobot`,
`UAV`, etc.

Dependencies:
- `pyrobolearn.simulators`
- `pyrobolearn.utils`
"""

# import rbdl
import numpy as np
# import quaternion
import collections
import os

from pyrobolearn.utils.orientation import *
from pyrobolearn.robots.base import ControllableBody


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Robot(ControllableBody):
    r"""Robot class.

    This is the class that all robots should inherit from. It contains all the useful methods to operate the robot,
    and has been implemented such that it is very generic.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1., *args, **kwargs):
        """
        Initialize the robot.

        Args:
            simulator: reference to the simulator such that the robot can access it.
            urdf (str): path to the URDF/MJCF file.
            position (np.float[3]): initial position.
            orientation (np.float[4]): initial orientation represented as a quaternion (x,y,z,w).
            fixed_base (bool, None): if True, the base of the robot will be fixed.
            scaling (float): scaling factor.
        """
        # check parameters
        if position is None:
            position = (0., 0., 1.5)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        # call parent constructor
        super(Robot, self).__init__(simulator)

        # load the robot
        # self.sim.configure_debug_visualizer(self.sim.COV_ENABLE_RENDERING, 0)

        if urdf[-3:] == 'xml' or urdf[-4:] == 'mjcf':
            self.id = self.sim.load_mjcf(urdf)[0]  # assume the first entity is the robot
        else:  # if urdf[-4:] == 'urdf':
            self.id = self.sim.load_urdf(urdf, position, orientation, use_fixed_base=fixed_base, scale=scaling)

        # self.sim.configure_debug_visualizer(self.sim.COV_ENABLE_RENDERING, 1)

        # rescale if specified
        if scaling != 1.0:
            # we rescale manually the mass and inertia matrices of each link
            for link in range(self.num_links):
                info = self.sim.get_dynamics_info(self.id, link)
                mass, local_inertia_diagonal = info[0], np.array(info[2])
                mass *= scaling**3      # because the density is unchanged when scaling
                local_inertia_diagonal *= scaling**5   # 5 = 3+2; 3 is for the mass, and 2 is for the distance: I~mr^2
                self.sim.change_dynamics(self.id, link, mass=mass, local_inertia_diagonal=local_inertia_diagonal)

        # set robot properties
        self.init_position = position
        self.init_orientation = orientation
        self.base_height = self.get_base_position()[2]
        self.base_up_vector = self.up_vector
        self.base_forward_vector = self.forward_vector
        # print("BASE HEIGHT: {}".format(self.base_height))
        # print("UP VECTOR: {}".format(self.base_up_vector))
        # print("FORWARD VECTOR: {}".format(self.base_forward_vector))
        self.com = None  # center of mass

        # set useful variables
        self.joints = []  # non-fixed joint/link indices in the simulator
        self.joint_names = {}  # joint name to id in the simulator
        self.link_names = {}  # link name to id in the simulator
        self.end_effectors = []  # end effector indices
        self.end_effector_names = {}  # end effector name to id in the simulator
        self.actuators = []  # list of actuators
        self.sensors = []  # list of sensors

        # get actuated joints
        for joint_id in range(self.num_joints):
            # Get joint info
            joint_info = self.sim.get_joint_info(self.id, joint_id)
            self.joint_names[joint_info[1]] = joint_info[0]
            self.link_names[joint_info[12]] = joint_info[0]

            if joint_info[2] != self.sim.JOINT_FIXED:  # if not a fixed joint
                self.joints.append(joint_info[0])

        # set automatically the end-effectors
        self._set_end_effectors()

        # visual debug: sliders and drawing
        self.joint_sliders = {}
        self.com_visual = None
        self.projected_com_visual = None

        # other variables
        self.coriolis_and_gravity_compensation = False
        self.floating_base = self._check_floating_base()

        # remember visual shapes
        # warning: the length of the returned list might be different from the number of links, because some links
        # don't have any visual shapes
        visual_shapes = self.sim.get_visual_shape_data(self.id)
        self.visual_shapes = {shape[1]: {'dimensions': shape[3], 'color': list(shape[7])} for shape in visual_shapes}

        # symbolic equations
        self.symbols = None

        # init joint positions
        self.init_joint_positions = self.get_joint_positions()
        self.joint_limits = self.get_joint_limits()

        # Gains
        self.kp, self.kd = None, None

    def __repr__(self):
        """
        Return the name of the class.

        Returns:
            str: name of the class
        """
        return self.__class__.__name__

    ##############
    # Properties #
    ##############

    @property
    def num_dofs(self):
        """Return the number of degrees of freedom (i.e. the number of joints that are not fixed)"""
        return len(self.joints)

    ########
    # Base #
    ########

    def get_base_pose(self):
        """
        Get base position and orientation with respect to the world frame.

        Returns:
            float[3]: position
            np.float[4]: orientation (x, y, z, w)
        """
        return self.sim.get_base_pose(self.id)

    def get_base_position(self):
        """
        Return the base position.

        Returns:
            float[3]: base position.
        """
        return self.sim.get_base_position(self.id)

    def get_base_orientation(self):
        """
        Get the base orientation in the form of a quaternion (x, y, z, w).

        Returns:
            quaternion (np.float[4]): base orientation in the form of a quaternion (x, y, z, w).
        """
        return self.sim.get_base_orientation(self.id)

    def get_base_velocity(self, concatenate=True):
        """
        Return the base linear and angular velocities.

        Returns:
            np.float[6]: linear and angular velocities of the base
        """
        lin_vel, ang_vel = self.sim.get_base_velocity(self.id)
        if concatenate:
            return np.concatenate((lin_vel, ang_vel))
        return lin_vel, ang_vel

    def get_base_linear_velocity(self):
        """
        Return the linear velocity of the base.

        Returns:
            float[3]: linear velocity of the base
        """
        return self.sim.get_base_linear_velocity(self.id)

    def get_base_angular_velocity(self):
        """
        Return the angular velocity of the base.

        Returns:
            float[3]: angular velocity of the base
        """
        return self.sim.get_base_angular_velocity(self.id)

    def _check_floating_base(self):
        """
        Return True if the robot has a floating base (i.e. floating root link). Otherwise, it is a fixed base.

        Returns:
            bool: True if the robot has a floating base.
        """
        # We used the fact if the robot has a floating base then the base velocity can be close to 0, but never
        # completely equal to 0, unless the base is fixed
        # return np.all(np.zeros(6) == self.get_base_velocity())

        # We check by computing the Jacobian (hopefully this only needs to be done once)
        if not self.joints:
            return False
        link_id = self.joints[0]
        jacobian = self.get_jacobian(link_id)

        # if floating base then the Jacobian will also include columns corresponding to the root link DoFs, while
        # with a fixed base, it will only have columns associated with the joints.
        if jacobian.shape[1] > len(self.joints):
            return True
        return False

    def has_floating_base(self):
        """
        Return True if the robot has a floating base (i.e. floating root link). Otherwise, it is a fixed base.

        Returns:
            bool: True if the robot has a floating base.
        """
        return self.floating_base

    def has_fixed_base(self):
        """
        Return True if the robot has a fixed base.

        Returns:
            bool: True if the robot has a fixed base.
        """
        return not self.has_floating_base()

    #######
    # CoM #
    #######

    def get_center_of_mass_position(self):
        """
        Return the center of mass position.

        Returns:
            np.float[3]: center of mass position
        """
        self.com = self.sim.get_center_of_mass_position(self.id)
        return self.com

    def get_center_of_mass_velocity(self):
        """
        Return the center of mass velocity.

        Returns:
            float[3]: center of mass velocity
        """
        return self.sim.get_center_of_mass_velocity(self.id)

    # def get_linear_momentum(self):
    #     """
    #     Compute the linear momentum around the center of mass.
    #
    #     .. math:: p = mv
    #
    #     where :math:`p` is the linear momentum, :math:`m` is the total mass, and :math:`v` is the velocity.
    #
    #     Returns:
    #         np.array[3]: linear momentum
    #     """
    #     return self.mass * self.get_base_linear_velocity()
    #
    # def get_angular_momentum(self):
    #     """
    #     Compute the angular momentum around the center of mass.
    #
    #     .. math:: h = I\omega
    #
    #     where :math:`h` is the angular momentum (based on the world origin), :math:`I` is the moment of inertia,
    #     and :math:`\omega` is the angular velocity.
    #
    #     Returns:
    #         np.array[3]: angular momentum
    #     """
    #     pass
    #
    # def get_centroidal_dynamics(self, q=None, dq=None):
    #     """
    #     Compute the centroidal momentum dynamics based on [1]. "The centroidal momentum of a rigid-body system
    #     consists of its net linear momentum as well as its net angular momentum about its center of mass (CoM)" [1]
    #
    #     Args:
    #         q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
    #             get the current joint positions (but note that this could lead to a decrease of performance).
    #         dq (float[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
    #             get the current joint velocities (but note that this could lead to a decrease of performance).
    #
    #     Returns:
    #         np.array[6, N+6]: centroidal momentum matrix :math:`A_G`
    #         np.array[6]: the dot product between the derivative of the centroidal momentum matrix with the
    #             generalized velocities vector. That is, :math:`\dot{A}_G \dot{q}`
    #
    #     References:
    #         [1] "Improved computation of the humanoid centroidal dynamics and application for whole-body control",
    #             Wensing and Orin, 2016
    #     """
    #     pass

    ########################
    # Joints (joint space) #
    ########################

    def get_joint_ids(self, joint=None):
        """
        Return the joint id(s) from the name(s) or q index(ices).

        Note that the joint id is unique and goes from 0 to the total number of joints (including fixed joints),
        while the q index goes from 0 to the number of actuated joints.

        Args:
            joint (str, int, list of str/int, None): if str, it will get the joint id associated to the given name.
                If int, it will get the joint id associated to the given q index. If it is a list of str and/or int,
                it will get the corresponding joint ids. If None, it will return all the (actuated) joint ids.

        Returns:
            if 1 joint:
                int: joint id
            if multiple joint:
                int[N]: joint ids
        """
        if joint is None:
            return self.joints

        def get_index(joint):
            if isinstance(joint, str):
                return self.joint_names[joint]
            elif isinstance(joint, int):
                return self.joints[joint]
            else:
                raise TypeError("Incorrect type")

        # list of joints
        if isinstance(joint, collections.Iterable) and not isinstance(joint, str):
            return [get_index(joint) for joint in joint]

        # one joint
        return get_index(joint)

    def get_joint_info(self, joint_ids=None):
        """
        Get information about the given joint(s).

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                [0] int:        the same joint id as the input parameter
                [1] str:        name of the joint (as specified in the URDF/SDF/etc file)
                [2] int:        type of the joint which implie the number of position and velocity variables.
                                The types include JOINT_REVOLUTE (=0), JOINT_PRISMATIC (=1), JOINT_SPHERICAL (=2),
                                JOINT_PLANAR (=3), and JOINT_FIXED (=4).
                [3] int:        q index - the first position index in the positional state variables for this body
                [4] int:        dq index - the first velocity index in the velocity state variables for this body
                [5] int:        flags (reserved)
                [6] float:      the joint damping value (as specified in the URDF file)
                [7] float:      the joint friction value (as specified in the URDF file)
                [8] float:      the positional lower limit for slider and revolute joints
                [9] float:      the positional upper limit for slider and revolute joints
                [10] float:     maximum force specified in URDF. Note that this value is not automatically used.
                                You can use maxForce in 'setJointMotorControl2'.
                [11] float:     maximum velocity specified in URDF. Note that this value is not used in actual
                                motor control commands at the moment.
                [12] str:       name of the link (as specified in the URDF/SDF/etc file)
                [13] float[3]:  joint axis in local frame (ignored for JOINT_FIXED)
                [14] float[3]:  joint position in parent frame
                [15] float[4]:  joint orientation in parent frame (x, y, z, w)
                [16] int:       parent link index, -1 for base

            if multiple joints: list of joint information (i.e. list of above)
        """
        if isinstance(joint_ids, int):
            return self.sim.get_joint_info(self.id, joint_ids)
        if joint_ids is None:
            joint_ids = self.joints
        return [self.sim.get_joint_info(self.id, joint_id) for joint_id in joint_ids]

    def get_joint_axes(self, joint_ids=None):
        """
        Get information about the given joint(s).

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the axis for all
                (actuated) joints.

        Returns:
            if 1 joint:
                np.float[3]: joint axis
            if multiple joint:
                [np.float[3]]: list of joint axis
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_axes(self.id, joint_ids)

    def get_q_indices(self, joint_ids=None):
        """
        Get the corresponding q index of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the q indices for all
                (actuated) joints.

        Returns:
            if 1 joint:
                int: q index
            if multiple joints:
                int[N]: q indices
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_q_indices(self.id, joint_ids)

    def get_joint_types(self, joint_ids=None, to_string=True):
        """
        Get the joint type as a string or integer.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.
            to_string (bool): if True, it will return the joint type in a readable string format

        Returns:
            if 1 joint:
                str/int: the name of the joint type, or the flag associated with it.
            if multiple joints: list of above
        """
        if joint_ids is None:
            joint_ids = self.joints
        if to_string:
            return self.sim.get_joint_type_names(self.id, joint_ids)
        return self.sim.get_joint_type_ids(self.id, joint_ids)

    def get_joint_limits(self, joint_ids=None):
        """
        Get the joint limits of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                np.float[2]: lower and upper limit
            if multiple joints:
                np.float[N,2]: lower and upper limit for each specified joint
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_limits(self.id, joint_ids)

    def get_joint_dampings(self, joint_ids=None):
        """
        Get the damping coefficient of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: damping coefficient of the given joint
            if multiple joints:
                float[N]: damping coefficient for each specified joint
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_dampings(self.id, joint_ids)

    def get_joint_frictions(self, joint_ids=None):
        """
        Get the friction coefficient of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: friction coefficient of the given joint
            if multiple joints:
                np.float[N]: friction coefficient for each specified joint
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_frictions(self.id, joint_ids)

    def get_joint_max_forces(self, joint_ids=None):
        """
        Get the maximum force that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: maximum force [N]
            if multiple joints:
                np.float[N]: maximum force for each specified joint [N]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_max_forces(self.id, joint_ids)

    def get_joint_max_velocities(self, joint_ids=None):
        """
        Get the maximum velocity that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: maximum velocity [rad/s]
            if multiple joints:
                float[N]: maximum velocities for each specified joint [rad/s]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_max_velocities(self.id, joint_ids)

    def get_joint_names(self, joint_ids=None):
        """
        Return the name of the given joint(s).

        Args:
            joint_ids (int, int[N]): joint id, or list of joint ids. If None, get the name of all (actuated) joints.

        Returns:
            if 1 joint:
                str: name of the joint
            if multiple joints:
                str[N]: name of each joint
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_names(self.id, joint_ids)

    def get_joint_states(self, joint_ids=None):
        """
        Get the state of the given joint(s).

        Args:
            joint_ids (int, int[N], None): id of the joint, or list of joint ids. If None, get the state of all
                (actuated) joints.

        Returns:
            for 1 joint:
                float: joint position [rad]
                float: joint velocity [rad/s]
                np.float[6]: joint reaction forces [fx,fy,fz,mx,my,mz]
                float: applied joint motor torque (during the last step)
            for multiple joints: list of each joint state
        """
        if isinstance(joint_ids, int):
            return self.sim.get_joint_state(self.id, joint_ids)
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_states(self.id, joint_ids)

    def get_joint_positions(self, joint_ids=None):
        """
        Get the position of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the position of all (actuated)
                joints.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.float[N]: joint positions [rad]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_positions(self.id, joint_ids)

    def get_joint_velocities(self, joint_ids=None):
        """
        Get the velocity of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the velocity of all (actuated)
                joints.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.float[N]: joint velocities [rad/s]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_velocities(self.id, joint_ids)

    def get_joint_accelerations(self, joint_ids=None):
        """
        Get the acceleration at the given joint(s). This is carried out by first getting the joint torques, then
        performing forward dynamics to get the joint accelerations from the joint torques.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the acceleration of all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: joint acceleration [rad/s^2]
            if multiple joints:
                np.float[N]: joint accelerations [rad/s^2]
        """
        # check joint id
        if joint_ids is None:
            joint_ids = self.joints

        # get the torques
        torques = self.get_joint_torques(joint_ids)

        # compute the accelerations
        accelerations = self.calculate_forward_dynamics(torques)

        # return the specified accelerations
        q_idx = self.get_q_indices(joint_ids)
        return accelerations[q_idx]

    def get_joint_reaction_forces(self, joint_ids=None):
        """
        Return the joint reaction forces at the given joint. Note that the torque sensor must be enabled, otherwise
        it will always return [0,0,0,0,0,0].

        Args:
            joint_ids (int, int[N], None): unique id of the joint, or list of joint ids. If None, get the joint
                reaction forces of all (actuated) joints.

        Returns:
            if 1 joint:
                np.float[6]: joint reaction force (fx,fy,fz,mx,my,mz) [N,Nm]
            if multiple joints:
                np.float[N,6]: joint reaction forces [N, Nm]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_reaction_forces(self.id, joint_ids)

    def get_joint_torques(self, joint_ids=None):
        """
        Get the applied torque on the given joint(s).

        Args:
            joint_ids (int, int[N], None): id of the joint, or list of joint ids. If None, get the joint torques of
                all (actuated) joints.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.float[N]: torques associated to the given joints [Nm]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_torques(self.id, joint_ids)

    def get_joint_powers(self, joint_ids=None):
        """
        Return the applied power at the given joint(s). Power = torque * velocity.

        Args:
            joint_ids (int, int[N], None): id of the joint, or list of joint ids. If None, get the joint powers of
                all (actuated) joints.

        Returns:
            if 1 joint:
                float: joint power [W]
            if multiple joints:
                np.float[N]: power at each joint [W]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_powers(self.id, joint_ids)

    # TODO: max_velocities and forces
    def set_joint_positions(self, positions, joint_ids=None, kp=None, kd=None, velocities=None, forces=None):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get all the actuated joints.
            positions (float, np.float[N]): desired position, or list of desired positions [rad]
            velocities (float, np.float[N], None): desired velocity, or list of desired velocities [rad/s]
            kp (float, np.float[N], None): position gain(s)
            kd (float, np.float[N], None): velocity gain(s)
            forces (float, np.float[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
        """
        if joint_ids is None:
            joint_ids = self.joints
        self.sim.set_joint_positions(self.id, joint_ids, positions, velocities=velocities, kps=kp, kds=kd,
                                     forces=forces)

        # if isinstance(joint_ids, int):
        #     kwargs = {}
        #     if kp is not None:
        #         kwargs['positionGain'] = kp
        #     if kd is not None:
        #         kwargs['velocityGain'] = kd
        #     if velocities is not None:
        #         kwargs['targetVelocity'] = velocities
        #     if forces is not None:
        #         kwargs['force'] = forces
        #     self.sim.setJointMotorControl2(self.id, joint_ids, self.sim.POSITION_CONTROL, position=positions,
        #                                    **kwargs)
        # else:
        #     if joint_ids is None:
        #         joint_ids = self.joints
        #     kwargs = {}
        #     if kp is not None:
        #         if isinstance(kp, (float, int)):
        #             kp = kp * np.ones(len(joint_ids))
        #         kwargs['positionGains'] = kp
        #     if kd is not None:
        #         if isinstance(kd, (float, int)):
        #             kd = kd * np.ones(len(joint_ids))
        #         kwargs['velocityGains'] = kd
        #     # qIdx = self.get_q_indices(jointId)
        #     # print("pos: ", position)
        #     # print(self.joint_limits[qIdx, 0], self.joint_limits[qIdx, 1])
        #     # TODO: the following clip causes an error... Check Minitaur...
        #     # position = np.clip(position, self.joint_limits[qIdx, 0], self.joint_limits[qIdx, 1])
        #     # kp = kp.tolist()
        #     # kd = kd.tolist()
        #     # print("pos: ", position)
        #     # print("kp: ", kp)
        #     # print("kd: ", kd)
        #     if velocities is not None:
        #         if isinstance(velocities, (float, int)):
        #             velocities = velocities * np.ones(len(joint_ids))
        #         kwargs['targetVelocities'] = velocities
        #     if forces is not None:
        #         if isinstance(forces, (float, int)):
        #             forces = forces * np.ones(len(joint_ids))
        #         kwargs['forces'] = forces
        #     self.sim.setJointMotorControlArray(self.id, joint_ids, self.sim.POSITION_CONTROL, positions=positions,
        #                                        **kwargs)

    # TODO: max_velocities and forces
    def set_joint_velocities(self, velocities, joint_ids=None, forces=None, max_velocity=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get all the actuated joints.
            velocities (float, float[N]): desired velocity, or list of desired velocities [rad/s]
            forces (float, np.float[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
            max_velocity (float, bool, None): if True, it will make sure that the given velocity(ies) are below their
                authorized maximum value(s) (inferred from the URDF, or set previously by the user). If you already
                did the check outside the method or if you don't want limits, set this variable to False.
        """
        if joint_ids is None:
            joint_ids = self.joints
        self.sim.set_joint_motor_control(self.id, joint_ids, self.sim.VELOCITY_CONTROL, velocities=velocities)

    # TODO: max_acceleration
    def set_joint_accelerations(self, accelerations, joint_ids=None, max_acceleration=True):
        """
        Set the acceleration of the given joint(s) (using force control). This is achieved by performing inverse
        dynamic which given the joint accelerations compute the joint torques to be applied.

        Args:
            accelerations (float, float[N]): desired joint acceleration, or list of desired joint accelerations [rad/s^2]
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get all the actuated joints.
            max_acceleration (bool, float, None): if True, it will make sure that the given acceleration(s) are below
                their authorized maximum value(s). If you already did the check outside the method or if you don't want
                limits, set this variable to False.
        """
        # check joint ids
        if joint_ids is None:
            joint_ids = self.joints
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        if isinstance(accelerations, (int, float)):
            accelerations = [accelerations]
        if len(accelerations) != len(joint_ids):
            raise ValueError("Expecting the desired accelerations to be of the same size as the number of joints; "
                             "{} != {}".format(len(accelerations), len(joint_ids)))

        # if joint accelerations vector is not the same size as the actuated joints
        if len(accelerations) != len(self.joints):
            q_idx = self.get_q_indices(joint_ids)
            acc = np.zeros(len(self.joints))
            acc[q_idx] = accelerations
            accelerations = acc

        # compute joint torques from Inverse Dynamics
        torques = self.calculate_inverse_dynamics(accelerations)

        # get corresponding torques
        if len(torques) != len(joint_ids):
            q_idx = self.get_q_indices(joint_ids)
            torques = torques[q_idx]

        # print("Robot - torques {} for joints {}".format(torques, jointId))

        # set the joint torques
        self.set_joint_torques(torques, joint_ids)

    def set_joint_torques(self, torque=None, joint_ids=None):
        """
        Set the torque to the given joint(s) (using force/torque control).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will set the joint torques to
                all (actuated) joints.
            torque (float, float[N], None): desired torque(s) to apply to the joint(s) [N]. If None, it will apply
                a torque of 0 to the given joint(s).
        """
        if isinstance(joint_ids, int):
            if torque is None:
                torque = 0
        else:
            if joint_ids is None:
                joint_ids = self.joints
            if not isinstance(joint_ids, collections.Iterable):
                raise TypeError("Expecting jointId to be a tuple, list, or numpy array, got instead "
                                "{}".format(type(joint_ids)))
            if torque is None:
                torque = [0]*len(joint_ids)
            elif isinstance(torque, (int, float)):
                torque = [torque]*len(joint_ids)

        self.sim.set_joint_motor_control(self.id, joint_ids, self.sim.TORQUE_CONTROL, forces=torque)

    def set_joint_motor_control(self, joint_ids, control_mode, **kwargs):
        r"""
        Set joint motor control.

        In position control:
        .. math:: error = Kp (x_{des} - x) + Kd (\dot{x}_{des} - \dot{x})

        In velocity control:
        .. math:: error = \dot{x}_{des} - \dot{x}

        Note that the maximum forces and velocities are not automatically used for the different control schemes.

        Args:
            joint_ids (int, int[N]): joint id, or list of joint ids
            control_mode (int): sim.VELOCITY_CONTROL (=0), sim.TORQUE_CONTROL (=1), sim.POSITION_CONTROL (=2)
            kwargs:
                positions (float, float[N]) (optional): target position of the joint (in position control) [rad]
                velocities (float, float[N]) (optional): target velocity of the joint (in position/velocity
                    control) [rad/s]
                forces (float, float[N]) (optional): in position/velocity control, this is the maximum force used
                    to reach the target value. In torque control, this is the force/torque to be applied.
                kp (float, float[N]) (optional): position gain :math:`Kp`
                kd (float, float[N]) (optional): velocity gain :math:`Kd`
                maxVelocity (float, float[N]) (optional): in position control, this limits the velocity to a maximum.
        """
        self.sim.set_joint_motor_control(self.id, joint_ids, control_mode, **kwargs)

    def disable_motor(self, joint_ids=None):
        """
        Disable the motor associated with the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will disable the motors of
                all actuated joints.
        """
        if joint_ids is None:
            joint_ids = self.joints
        self.sim.set_joint_motor_control(self.id, joint_ids, self.sim.VELOCITY_CONTROL, forces=[0] * len(joint_ids))

    def reset_joint_states(self, q=None, dq=None, joint_ids=None):
        """
        Reset the state of the robot.

        Warnings: This is only valid in the simulator, and note that calling this method overrides all physics
        simulation.
        """
        # check joint_ids
        if not joint_ids:
            joint_ids = self.joints
        if isinstance(joint_ids, int):
            joint_ids = [joint_ids]

        # check q
        if q is None:
            q = np.zeros(len(joint_ids))
        elif isinstance(q, (int, float)):
            q = [q]
        else:
            if len(q) != len(joint_ids):
                raise ValueError("The number of joint ids does not match up with the number of q's")

        # check dq
        if dq is None:
            dq = np.zeros(len(joint_ids))
        elif isinstance(dq, (int, float)):
            dq = [dq]
        else:
            if len(dq) != len(joint_ids):
                raise ValueError("The number of joint ids does not match with the number of dq's")

        # reset the joint state
        for joint_id, position, velocity in zip(joint_ids, q, dq):
            self.sim.reset_joint_state(self.id, joint_id, position, velocity)

    def get_home_joint_positions(self):
        """
        Return the joint positions for the home position defined by the user. This method has to be overwritten in
        the child class.
        """
        return np.zeros(self.num_actuated_joints)

    def set_joint_home_positions(self):
        """
        Set the joints to their home position defined by the user.
        """
        joint_positions = self.get_home_joint_positions()
        if joint_positions is not None:
            self.reset_joint_states(joint_positions)

    def move_joint_home_positions(self):
        """
        Move the joints to their home position defined by the user. This method can be overwritten in the child
        class.

        The difference between this method and the `setJointHomePosition` is that the latter directly (re)set the
        joints to their home position while this one moves the joints to their home position.
        """
        joint_positions = self.get_home_joint_positions()
        if joint_positions is not None:
            self.set_joint_positions(joint_positions)

    def set_joint_init_positions(self):
        self.set_joint_positions(self.init_joint_positions)

    ##################################
    # Links (task/operational space) #
    ##################################

    def get_link_ids(self, link=None):
        """
        Return the link id(s) from the name(s) or q index(ices).

        Note that the link id is unique and goes from 0 to the total number of links (including fixed links),
        while the q index goes from 0 to the number of links associated with actuated joints.

        Args:
            link (str, int, list of str/int, None): if str, it will get the link id associated to the given name.
                If int, it will get the link id associated to the given q index. If it is a list of str and/or int,
                it will get the corresponding link ids. If None, it will return all the link ids (associated to
                actuated joints).

        Returns:
            if 1 link:
                int: link id
            if multiple links:
                int[N]: link ids
        """
        if link is None:
            return self.joints

        def get_index(link):
            if isinstance(link, str):
                return self.link_names[link]
            elif isinstance(link, int):
                return self.joints[link]
            else:
                raise TypeError("Incorrect type")

        # list of links
        if isinstance(link, collections.Iterable) and not isinstance(link, str):
            return [get_index(link) for link in link]

        # one link
        return get_index(link)

    def get_parent_link_ids(self, link_ids=None):
        """
        Return the parent link of the given link(s)

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the state of all links
                associated to actuated joints.

        Returns:
            if 1 link:
                int: link id
            if multiple links:
                int[N]: link ids
        """
        if isinstance(link_ids, int):
            return self.sim.get_joint_info(self.id, link_ids)[-1]
        if link_ids is None:
            link_ids = self.joints
        return [self.sim.get_joint_info(self.id, link)[-1] for link in link_ids]

    def get_chain_link_ids(self, to_link_id, from_link_id=None):
        """
        Return the link ids that constitute the chain(s) that go(es) from `fromLinkId` to `toLinkId`.

        Args:
            to_link_id (int, int[M]): link id(s) that end(s) the chain(s).
            from_link_id (int, int[M], None): link id(s) that start(s) the chain(s). `fromLinkId` has to be a parent or
                ancestor of the `toLinkId`. If None, it will return the chain going from the base to the `toLinkId`.

        Returns:
            int[N], [int[N]]: chain(s) containing the link ids.
        """
        if from_link_id is None:
            if isinstance(to_link_id, collections.Iterable):
                from_link_id = [-1] * len(to_link_id)
            else:
                from_link_id = -1

        def get_chain(to_link, from_link):
            chain = [to_link]
            for link_id in chain:
                link_id = self.get_parent_link_ids(link_id)
                chain.append(link_id)
                if link_id == from_link:
                    break
            return chain[::-1]

        if isinstance(to_link_id, int):
            return get_chain(to_link_id, from_link_id)
        else:
            return [get_chain(to_link, from_link) for to_link, from_link in zip(to_link_id, from_link_id)]

    def get_link_states(self, link_ids=None, compute_link_velocity=True, compute_forward_kinematics=True):
        """
        Return the state of the given link(s).

        Warning: note that we do not convert the data here.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the state of all links
                associated to actuated joints.
            compute_link_velocity (bool): if True, the Cartesian world velocity will be computed and returned.
            compute_forward_kinematics (bool): if True, the Cartesian world position/orientation will be recomputed
                using forward kinematics.

        Returns:
            if 1 link:
                [0] np.float[3]: Cartesian position of center of mass
                [1] np.float[4]: Cartesian orientation of center of mass
                [2] np.float[3]: local position offset of inertial frame (CoM) expressed in the URDF link frame
                [3] np.float[4]: local orientation (quat. [x,y,z,w]) offset of the inertial frame expressed in URDF
                    link frame
                [4] np.float[3]: world position of the URDF link frame
                [5] np.float[4]: world orientation of the URDF link frame
                [6] np.float[3]: Cartesian world linear velocity
                [7] np.float[3]: Cartesian world angular velocity
            if multiple links: list of above
        """
        if isinstance(link_ids, int):  # one link
            return self.sim.get_link_state(self.id, link_ids, compute_velocity=compute_link_velocity,
                                           compute_forward_kinematics=compute_forward_kinematics)
        if link_ids is None:
            link_ids = self.joints
        return self.sim.get_link_states(self.id, link_ids, compute_velocity=compute_link_velocity,
                                        compute_forward_kinematics=compute_forward_kinematics)

    def get_link_names(self, link_ids=None):
        r"""
        Return the name of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the name of all links
                associated to actuated joints.

        Returns:
            if 1 link:
                str: link name
            if multiple links:
                str[N]: link names
        """
        if isinstance(link_ids, int):
            return self.sim.get_joint_info(self.id, link_ids)[12]
        if link_ids is None:
            link_ids = self.joints
        return [self.sim.get_joint_info(self.id, link)[12] for link in link_ids]

    def get_link_masses(self, link_ids=None):
        r"""
        Return the mass of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the mass of all the links
                (even of fixed links).

        Returns:
            if 1 link:
                float: mass of the given link
            else:
                np.float[N]: mass of each link
        """
        if isinstance(link_ids, int):
            return self.sim.get_dynamics_info(self.id, link_ids)[0]
        if link_ids is None:
            link_ids = list(range(self.num_links))
        return np.array([self.sim.get_dynamics_info(self.id, link)[0] for link in link_ids])

    def get_link_frames(self, link_ids=None, flatten=False):
        r"""
        Return the link frame position and orientation (expressed in the world space).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the frame position of all
                links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[3]: the link frame position in the world space
                np.float[4]: Cartesian orientation of the link frame [x,y,z,w]
            if multiple links:
                np.float[Nx3], np.float[N,3]: link frame position of each link in world space
                np.float[Nx4], np.float[N,4]: orientation of each link frame [x,y,z,w]

        """
        return self.get_link_frame_world_positions(link_ids, flatten), self.get_link_frame_world_orientations(link_ids,
                                                                                                              flatten)

    def get_link_frame_world_positions(self, link_ids=None, flatten=False):
        r"""
        Return the frame position (in the Cartesian world space coordinates) of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the frame position of all
                links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[3]: the link frame position in the world space
            if multiple links:
                np.float[Nx3], np.float[N,3]: link frame position of each link in world space
        """
        if isinstance(link_ids, int):
            return np.array(self.sim.get_link_state(self.id, link_ids)[4])
        if link_ids is None:
            link_ids = self.joints
        pos = np.array([self.sim.get_link_state(self.id, link)[4] for link in link_ids])
        if flatten:
            return pos.reshape(-1)  # 1D array
        return pos  # 2D array

    def get_link_frame_world_orientations(self, link_ids=None, flatten=False):
        r"""
        Return the frame orientation (in the Cartesian world space) of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the frame orientation of
                all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array of float numbers instead of an array of quaternion

        Returns:
            if 1 link:
                np.float[4]: Cartesian orientation of the link frame [x,y,z,w]
            if multiple links:
                np.float[Nx4], np.float[N,4]: orientation of each link frame [x,y,z,w]
        """
        if isinstance(link_ids, int):
            return self.sim.get_link_state(self.id, link_ids)[5]
        if link_ids is None:
            link_ids = self.joints
        orientation = np.array([self.sim.get_link_state(self.id, link)[5] for link in link_ids])
        if flatten:
            return orientation.reshape(-1)  # 1D array
        return orientation  # 2D array

    def get_link_world_positions(self, link_ids=None, flatten=True):
        r"""
        Return the CoM position (in the Cartesian world space coordinates) of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the position of all links
                associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[3]: the link CoM position in the world space
            if multiple links:
                np.float[Nx3], np.float[N,3]: CoM position of each link in world space
        """
        if isinstance(link_ids, int):
            if link_ids == -1:
                return self.get_base_position()
            return np.array(self.sim.get_link_state(self.id, link_ids)[0])
        if link_ids is None:
            link_ids = self.joints
        pos = np.array([self.sim.get_link_state(self.id, link)[0] for link in link_ids])
        if flatten:
            return pos.reshape(-1)  # 1D array
        return pos  # 2D array

    def get_link_positions(self, link_ids=None, wrt_link_id=None, flatten=True):
        r"""
        Return the link CoM position wrt the position of another link. By default, it is the base.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the position of all links
                associated to actuated joints.
            wrt_link_id (int, int[N], None): the other link id(s). If None, returns the position wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[3]: the link CoM position
            if multiple links:
                np.float[Nx3], np.float[N,3]: CoM position of each link
        """
        p1 = self.get_link_world_positions(link_ids, flatten=False)
        p0 = self.get_base_position() if wrt_link_id is None or wrt_link_id == -1 \
                else self.get_link_world_positions(wrt_link_id, flatten=False)
        p = (p1 - p0)
        if flatten:
            return p.reshape(-1)
        return p

    def get_link_world_orientations(self, link_ids=None, flatten=True):
        r"""
        Return the CoM orientation (in the Cartesian world space) of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the orientation of all
                links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array of float numbers instead of an array of quaternion

        Returns:
            if 1 link:
                np.float[4]: Cartesian orientation of the link CoM [x,y,z,w]
            if multiple links:
                float[Nx4], np.float[N,4]: CoM orientation of each link [x,y,z,w]
        """
        if isinstance(link_ids, int):
            return self.sim.get_link_state(self.id, link_ids)[1]
        if link_ids is None:
            link_ids = self.joints
        orientation = np.array([self.sim.get_link_state(self.id, link)[1] for link in link_ids])
        if flatten:
            return orientation.reshape(-1)
        return orientation  # 2D array

    def get_link_orientations(self, link_ids=None, wrt_link_id=None, flatten=True):
        r"""
        Return the link CoM orientation wrt the orientation of another link. By default, it is the base.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the orientation of all
                links associated to actuated joints.
            wrt_link_id (int, int[N], None): the other link id(s). If None, returns the orientation wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[4]: Cartesian orientation of the link CoM [x,y,z,w]
            if multiple links:
                float[Nx4], np.float[N,4]: CoM orientation of each link [x,y,z,w]
        """
        q1 = self.get_link_world_orientations(link_ids)
        if wrt_link_id is None or wrt_link_id == -1:
            q0 = get_quaternion_inverse(self.get_base_orientation())
        else:
            if isinstance(wrt_link_id, int):
                q0 = get_quaternion_inverse(self.get_link_world_orientations(wrt_link_id))
            else:
                q0 = np.array([get_quaternion_inverse(self.get_link_world_orientations(link)) for link in wrt_link_id])

        q = get_quaternion_product(q0, q1)
        if flatten:
            q.reshape(-1)
        return q

    def get_link_world_linear_velocities(self, link_ids=None, flatten=True):
        r"""
        Return the linear velocity of the link(s) expressed in the Cartesian world space coordinates.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the linear velocities of
                all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[3]: linear velocity of the link in the Cartesian world space
            if multiple links:
                np.float[Nx3], np.float[N,3]: linear velocity of each link
        """
        if isinstance(link_ids, int):
            return np.array(self.sim.get_link_state(self.id, link_ids, compute_velocity=True)[6])
        if link_ids is None:
            link_ids = self.joints
        vel = np.array([self.sim.get_link_state(self.id, link, compute_velocity=True)[6] for link in link_ids])
        if flatten:
            return vel.reshape(-1)  # 1D array
        return vel  # 2D array

    def get_link_world_angular_velocities(self, link_ids=None, flatten=True):
        r"""
        Return the angular velocity of the link(s) in the Cartesian world space coordinates.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the angular velocities of
                all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[3]: angular velocity of the link in the Cartesian world space
            if multiple links:
                np.float[Nx3], np.float[N,3]: angular velocity of each link
        """
        if isinstance(link_ids, int):
            return np.array(self.sim.get_link_state(self.id, link_ids, compute_velocity=True)[7])
        if link_ids is None:
            link_ids = self.joints
        vel = np.array([self.sim.get_link_state(self.id, link, compute_velocity=True)[7] for link in link_ids])
        if flatten:
            return vel.reshape(-1)  # 1d array
        return vel  # 2D array

    def get_link_world_velocities(self, link_ids=None, flatten=True):
        r"""
        Return the linear and angular velocities (expressed in the Cartesian world space coordinates) for the given
        link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the linear and angular
                velocities of all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[6]: linear and angular velocity of the link in the Cartesian world space
            if multiple links:
                float[Nx6], float[N,6]: linear and angular velocity of each link
        """
        if isinstance(link_ids, int):
            lin_vel, ang_vel = self.sim.get_link_state(self.id, link_ids, compute_velocity=True)[6:8]
            return np.array(lin_vel + ang_vel)
        if link_ids is None:
            link_ids = self.joints
        vel = []
        for link in link_ids:
            lin_vel, ang_vel = self.sim.get_link_state(self.id, link, compute_velocity=True)[6:8]
            vel.append(lin_vel + ang_vel)
        vel = np.array(vel)
        if flatten:
            return vel.reshape(-1)  # 1d array
        return vel  # 2D array

    def get_link_linear_velocities(self, link_ids=None, wrt_link_id=None, flatten=True):
        r"""
        Return the linear velocity of the given link(s) wrt the other specified link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the linear velocity of
                all links associated to actuated joints.
            wrt_link_id (int, int[N], None): the other link id(s). If None, returns the linear velocity wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[3]: the linear velocity of the given link wrt to the other link
            if multiple links:
                np.float[Nx3], np.float[N,3]: linear velocity of each link wrt to the other link(s)
        """
        v1 = self.get_link_world_linear_velocities(link_ids, flatten=False)
        v0 = self.get_base_linear_velocity() if wrt_link_id is None or wrt_link_id == -1 \
                else self.get_link_world_linear_velocities(wrt_link_id, flatten=False)
        v = (v1 - v0)
        if flatten:
            return v.reshape(-1)
        return v

    def get_link_angular_velocities(self, link_ids=None, wrt_link_id=None, flatten=True):
        r"""
        Return the angular velocity of the given link(s) wrt to the other specified link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the angular velocity of
                all links associated to actuated joints.
            wrt_link_id (int, int[N], None): the other link id(s). If None, returns the angular velocity wrt to the
                base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[3]: the angular velocity of the given link wrt to the other link
            if multiple links:
                np.float[Nx3], np.float[N,3]: angular velocity of each link wrt to the other link(s)
        """
        w1 = self.get_link_world_angular_velocities(link_ids, flatten=False)
        w0 = self.get_base_angular_velocity() if wrt_link_id is None or wrt_link_id == -1 \
                else self.get_link_world_angular_velocities(wrt_link_id, flatten=False)
        w = (w1 - w0)
        if flatten:
            return w.reshape(-1)
        return w

    def get_link_velocities(self, link_ids=None, wrt_link_id=None, flatten=True):
        r"""
        Return the linear and angular velocity of the given link(s) wrt to the other specified link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the angular velocity of
                all links associated to actuated joints.
            wrt_link_id (int, int[N], None): the other link id(s). If None, returns the angular velocity wrt to the
                base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.float[6]: the linear and angular velocity of the given link wrt to the other link
            if multiple links:
                np.float[Nx6], np.float[N,6]: linear and angular velocity of each link wrt to the other link(s)
        """
        v1 = self.get_link_world_velocities(link_ids, flatten=False)
        v0 = self.get_base_velocity() if wrt_link_id is None or wrt_link_id == -1 \
                else self.get_link_world_velocities(wrt_link_id, flatten=False)
        v = (v1 - v0)
        if flatten:
            return v.reshape(-1)
        return v

    def get_link_linear_accelerations(self, link_ids=None):
        pass

    def get_link_angular_accelerations(self, link_ids=None):
        pass

    def get_link_accelerations(self, link_ids=None):
        pass

    def get_link_contacts(self, link_ids):
        """
        Check if the given link(s) is/are in contact with something in the environment, and return all the contact
        points involving the given robot link(s).

        Warnings: note that in reality, you can't know if your link(s) is/are in contact with an object unless there
        is a sensor attached to it. However, this can be useful in simulation to optimize, for instance, trajectories.

        Args:
            link_ids (int, int[N]): link id, or list of desired link ids.

        Returns:
            if 1 link:
                list: list of contact points where each contact point has:
                    int: contact flag
                    int: unique id of body A (this should be the robot id)
                    int: unique id of body B
                    int: link index of body A (-1 for base, this should be the same as the given link)
                    int: link index of body B (-1 for base)
                    np.float[3]: contact position on A (in Cartesian world coordinates)
                    np.float[3]: contact position on B (in Cartesian world coordinates)
                    np.float[3]: contact normal on B pointing towards A
                    float: contact distance (positive for separation and negative for penetration)
                    float: normal force applied during the last simulation step
            if multiple links: list of above
        """
        if isinstance(link_ids, int):
            return self.sim.get_contact_points(body1=self.id, link1_id=link_ids)
        return [self.sim.get_contact_points(body1=self.id, link1_id=link) for link in link_ids]

    def set_link_positions(self, link_ids, position, orientation=None):
        """
        Set the position(s) of the given link(s) using inverse kinematics (IK).

        Warnings: be careful that at the end we get joint position(s) using IK, and thus if you are trying to set
        the position of multiple links that share some joints, you will get positions that are inconsistents.

        Args:
            link_ids (int, int[N]): link id, or list of desired link ids.
            position (np.float[3], [float[3]], float[N,3]):
            orientation (np.float[4], [float[4]], float[N,4]):
        """
        pass

    #################
    # End-Effectors #  # same interface than Links (but easier to manipulate) #
    #################

    @property
    def num_end_effectors(self):
        """
        Return the number of end-effectors.

        Returns:
            int: number of end-effectors
        """
        return len(self.end_effectors)

    def _set_end_effectors(self):
        """
        Set automatically the end-effector ids and names based on the URDF. Here, all the leaves of the robot
        kinematic tree will be considered as end-effectors. Thus, use it with caution.
        """
        if len(self.end_effectors) == 0:
            end_effectors = {}

            # go through all the joints/links
            for joint in range(self.num_joints):
                # get useful information from current joint/link
                info = self.sim.get_joint_info(self.id, joint)
                parent_idx, link_name = info[-1], info[-5]

                # add this link in the end-effectors dict
                end_effectors[joint] = link_name

                # remove parent index from the end-effectors dict if present
                end_effectors.pop(parent_idx, None)

            self.end_effectors = end_effectors.keys()
            self.end_effector_names = {name: idx for idx, name in end_effectors.items()}

    def get_end_effector_ids(self, end_effector=None):
        """
        Get the end effector ids from the name(s) or index(ices).

        Note that the end-effector id is unique and goes from 0 to the total number of end-effectors.

        Args:
            end_effector (str, int, list of str/int, None): if str, it will get the end-effector id associated to the
                given name. If int, it will get the end-effector id associated to the given q index. If it is a list
                of str and/or int, it will get the corresponding end-effector ids. If None, it will return all the
                end-effector ids.

        Returns:
            if 1 end-effector:
                int: end-effector id
            if multiple end-effectors:
                int[N]: end-effector ids
        """
        if end_effector is None:
            return self.end_effectors

        def get_index(link):
            if isinstance(link, str):
                return self.end_effector_names[link]
            elif isinstance(link, int):
                return self.end_effectors[link]
            else:
                raise TypeError("Incorrect type")

        # list of links
        if isinstance(end_effector, collections.Iterable) and not isinstance(end_effector, str):
            return [get_index(link) for link in end_effector]

        # one link
        return get_index(end_effector)

    ##############
    # Transforms #
    ##############

    @staticmethod
    def get_homogeneous_transform(position, orientation):
        r"""
        Return the Homogeneous transform matrix given the position vector and the orientation.

        Args:
            position (np.float[3]): position vector
            orientation (np.float[4], np.float[3,3], np.float[3]): orientation

        Returns:
            np.float[4,4]: homogeneous matrix
        """
        if isinstance(orientation, quaternion.quaternion):
            R = quaternion.as_rotation_matrix(orientation)
        else:
            orientation = np.array(orientation)
            if orientation.shape == (3,):  # RPY Euler angles
                R = get_matrix_from_rpy(orientation)
            elif orientation.shape == (4,):  # quaternion in the form [x,y,z,w]
                R = get_matrix_from_quaternion(orientation)
            elif orientation.shape == (3, 3):  # Rotation matrix
                R = orientation
            else:
                raise ValueError("Expecting a quaternion, RPY Euler angles, or rotation matrix")

        H = np.vstack((np.hstack((R, position.reshape(-1, 1))), np.array([[0, 0, 0, 1]])))
        return H

    ##############
    # Kinematics #
    ##############

    # TODO: allow to slice the Jacobian to only get what interests the user
    def get_jacobian(self, link_id, q=None, local_position=None):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q), J_{ang}(q)]^T`, such that:

        .. math:: v = [\dot{p}, \omega]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            link_id (int): link id.
            q (np.float[N], None): joint positions of size N, where N is the number of DoFs. If None, it will compute q
                based on the current joint positions.
            local_position (None, np.array[3]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.float[6,N], np.float[6,(6+N)]: full geometric (linear and angular) Jacobian matrix. The number of columns
                depends if the base is fixed or floating.
        """
        if q is None:
            q = self.get_joint_positions()
        else:
            if len(q) != len(self.joints):
                raise ValueError("The length of q ({}) is different from the number of DoFs"
                                 " ({}).".format(len(q), len(self.joints)))

        if isinstance(q, np.ndarray):
            q = q.tolist()  # Note that q has to be a list; it doesn't work if numpy array in Pybullet
        dq = [0]*len(self.joints)

        # specify point on the link
        if local_position is None:
            local_position = self.sim.get_link_state(self.id, link_id)[2]  # Link CoM position in the link frame

        # calculate full jacobian
        return self.sim.calculate_jacobian(self.id, link_id, local_position=local_position, q=q, dq=dq, des_ddq=dq)

    def get_linear_jacobian(self, link_id, q=None, local_position=None):
        r"""
        Return the full linear (geometric) Jacobian matrix :math:`J_{lin}(q)`, such that:

        .. math:: \dot{p} = J_{lin}(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            link_id (int): link id
            q (np.float[N]): joint positions of size N, where N is the number of DoFs. If None, it will compute q based
                on the current joint positions.
            local_position: the point on the specified link to compute the Jacobian (in link local coordinates around
                its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.float[3,N], np.float[3,(6+N)]: full linear geometric Jacobian matrix. The number of columns depends if
                the base is fixed or floating.
        """
        return self.get_jacobian(link_id, q, local_position)[:3]

    def get_angular_jacobian(self, link_id, q=None, local_position=None):
        r"""
        Return the full angular (geometric) Jacobian matrix :math:`J_{ang}(q)`, such that:

        .. math:: \omega = J_{ang}(q) \dot{q}

        where :math:`\omega` is the link angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            link_id (int): link id
            q (np.float[N]): joint positions of size N, where N is the number of DoFs. If None, it will compute q based
                on the current joint positions.
            local_position: the point on the specified link to compute the Jacobian (in link local coordinates around
                its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.float[3,N], np.float[3,(6+N)]: full angular geometric Jacobian matrix. The number of columns depends if
                the base is fixed or floating.
        """
        return self.get_jacobian(link_id, q, local_position)[3:]

    @staticmethod
    def get_jacobian_derivative_rpy_to_angular_velocity(rpy_angle):
        r"""
        Return the Jacobian that maps RPY angle rates to angular velocities, i.e. :math:`\omega = T(\phi) \dot{\phi}`.

        Warnings: :math:`T` is singular when the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}`

        Args:
            rpy_angle (np.float[3]): RPY Euler angles [rad]

        Returns:
            np.float[3,3]: Jacobian matrix that maps RPY angle rates to angular velocities.
        """
        r, p, y = rpy_angle
        T = np.array([[1., 0., np.sin(p)],
                      [0., np.cos(r), -np.cos(p) * np.sin(r)],
                      [0., np.sin(r), np.cos(p) * np.cos(r)]])
        return T

    @staticmethod
    def get_jacobian_derivative_zyz_to_angular_velocity(zyzAngle):
        r"""
        Return the Jacobian that maps ZYZ angle rates to angular velocities, i.e. :math:`\omega = T(\phi) \dot{\phi}`.

        Warnings: :math:`T` is singular when the angle associated with `Y` is :math:`0` or :math:`\pi`.

        Args:
            rpyAngle (np.float[3]): ZYZ Euler angles [rad]

        Returns:
            np.float[3,3]: Jacobian matrix that maps ZYZ angle rates to angular velocities.
        """
        z, y = zyzAngle[:2]
        T = np.array([[0., -np.sin(z), np.cos(z) * np.sin(y)],
                      [0., np.cos(z), np.sin(z) * np.sin(y)],
                      [1., 0., np.cos(y)]])
        return T

    def get_analytical_jacobian(self, jacobian, rpy_angle):
        r"""
        Return the analytical Jacobian :math:`J_{a}(q) = [J_{lin}(q), J_{\phi}(q)]^T`, which respects:

        .. math:: \dot{x} = [\dot{p}, \dot{\phi}]^T = J_{a}(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\phi` are the Euler angles
        representing the orientation of the link. In general, the derivative of the Euler angles is not equal to
        the angular velocity, i.e. :math:`\dot{\phi} \neq \omega`.

        The analytical and geometric Jacobian are related by the following expression:

        .. math::

            J_{a}(q) = \left[\begin{array}{cc}
                I_{3 \times 3} & 0_{3 \times 3} \\
                0_{3 \times 3} & T^{-1}(\phi)
                \end{array} \right] J(q)

        where :math:`T` is the matrix that respects: :math:`\omega = T(\phi) \dot{\phi}`.

        Warnings:
            - We assume that the Euler angles used are roll, pitch, yaw (RPY)
            - We currently compute the analytical Jacobian from the geometric Jacobian. If we assume that we use RPY
                Euler angles then T is singular when the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}.

        Args:
            jacobian (np.float[6,N], np.float[6,6+N]): full geometric Jacobian.
            rpy_angle (np.float[3]): RPY Euler angles

        Returns:
            np.float[6,N], np.foat[6,(6+N)]: the full analytical Jacobian. The number of columns depends if the base
                is fixed or floating.
        """
        T = self.get_jacobian_derivative_rpy_to_angular_velocity(rpy_angle)
        Tinv = np.linalg.inv(T)
        Ja = np.vstack((np.hstack((np.identity(3), np.zeros((3, 3)))),
                        np.hstack((np.zeros((3, 3)), Tinv)))).dot(jacobian)
        return Ja

    def get_angular_velocities_from_derivative_rpy(self, rpy_angle, dRPY):
        r"""
        Return the angular velocities :math:`\omega` from the derivative of RPY Euler angles \math:`\dot{\phi}`.
        These 2 quantities are related by the following equation:

        .. math:: \omega = T(\phi) \dot{\phi}

        where in the case we have RPY as Euler angles, the matrix :math:`T` is given by:

        .. math::

            T = \left[ \begin{array}{ccc}
                    1 & 0 & \sin(\theta_p) \\
                    0 & \cos(\theta_r) & - \cos(\theta_p) \sin(\theta_r) \\
                    0 & \sin(\theta_r) & \cos(\theta_p) \cos(\theta_r)
                \end{array} \right]

        Note that :math:`T` is singular when the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}`.

        Args:
            rpy_angle (np.float[3]): RPY Euler angles [rad]
            dRPY (np.float[3]): time derivative of RPY Euler angles [rad/s]

        Returns:
            np.float[3]: angular velocities [rad/s]
        """
        T = self.get_jacobian_derivative_rpy_to_angular_velocity(rpy_angle)
        return T.dot(dRPY)

    def get_derivative_rpy_from_angular_velocities(self, rpy_angle, angular_velocity):
        r"""
        Return the time derivative of RPY Euler angles :math:`\dot{\phi}` given the angular velocities :math:`\omega`.

        . .math:: \dot{\phi} = T^{-1}(\phi) \omega

        Warning: if the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}`, then :math:`T` is singular, and the
        corresponding angular velocities :math:`\omega` are not defined.

        Args:
            rpy_angle (np.float[3]): RPY Euler angles [rad]
            angular_velocity (np.float[3]): angular velocities [rad/s]

        Returns:
            np.float[3]: time derivative of RPY Euler angles [rad/s]

        Raises:
            LinAlgError: if singular configuration.
        """
        T = self.get_jacobian_derivative_rpy_to_angular_velocity(rpy_angle)
        Tinv = np.linalg.inv(T)
        return Tinv.dot(angular_velocity)

    @staticmethod
    def get_JJT(jacobian):
        r"""
        Given the Jacobian, it returns :math:`JJ^T`. This relation is used in many places in robotics.

        Args:
            jacobian (np.float[D,N]): Jacobian matrix

        Returns:
            np.float[D,D]: :math:`JJ^T`
        """
        return jacobian.dot(jacobian.T)

    @staticmethod
    def get_damped_least_squares_inverse(jacobian, damping_factor=0.01):
        r"""
        Return the damped least-squares (DLS) inverse, given by:

        .. math:: \hat{J} = J^T (JJ^T + k^2 I)^{-1}

        which can then be used to get joint velocities :math:`\dot{q}` from the cartesian velocities :math:`v`, using
        :math:`\dot{q} = \hat{J} v`.

        Args:
            jacobian (np.float[D,N]): Jacobian matrix
            damping_factor (float): damping factor

        Returns:
            np.float[N,D]: DLS inverse matrix
        """
        J, k = jacobian, damping_factor
        return (J.T).dot(np.linalg.inv(J.dot(J.T) + k**2 * np.identity(J.shape[0])))

    # alias
    getDLSInverse = get_damped_least_squares_inverse

    @staticmethod
    def get_pinv_jacobian(jacobian):
        r"""
        Return the right pseudo-inverse of the jacobian, i.e. :math:`J^\dagger = J^T(JJ^T)^{-1}`.

        Args:
            jacobian (np.float[D,N]): Jacobian matrix

        Returns:
            np.float[N,N]: right pseudo-inverse of the Jacobian
        """
        return np.linalg.pinv(jacobian)

    def get_null_space_projector(self, jacobian):
        r"""
        The null space projector :math:`P` is the matrix that projects any vectors to the null space of :math:`J`.
        This is given by: :math:`P = (I - J^\dagger J)`, where :math:`J^\dagger = J^T(JJ^T)^{-1}` is the right
        pseudo-inverse of the jacobian :math:`J`. This is notably used to perform inverse kinematics, where
        :math:`\dot{q} = J^\dagger v + P \dot{q}_0` with :math:`\dot{q}_0` representing arbitrary joint velocities.

        Args:
            jacobian (np.float[D,N]): Jacobian matrix

        Returns:
            np.float[N,N]: null space projector matrix
        """
        J = jacobian
        JJT = self.get_JJT(jacobian)
        I = np.identity(J.shape[1])
        return I - self.get_pinv_jacobian(J=J).dot(J)

    def compute_manipulability_measure(self, jacobian):
        r"""
        Compute the manipulability measure `w(q) = sqrt( det(J(q)J(q)^T) )`. This is useful to get a general sense
        about the manipulation ability of the manipulator. This term, for instance, vanishes at singular
        configurations (see [1]).

        Args:
            jacobian (np.float[D,N]): Jacobian matrix

        Returns:
            float: manipulability measure :math:`w(q)`

        References:
            [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010, chap 3.5 and 3.9
        """
        return np.sqrt(np.linalg.det(self.get_JJT(jacobian)))

    @staticmethod
    def in_singular_configuration(jacobian):
        r"""
        Return True if we are in a singular configuration.

        Singularities are interesting because (see [1]):
        - they represent configurations where the mobility of the manipulator is reduced
        - infinite solutions to the IK problem may exist
        - around them, small velocities in the task/operational space may cause large velocities in the joint space

        Args:
            jacobian (np.float[D,N]): Jacobian matrix

        Returns:
            bool: True if in a singular configuration

        References:
            [1] "Robotics: Modelling, Planning, and Control" (book), Siciliano et al., 2010, chap 3.3
        """
        # TODO: define close to singular configuration using SVD
        J = jacobian
        m = np.min(J.shape)
        r = np.linalg.matrix_rank(J)  # this uses SVD to compute the rank
        return r < m

    def get_joint_velocities_from_cartesian_velocities(self, jacobian, velocity):
        r"""
        Return the joint velocities :math:`\dot{q}` from the cartesian velocities :math:`v`.

        .. math:: \dot{q} = J^\dagger v

        where :math:`J^\dagger` is the right pseudo-inverse of J, i.e. :math:`J^\dagger = J^T(JJ^T)^{-1}`.

        Args:
            jacobain (np.float[3,N], np.float[6,N]): Jacobian matrix
            velocity (np.float[3], np.float[6]): linear and/or angular velocities

        Returns:
            np.float[N]: joint velocities
        """
        Jpinv = self.get_pinv_jacobian(jacobian)
        return Jpinv.dot(velocity)

    @staticmethod
    def get_cartesian_velocities_from_joint_velocities(jacobian, dq):
        r"""
        Return the Cartesian velocities :math:`v = [\dot{p}, \omega]^T` where :math:`\dot{p}` and :math:`\omega`
        are the linear and angular velocities, respectively.

        .. math:: v = J(q) \dot{q}

        Returns:
            np.float[6]: Cartesian linear and angular velocities
        """
        return jacobian.dot(dq)

    # TODO: implement IK for several links (also by exploiting the null space)
    def calculate_inverse_kinematics(self, link_id, position, orientation=None, lower_limits=None, upper_limits=None, 
                                     joint_ranges=None, rest_poses=None, joint_dampings=None, max_iters=1, 
                                     threshold=1e-4):
        r"""
        Compute the FULL Inverse kinematics; it will return a position for all the actuated joints.

        Args:
            link_id (int): end effector link index.
            position (np.float[3]): target position of the end effector (its link coordinate, not center of mass
                coordinate!). By default this is in Cartesian world space, unless you provide `q_curr` joint angles.
            orientation (np.float[4]): target orientation in Cartesian world space, quaternion [x,y,w,z]. If not
                specified, pure position IK will be used.
            lower_limits (np.float[N], list of N floats): lower joint limits. Optional null-space IK.
            upper_limits (np.float[N], list of N floats): upper joint limits. Optional null-space IK.
            joint_ranges (np.float[N], list of N floats): range of value of each joint.
            rest_poses (np.float[N], list of N floats): joint rest poses. Favor an IK solution closer to a given rest
                pose.
            joint_dampings (np.float[N], list of N floats): joint damping factors. Allow to tune the IK solution using
                joint damping factors.
            solver (int): p.IK_DLS (=0) or p.IK_SDLS (=1), Damped Least Squares or Selective Damped Least Squares, as
                described in the paper by Samuel Buss "Selectively Damped Least Squares for Inverse Kinematics".
            q_curr (np.float[N]): list of joint positions. By default PyBullet uses the joint positions of the body.
                If provided, the target_position and targetOrientation is in local space!
            max_iters (int): maximum number of iterations. Refine the IK solution until the distance between target
                and actual end effector position is below this threshold, or the `max_iters` is reached.
            threshold (float): residual threshold. Refine the IK solution until the distance between target and actual
                end effector position is below this threshold, or the `max_iters` is reached.

        Returns:
            np.float[M]: joint positions (for each actuated joint).
        """
        # calculate joint positions solving IK and return them
        return self.sim.calculate_inverse_kinematics(self.id, link_id, position=position, orientation=orientation,
                                                     lower_limits=lower_limits, upper_limits=upper_limits,
                                                     joint_ranges=joint_ranges, rest_poses=rest_poses,
                                                     joint_dampings=joint_dampings, max_iters=max_iters,
                                                     threshold=threshold)

    def hard_priorities(self, jacobians, task_velocities, method='backtrack'):
        r"""
        Return dq.

        Args:
            jacobians:
            task_velocities:
            methods: 'successive', 'augmented', 'backtrack'.

        Returns:

        """
        pass

    ############
    # Dynamics #
    ############

    def calculate_inverse_dynamics(self, des_ddq, dq=None, q=None):
        r"""
        Starting from the specified joint positions :math:`q` and velocities :math:`\dot{q}`, it computes the joint
        torques :math:`\tau` required to reach the desired joint accelerations :math:`\ddot{q}_{des}`. That is,
        :math:`\tau = ID(model, q, \dot{q}, \ddot{q}_{des})`.

        Specifically, it uses the rigid-body equation of motion in joint space given by (see [1]):

        .. math:: \tau = H(q)\ddot{q} + C(q,\dot{q})

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Normally, a more popular form of this equation of motion (in joint space) is given by:

        .. math:: H(q) \ddot{q} + S(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F

        which is the same as the first one with :math:`C = S\dot{q} + g(q) - J^T(q) F`. However, this last formulation
        is useful to understand what happens when we set some variables to 0.
        Assuming that there are no forces acting on the system, and giving desired joint accelerations of 0, this
        method will return :math:`\tau = S(q,\dot{q}) \dot{q} + g(q)`. If in addition joint velocities are also 0,
        it will return :math:`\tau = g(q)` which can for instance be useful for gravity compensation.

        For forward dynamics, which computes the joint accelerations given the joint positions, velocities, and
        torques (that is, :math:`\ddot{q} = FD(model, q, \dot{q}, \tau)`, this can be computed using
        :math:`\ddot{q} = H^{-1} (\tau - C)` (see also `computeFullFD`). For more information about different
        control schemes (position, force, impedance control and others), or about the formulation of the equation
        of motion in task/operational space (instead of joint space), check the references [1-4].

        Args:
            q (np.float[M]): joint positions
            dq (np.float[M]): joint velocities
            des_ddq (np.float[M]): desired joint accelerations

        Returns:
            np.float[M]: joint torques computed using the rigid-body equation of motion

        References:
            [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        # if the joint velocities and positions are not provided, read them
        if dq is None:
            dq = self.get_joint_velocities()
        if q is None:
            q = self.get_joint_positions()

        # return the joint torques to be applied for the desired joint accelerations
        return self.sim.calculate_inverse_dynamics(self.id, q, dq, des_ddq)

    def calculate_forward_dynamics(self, torques, dq=None, q=None):
        r"""
        Given the specified joint positions :math:`q` and velocities :math:`\dot{q}`, and joint torques :math:`\tau`,
        it computes the joint accelerations :math:`\ddot{q}`. That is, :math:`\ddot{q} = FD(model, q, \dot{q}, \tau)`.

        Specifically, it uses the rigid-body equation of motion in joint space given by (see [1]):

        .. math:: \ddot{q} = H(q)^{-1} (\tau - C(q,\dot{q}))

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Normally, a more popular form of this equation of motion (in joint space) is given by:

        .. math:: H(q) \ddot{q} + S(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F

        which is the same as the first one with :math:`C = S\dot{q} + g(q) - J^T(q) F`. However, this last formulation
        is useful to understand what happens when we set some variables to 0.
        Assuming that there are no forces acting on the system, and giving desired joint torques of 0, this
        method will return :math:`\ddot{q} = - H(q)^{-1} (S(q,\dot{q}) \dot{q} + g(q))`. If in addition
        the joint velocities are also 0, it will return :math:`\ddot{q} = - H(q)^{-1} g(q)` which are
        the accelerations due to gravity.

        For inverse dynamics, which computes the joint torques given the joint positions, velocities, and
        accelerations (that is, :math:`\tau = ID(model, q, \dot{q}, \ddot{q})`, this can be computed using
        :math:`\tau = H(q)\ddot{q} + C(q,\dot{q})`. For more information about different
        control schemes (position, force, impedance control and others), or about the formulation of the equation
        of motion in task/operational space (instead of joint space), check the references [1-4].

        Args:
            q (np.float[M]): joint positions
            dq (np.float[M]): joint velocities
            torques (np.float[M]): desired joint torques

        Returns:
            np.float[M]: joint accelerations computed using the rigid-body equation of motion

        References:
            [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        # if the joint velocities and positions are not provided, read them
        if dq is None:
            dq = self.get_joint_velocities()
        if q is None:
            q = self.get_joint_positions()

        # compute and return joint accelerations
        torques = np.array(torques)
        Hinv = np.linalg.inv(self.get_mass_matrix(q))
        C = self.calculate_inverse_dynamics(np.zeros(len(q)), dq=dq, q=q)
        acc = Hinv.dot(torques - C)
        return acc

    def get_mass_matrix(self, q=None, q_idx=None):
        """
        Return the mass/inertia matrix :math:`H(q)`.

        Warnings: If the base is floating, it will return a [6+N,6+N] inertia matrix, where N is the number of actuated
            joints. If the base is fixed, it will return a [N,N] inertia matrix

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            q_idx (slice, None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N).

        Returns:
            float[N,N], float[6+N,6+N], float[M,M]: inertia matrix
        """
        if q is None:
            q = self.get_joint_positions()
        else:
            if len(q) != self.num_dofs:
                raise ValueError("All the joint positions need to be given to this method. You can then slice the"
                                 "inertia matrix afterward.")

        # TODO: we need to get all the joints (even the fixed ones) --> need to test
        # make sure that we have all the joints even the fixed ones
        q_aug = np.zeros(self.num_joints)
        q_aug[self.joints] = q
        q_aug = q_aug.tolist()  # Note that pybullet doesn't accept numpy arrays here

        if q_idx is None:
            return np.array(self.sim.calculate_mass_matrix(self.id, q_aug))
        return np.array(self.sim.calculate_mass_matrix(self.id, q_aug))[q_idx, q_idx]

    def get_cartesian_inertia_matrix(self, H=None, Ja=None):
        """
        Return the cartesian inertia matrix.

        .. math:: H_{x}(q) = J_{a}^{-T}(q) H(q) J_{a}^{-1}(q)

        where :math:`H(q)` is the joint inertia matrix, :math:`J_{a}` is the analytical Jacobian, i.e. it
        respects the relation :math:`\dot{x} = [\dot{p} \dot{\phi}]^T = J_{a}(q) \dot{q}` where :math:`\phi` are the
        Euler angles. This is different from the geometric Jacobian :math:`J` which respects
        :math:`v = [\dot{p} \omega]^T = J(q) \dot{q}`, where :math:`\omega` are the angular velocities.

        Args:
            H (float[N,N], None): Joint inertia matrix. If None, it will be computed here (the q's then need to be
                provided).
            Ja (float[6,N], None): Analytical Jacobian. If None, it will be computed here (the q's then need to be
                provided and the link_id

        Returns:
            float[6,6]: Cartesian inertia matrix
        """
        Ja_inv = np.linalg.inv(Ja)
        return Ja_inv.T.dot(H).dot(Ja_inv)

    def get_kinetic_energy(self, q=None, dq=None, q_idx=None):
        """
        Return the kinetic energy due to the movement of the specified joint(s).

        .. math:: T(q,\dot{q}) = \frac{1}{2} \dot{q}^T H(q) \dot{q}

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (float[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).
            q_idx (slice, None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: kinetic energy
        """
        if dq is None:
            dq = self.get_joint_velocities()
            if q_idx is not None and len(dq) != len(q_idx):
                dq = dq[q_idx]
        H = self.get_mass_matrix(q, q_idx)
        return 1./2 * dq.dot(H.dot(dq))

    def get_gravity_potential_energy(self, q=None, q_idx=None, g=np.array((0., 0., -9.81))):
        """
        Return the potential energy due to gravity.

        .. math:: V(q) = - \sum_{i=1}^N m_{l_i} g^T p_{l_i}

        where :math:`l_i` represents the link `i`, :math:`m_l` is the mass of the link, :math:`g` is the gravity
        vector, and :math:`p_l` is the position of the link.

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. THIS IS CURRENTLY
                NOT USED, as we can get the link positions from the simulator (instead of using forward kinematics).
            q_idx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.
            g (np.float[3]): gravity vector.

        Returns:
            float: potential energy due to gravity
        """
        link_ids = list(range(self.num_links))
        p = self.get_link_world_positions(link_ids=link_ids, flatten=False)
        m = self.get_link_masses(link_ids=link_ids)
        if q_idx is not None:
            p = p[self.joints[q_idx]]
            m = m[self.joints[q_idx]]
        return np.sum((p.T * m).T * g)

    def get_potential_energy(self, q=None, dq=None, q_idx=None):
        """
        Return the potential energy of the system.

        WARNING: Note that we currently assume rigid body systems (thus rigid links). With this assumption, the
        potential energy is only due to gravitational forces. So, for now this is just an alias to
        `get_gravity_potential_energy`.

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (float[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).
            q_idx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: potential energy
        """
        return self.get_gravity_potential_energy(q, q_idx)

    def get_lagrangian(self, q=None, dq=None, q_idx=None):
        """
        Return the Lagrangian evaluate at the given configuration.

        .. math:: L(q, \dot{q}) = T(q, \dot{q}) - V(q)

        where :math:`T` and :math:`V` are the kinetic and potential energy respectively.

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (float[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).
            q_idx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: value of the Lagrangian
        """
        T = self.get_kinetic_energy(q=q, dq=dq, q_idx=q_idx)
        V = self.get_potential_energy(q=q, q_idx=q_idx)
        return T - V

    def apply_external_force(self, force, link_id=-1, position=(0., 0., 0.), frame=1):
        """
        Apply an external force on a body, or a link of the body. Note that after each simulation step, the external
        forces are cleared to 0.

        Warnings: This does not work when using `sim.setRealTimeSimulation(1)`.

        Args:
            force (float[3]): Cartesian forces to be applied on the body
            link_id (int): link id to apply the force, if -1 it will apply the force on the base
            position (float[3]): position on the link where the force is applied.
            frame (int): allows to specify the coordinate system of force/position. sim.LINK_FRAME (=1) for local
                link frame, and sim.WORLD_FRAME (=2) for world frame. By default, it is the link frame.
        """
        self.sim.apply_external_force(self.id, link_id, force, position, frame)

    def apply_external_torque(self, torque, link_id=-1, frame=1):
        """
        Apply an external torque on a body, or a link of the body. Note that after each simulation step, the external
        torques are cleared to 0.

        Warnings: This does not work when using `sim.setRealTimeSimulation(1)`.

        Args:
            torque (float[3]): Cartesian torques to be applied on the body
            link_id (int): link id to apply the torque, if -1 it will apply the torque on the base
            frame (int): allows to specify the coordinate system of torque. sim.LINK_FRAME (=1) for local
                link frame, and sim.WORLD_FRAME (=2) for world frame. By default, it is the link frame.
        """
        self.sim.apply_external_torque(self.id, link_id, force, frame)

    def get_joint_torques_from_cartesian_wrench(self, jacobian, wrench):
        """
        Return the joint torques from the given Cartesian wrench (=force and torque) using the provided Jacobian.

        .. math:: \tau = J^T(q) f

        where :math:`\tau` are the joint torques, :math:`f` is the wrench vector (i.e. it contains the forces/torques
        applied at the link), and :math:`J` is the geometric Jacobian.

        Returns:
            float[N]: joint torques [Nm]
        """
        return jacobian.T.dot(wrench)

    def get_cartesian_wrench_from_joint_torques(self, jacobian, torque):
        """
        Return the Cartesian wrench (=force and torque) from the given joint torques using the provided Jacobian.

        .. math:: f = J(J^TJ)^{-1} \tau

        where :math:`\tau` are the joint torques, :math:`f` is the wrench vector (i.e. it contains the forces/torques
        applied at the link), and :math:`J` is the geometric Jacobian.

        Returns:
            float[6]: forces and torques in the Cartesian world space [N,Nm]
        """
        J = jacobian
        return J.dot(np.linalg.inv(J.T.dot(J))).dot(torque)

    def enable_coriolis_and_gravity_compensation(self, enable=True):
        """
        Enable the gravity and Coriolis compensation when applying torques. This will automatically compute these
        terms and add them automatically to the given torques when using torque control.

        Args:
            enable (bool): If True, enable the gravity and Coriolis compensation when applying torques.
        """
        self.coriolis_and_gravity_compensation = enable

    def get_coriolis_and_gravity_compensation_torques(self, q=None, dq=None, qIdx=None):
        """
        Return the torques that need to be applied to the robot joints such that it compensates for gravity and
        Coriolis effects.

        From the equations of motion:

        .. math:: H(q) \ddot{q} + C(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F,

        we can see that if we set :math:`F` and :math:`\ddot{q}` to 0, then we have:

        .. math::  \tau = C(q,\dot{q}) \dot{q} + g(q).

        These are the torques that need to be applied to the robot joints to compensate for gravity and Coriolis
        effects.

        Args:
            q (float[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (float[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            qIdx (int[M], None): slice the torques at the given q indices (0 < M <= N).

        Returns:
            float[M]: joint torques to be applied [Nm]
        """
        if q is None:
            q = self.get_joint_positions()
        if dq is None:
            dq = self.get_joint_velocities()

        ddq = np.zeros(len(self.joints)).tolist()

        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(dq, np.ndarray):
            dq = dq.tolist()

        if qIdx is None:
            return self.sim.calculate_inverse_dynamics(self.id, q, dq, ddq)
        return self.sim.calculate_inverse_dynamics(self.id, q, dq, ddq)[qIdx]

    def get_gravity_compensation_torques(self, q=None, qIdx=None):
        if q is None:
            q = self.get_joint_positions()
        dq = np.zeros(len(q))
        return self.get_coriolis_and_gravity_compensation_torques(q, dq, qIdx)

    def apply_coriolis_and_gravity_compensation(self, q=None, dq=None, qIdx=None, external_torques=0.):
        """
        Apply Coriolis and Gravity Compensation; set the torques using torque control.

        The torques are given by:

        .. math::  \tau = C(q,\dot{q}) \dot{q} + g(q).

        Args:
            q (float[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (float[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            qIdx (int[M], None): slice the torques at the given q indices (0 < M <= N).
        """
        jointId = self.joints if qIdx is None else self.joints[qIdx]
        torques = self.get_coriolis_and_gravity_compensation_torques(q, dq, qIdx)
        self.set_joint_torques(jointId, torques + external_torques)

    # TODO: finish to implement the method + think about multiple links + think about dimensions
    def get_active_compliant_torques(self, q=None, dq=None, qIdx=None, jacobian=None, linkVelocity=None, link_ids=None, kd=60):
        """
        Return the torques that need to be applied to enable active compliance. This is done by enabling Coriolis
        and gravity compensation along with a damping force projected from the Cartesian space to the joint space.

        The torques to be applied are given by:

        .. math::  \tau = C(q,\dot{q}) \dot{q} + g(q) + J^T F

        where :math:`F = - D v` with :math:`v` are the Cartesian velocities, and :math:`D` is the damping factor.

        Args:
            q (float[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (float[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            qIdx (int[M], None): slice the torques at the given q indices (0 < M <= N).

        Returns:
            float[M]: joint torques to be applied [Nm]
        """
        if q is None:
            q = self.get_joint_positions()
        if dq is None:
            dq = self.get_joint_velocities()
        if jacobian is None:
            jacobian = self.get_jacobian(link_id, q)
        if linkVelocity is None:
            linkVelocity = self.get_link_world_velocities(link_id)
        if isinstance(kd, int):
            kd = kd * np.identity(6)

        torques = self.get_coriolis_and_gravity_compensation_torques(q, dq, qIdx)
        torques += jacobian.T.dot(-kd * linkVelocity)
        return torques

    # TODO: finish to implement the method
    def apply_active_compliance(self, q=None, dq=None, qIdx=None, external_torques=0.):
        """
        Apply active compliance; this is done by enabling Coriolis and gravity compensation along with a damping
        force projected from the Cartesian space to the joint space.

        Args:
            q (float[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (float[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            qIdx (int[M], None): slice the torques at the given q indices (0 < M <= N).
        """
        jointId = self.joints if qIdx is None else self.joints[qIdx]
        torques = self.get_active_compliant_torques(q, dq, qIdx)
        self.set_joint_torques(jointId, torques + external_torques)

    def get_impedance_torques(self, x=0, dx=0, ddx=0):
        """
        .. math:: F_{a} = H_m (\ddot{x} - \ddot{x}_d) + D_m (\dot{x} - \dot{x}_d) + K_m (x - x_d)
        """
        pass

    def apply_task_impedance_control(self):
        """
        .. math:: F_{a} = H_m (\ddot{x} - \ddot{x}_d) + D_m (\dot{x} - \dot{x}_d) + K_m (x - x_d)
        """
        pass

    def get_attractor_torques(self):
        """
        The torques to be applied are given by:

        .. math::  \tau = C(q,\dot{q}) \dot{q} + g(q) + J^T F

        where :math:`F = K(x_d - x) - D v` with :math:`x` and :math:`v` are the Cartesian position and velocities,
        and :math:`K` and :math:`D` are the stiffness and damping factor, respectively.
        """
        pass

    ######################
    # Symbolic Equations #
    ######################

    def get_symbolic_equations_of_motion(self):
        """
        This returns the symbolic equation of motions of the robot (using the URDF). Internally, this used the
        `sympy.mechanics` module.

        Returns:

        References:
            [1] `sympy.mechanics`: http://docs.sympy.org/latest/modules/physics/mechanics/index.html
            [2] https://github.com/pydy/pydy-tutorial-human-standing
            [3] https://github.com/pydy/pydy/tree/master/examples
        """
        pass

    def linearize_equations_of_motion(self, point=None):
        """
        Linearize the equation of motions around the given point. That is, instead of having :math:`\dot{x} = f(x,u)`
        where :math:`f` is in general a non-linear function, linearize it around a certain point.

        .. math:: \dot{x} = A x + B u

        where :math:`x` is the state vector, :math:`u` is the control input vector, and :math:`A` and :math:`B` are
        the matrices.

        Args:
            point:

        Returns:
            float[M,M]: :math:`A` matrix, where M is the size of the state vector
            float[M,N]: :math:`B` matrix, where N is the size of the input vector

        References:
            [1] "State-Space Representation of LTI Systems", Rowell, 2002 (handout):
                http://web.mit.edu/2.14/www/Handouts/StateSpace.pdf
            [2] "Time-Domain Solution of LTI State Equations", Rowell, 2002 (handout):
                http://web.mit.edu/2.14/www/Handouts/StateSpaceResponse.pdf
            [3] `sympy.mechanics`: http://docs.sympy.org/latest/modules/physics/mechanics/index.html
        """
        pass

    ###########
    # Sensors #
    ###########

    @property
    def num_sensors(self):
        """
        Return the total number of sensors.

        Returns:
            int: total number of sensors.
        """
        return len(self.sensors)

    def enable_joint_force_torque_sensor(self, joint_ids=None, enable=True):
        """
        Enable/disable the force/torque sensors of the specified joint(s).

        Warnings: Note that you should normally use a F/T sensor. However, enabling/disabling F/T sensors can be
        useful for debug among other things.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, enable/disable the F/T sensors
                on all the actuated joints.
            enable (bool): If True, it will enable the F/T sensors, otherwise it will disable them.
        """
        if joint_ids is None:
            joint_ids = self.joints
        self.sim.enable_joint_force_torque_sensor(self.id, joint_ids, enable=enable)

    def disable_joint_force_torque_sensor(self, joint_ids=None):
        """
        Disable the force/torque sensors of the specified joint(s).

        Warnings: Note that you should normally use a F/T sensor. However, enabling/disabling F/T sensors can be
        useful for debug among other things.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, disable the F/T sensors on all the
                actuated joints.
        """
        if joint_ids is None:
            joint_ids = self.joints
        self.sim.enable_joint_force_torque_sensor(self.id, joint_ids, enable=False)

    def get_sensors(self, idx=None):
        """
        Return the specified sensor.

        Args:
            idx (int): index of the sensor

        Returns:
            Sensor, Sensor[M]: return the specified sensor, or all the sensors
        """
        if idx is None:
            return self.sensors
        return self.sensors[idx]

    def get_imu(self, idx=0):
        pass

    def get_force_torque_sensor(self, idx=0):
        pass

    def has_camera(self):
        return False

    def get_camera(self, idx=0):
        pass

    def get_camera_image(self, idx=0):
        pass

    def get_main_camera(self):
        pass

    def get_main_camera_image(self):
        pass

    #############
    # Actuators #
    #############

    @property
    def num_actuators(self):
        """
        Return the total number of actuators.

        Returns:
            int: total number of actuators.
        """
        return len(self.actuators)

    def get_actuators(self, idx=None):
        """
        Return the specified actuator.

        Args:
            idx (int): index of the actuator.

        Returns:
            Actuator, Actuator[M]: return the specified actuator, or all the actuators
        """
        if idx is None:
            return self.actuators
        return self.actuators[idx]

    #######################
    # Contacts/Collisions #
    #######################

    def get_contacts(self):
        """
        Return all the contacts made by the robot.

        Warnings: note that in reality, you can't know if your link(s) is/are in contact with an object unless there
        is a sensor attached to it. However, this can be useful in simulation to optimize, for instance, trajectories.

        Returns:
            list: list of contact points where each contact point has:
                int: contact flag
                int: unique id of body A (this should be the robot id)
                int: unique id of body B
                int: link index of body A (-1 for base, this should be the same as the given link)
                int: link index of body B (-1 for base)
                float[3]: contact position on A (in Cartesian world coordinates)
                float[3]: contact position on B (in Cartesian world coordinates)
                float[3]: contact normal on B pointing towards A
                float: contact distance (positive for separation and negative for penetration)
                float: normal force applied during the last simulation step
        """
        return self.sim.get_contact_points(body1=self.id)

    #########
    # Debug #
    #########

    def _get_joint_type_str(self, idx):
        """
        Return the joint type as a string based on the flag.

        Args:
            idx (int): flag for the type of joint

        Returns:
            str: name of the joint type
        """
        return ['revolute', 'prismatic', 'spherical', 'planar', 'fixed', 'point2point', 'gear'][idx]

    def print_joint_info(self, joint_id):
        """
        Print information about the given joint.

        Args:
            joint_id (int): unique joint id.
        """
        joint = self.sim.get_joint_info(self.id, joint_id)
        print('joint index: {}'.format(joint[0]))
        print('joint name: {}'.format(joint[1]))
        print('joint type: {}'.format(self._get_joint_type_str(joint[2])))
        print('q index: {}'.format(joint[3]))
        print('qd index: {}'.format(joint[4]))
        print('joint damping: {}'.format(joint[6]))
        print('joint friction: {}'.format(joint[7]))
        print('joint lower limit: {}'.format(joint[8]))
        print('joint upper limit: {}'.format(joint[9]))
        print('joint max force: {}'.format(joint[10]))
        print('joint max velocity: {}'.format(joint[11]))
        print('associated link name: {}'.format(joint[12]))
        print('joint axis: {}'.format(joint[13]))
        print('position wrt parent frame: {}'.format(joint[14]))
        print('orientation wrt parent frame: {}'.format(joint[15]))
        print('parent link index: {}'.format(joint[16]))

    def print_link_info(self, link_id):
        """
        Print information about the given link. The information printed include the link frame position and
        orientation, its center of mass position and orientation, its dimensions, its mass, its local inertia
        diagonal, etc.

        Args:
            link_id (int): unique link id
        """
        state = self.sim.get_link_state(self.id, link_id)
        print('link name: {}'.format(self.sim.get_link_names(self.id, link_id)))
        print('link world position: {}'.format(state[0]))
        print('link world orientation: {}'.format(state[1]))
        print('link inertial frame position: {}'.format(state[2]))
        print('link inertial frame orientation: {}'.format(state[3]))
        print('world link frame position: {}'.format(state[4]))
        print('world link frame orientation: {}'.format(state[5]))
        print('world link linear velocity: {}'.format(state[6]))
        print('world link angular velocity: {}'.format(state[7]))

    def print_info(self):
        """
        Print general information about the robot.
        """
        print("\nRobot: {}".format(self))
        print("Number of DoFs: {}".format(self.num_dofs))
        print("Joint ids: {}".format(list(range(self.num_joints))))
        print("Actuated joint ids: {}".format(self.joints))
        print("Link names (associated with actuated joints): {}".format(self.get_link_names(self.joints)))
        print("End-effector names: {}".format(self.get_link_names(self.end_effectors)))
        print("Floating base? {}".format(self.has_floating_base()))
        print("Total mass = {} kg".format(self.mass))

    def add_joint_slider(self, joint_ids=None):
        """
        Add a slider for the given joint id.

        Args:
            joint_ids (int, str, list of str/int, None): if int, the id is between {0, N} where N=number of non-fixed
                joint. If str, the name of the joint. If list/tuple, it contains the id or name of the joints.
                If None, add a slider for each non-fixed joint.
        """
        # show debug visualizer
        self.sim.configure_debug_visualizer(self.sim.COV_ENABLE_GUI, 1)

        def getIndex(jnt):
            if isinstance(jnt, int): # joint id
                return jnt
            elif isinstance(jnt, str): # joint name
                return self.get_joint_ids(jnt)
            else:
                raise TypeError('Expecting a str or int for the joint: {}'.format(jnt))

        # get the joint indices
        if joint_ids is None:
            joint_ids = self.joints
        else:
            if isinstance(joint_ids, int):  # joint id
                joint_ids = [joint_ids]
            elif isinstance(joint_ids, str):  # joint name
                joint_ids = [self.get_joint_ids(joint_ids)]
            elif isinstance(joint_ids, collections.Iterable):
                joint_ids = [getIndex(jnt) for jnt in joint_ids]
            else:
                raise TypeError("jointId has to be a None, int, str, or a list/tuple of int/str.")

        # get information about the joints
        names = self.get_joint_names(joint_ids)
        limits = self.get_joint_limits(joint_ids)
        positions = self.get_joint_positions(joint_ids)
        lower_limits, upper_limits = limits[:, 0], limits[:, 1]
        # apply lower and upper limit
        lower_limits[lower_limits < -2 * np.pi] = -2. * np.pi
        upper_limits[upper_limits > 2 * np.pi] = 2. * np.pi

        # add sliders in PyBullet
        for i in range(len(joint_ids)):
            slider = self.sim.add_user_debug_parameter(names[i], lower_limits[i], upper_limits[i], positions[i])
            self.joint_sliders[joint_ids[i]] = slider

    def update_joint_slider(self):
        """
        Read the specified joint slider value, and set the robot's corresponding joint to this one
        using position control.
        """
        # for each slider
        for joint_id, slider in self.joint_sliders.items():
            # read joint value from slider
            pos = self.sim.read_user_debug_parameter(slider)

            # set joint position to the read value
            self.sim.set_joint_motor_control(self.id, joint_id, self.sim.POSITION_CONTROL, positions=pos)

    # alias
    read_joint_slider = update_joint_slider

    def remove_joint_slider(self, joint_ids=None):
        """
        Remove the specified joint slider(s).

        Args:
            joint_ids (int, str, list of str/int, None): if int, the id is between {0, N} where N=number of non-fixed
                joint. If str, the name of the joint. If list/tuple, it contains the id or name of the joints.
                If None, add a slider for each non-fixed joint.
        """
        def get_index(joint):
            if isinstance(joint, int):  # joint id
                return joint
            elif isinstance(joint, str):  # joint name
                return self.get_joint_ids(joint)
            else:
                raise TypeError('Expecting a str or int for the joint: {}'.format(joint))

        # get the joint indices
        if joint_ids is None:
            joint_ids = self.joints
        else:
            if isinstance(joint_ids, int):  # joint id
                joint_ids = [joint_ids]
            elif isinstance(joint_ids, str):  # joint name
                joint_ids = [self.get_joint_ids(joint_ids)]
            elif isinstance(joint_ids, collections.Iterable):
                joint_ids = [get_index(joint) for joint in joint_ids]
            else:
                raise TypeError("jointId has to be a None, int, str, or a list/tuple of int/str.")

        # remove sliders in pybullet
        for joint in joint_ids:
            if joint in self.joint_sliders:
                self.sim.remove_user_debug_item(self.joint_sliders[joint])
                self.joint_sliders.pop(joint)

        # if no sliders anymore, remove the debug visualizer
        self.sim.configure_debug_visualizer(self.sim.COV_ENABLE_GUI, 0)

    ####################
    # online plotting  # # WARNING: ALL THE FOLLOWING METHODS NEED A SIMULATOR IN WHICH TO RUN #
    ####################

    def plot_joint_positions(self, joint_ids=None):
        pass

    def plot_joint_velocities(self, joint_ids=None):
        pass

    def plot_joint_accelerations(self, joint_ids=None):
        pass

    def plot_com_position(self):
        pass

    def plot_com_velocity(self):
        pass

    def plot_com_acceleration(self):
        pass

    def plot_cartesian_positions(self, link_ids=None):
        pass

    def plot_cartesian_velocities(self, link_ids=None):
        pass

    def plot_cartesian_accelerations(self, link_ids=None):
        pass

    ########
    # draw # # WARNING: ALL THE FOLLOWING METHODS NEED A SIMULATOR IN WHICH TO RUN #
    ########

    def _draw_sphere(self, position, radius=0.1, color=(1, 1, 1, 1)):
        visual = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
        body = self.sim.create_body(mass=0, visual_shape_id=visual, position=position)
        return body

    def _draw_cylinder(self, position, orientation, radius=1, height=1, color=(1, 1, 1, 1)):
        visual = self.sim.create_visual_shape(self.sim.GEOM_CYLINDER, radius=radius, length=height, rgba_color=color)
        body = self.sim.create_body(mass=0., visual_shape_id=visual, position=position, orientation=orientation)
        return body

    def _draw_frame(self, position, orientation, radius, length):
        R = get_matrix_from_quaternion(orientation)

        x = R.dot(np.array([length/2., 0, 0])) + position
        y = R.dot(np.array([0, length/2., 0])) + position
        z = R.dot(np.array([0, 0, length/2.])) + position

        qx = np.array([0.707, 0, 0, 0.707])  # 90deg around x
        qy = np.array([0, 0.707, 0, 0.707])  # 90deg around y

        # draw x, y, z cylinders
        self._draw_cylinder(x, get_quaternion_product(orientation, qy), radius, length, color=(1, 0, 0, 1))
        self._draw_cylinder(y, get_quaternion_product(orientation, qx), radius, length, color=(0, 1, 0, 1))
        self._draw_cylinder(z, orientation, radius, length, color=(0, 0, 1, 1))

    def _draw_debug_box(self, aabb_min, aabb_max):
        (x0, y0, z0), (xf, yf, zf) = aabb_min, aabb_max
        self.sim.add_user_debug_line((x0, y0, z0), (x0, yf, z0), (1, 1, 1))
        self.sim.add_user_debug_line((x0, yf, z0), (x0, yf, zf), (1, 1, 1))
        self.sim.add_user_debug_line((x0, yf, zf), (x0, y0, zf), (1, 1, 1))
        self.sim.add_user_debug_line((x0, y0, zf), (x0, y0, z0), (1, 1, 1))

        self.sim.add_user_debug_line((xf, y0, z0), (xf, yf, z0), (1, 1, 1))
        self.sim.add_user_debug_line((xf, yf, z0), (xf, yf, zf), (1, 1, 1))
        self.sim.add_user_debug_line((xf, yf, zf), (xf, y0, zf), (1, 1, 1))
        self.sim.add_user_debug_line((xf, y0, zf), (xf, y0, z0), (1, 1, 1))

        self.sim.add_user_debug_line((x0, y0, z0), (xf, y0, z0), (1, 1, 1))
        self.sim.add_user_debug_line((x0, yf, z0), (xf, yf, z0), (1, 1, 1))
        self.sim.add_user_debug_line((x0, y0, zf), (xf, y0, zf), (1, 1, 1))
        self.sim.add_user_debug_line((x0, yf, zf), (xf, yf, zf), (1, 1, 1))

    def change_transparency(self, alpha=0.5):
        """
        Change the transparency of a robot.

        WARNING: THIS CAN CHANGE THE COLOR OF SOME LINKS IF THEY WERE NOT DEFINED IN THE URDF!!

        Args:
            alpha (float): alpha channel. 1 is opaque, and 0 is completely transparent.
        """
        for shapeId in self.visual_shapes:
            rgba = self.visual_shapes[shapeId]['color']
            rgba[-1] = alpha
            self.sim.change_visual_shape(self.id, shapeId, rgba_color=rgba)
            # print("Link {} - color: {}".format(link, rgba))

    def update_visual(self):
        """
        Update all visuals.
        """
        pass

    def compute_and_draw_com_position(self, radius=0.05, color=(1, 0, 0, 1)):
        """
        Compute the CoM and draw it as a sphere in the simulator.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot.
            color (float[4]): rgba color of the sphere. By default, it is red.

        Returns:
            float[3]: center of mass
        """
        self.get_center_of_mass_position()
        self.draw_com_position(radius=radius, color=color)
        return self.com

    def draw_com_position(self, radius=0.05, color=(1, 0, 0, 1)):
        """
        Draw the CoM in the simulator.

        WARNING: `get_center_of_mass_position()` must be called before calling this method. Otherwise, check the other
        method `compute_and_draw_com_position()`.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot
            color (float[4]): rgba color of the sphere. By default it is red.
        """
        if self.com_visual is None: # create visual shape if not already created
            com_visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
            self.com_visual = self.sim.create_body(mass=0, visual_shape_id=com_visual_shape, position=self.com)
        else:  # set CoM position
            self.sim.reset_base_pose(self.com_visual, self.com, [0, 0, 0, 1])

    def remove_com(self):
        """
        Remove the CoM from the simulator.
        """
        if self.com_visual is not None:
            self.sim.remove_body(self.com_visual)
            self.com_visual = None

    def get_projected_com_position(self, max_depth=5):
        """
        Get the projected center of mass position.

        WARNING: This method only works in the simulator!! It requires some knowledge about the environment.

        Args:
            max_depth (float): if there is an object more than max_depth, it is not considered

        Returns:
            float[3], None: position of the projected CoM, or None if it couldn't project the CoM
        """
        com = self.get_center_of_mass_position()
        object_id, _, _, hit_position, _ = self.sim.ray_test(com, com - np.array([0., 0., max_depth]))[0]
        if object_id >= 0:  # if there is a collision
            return hit_position  # = projected com
        else:
            return None

    def compute_and_draw_projected_com_position(self, radius=0.05, color=(0, 0, 1, 1)):
        """
        Compute and draw the projected center of mass.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot
            color (float[4]): rgba color of the sphere. By default it is blue.

        Returns:
            float[3], None: position of the projected CoM, or None if it couldn't project the CoM
        """
        projected_com = self.get_projected_com_position()
        if projected_com is not None:
            # if visual shape not already created, create one
            if self.projected_com_visual is None:
                visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
                self.projected_com_visual = self.sim.create_body(mass=0, visual_shape_id=visual_shape,
                                                                 position=projected_com)

            # otherwise update projected CoM position
            else:
                self.sim.reset_base_pose(self.projected_com_visual, projected_com, [0, 0, 0, 1])

        return projected_com

    def remove_projected_com(self):
        """
        Remove the projected CoM from the simulator.
        """
        if self.projected_com_visual is not None:
            self.sim.remove_body(self.projected_com_visual)
            self.projected_com_visual = None

    # def drawProjectedCoM(self, radius=0.05, color=(1,0,0,1)):
    #     """
    #     draw the projected CoM on the walking surface
    #     """
    #     pass

    def draw_link_coms(self, link_ids=None, scaling=1.):
        """
        Draw the CoM of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get links associated to
                actuated joints.
            scaling (float): scaling factor
        """
        if link_ids is None:
            link_ids = self.joints
        elif isinstance(link_ids, int):
            link_ids = [link_ids]

        for link in link_ids:
            if link in self.visual_shapes:
                pos = self.get_link_world_positions(link)
                dim = self.visual_shapes[link]['dimensions']
                # radius = min(dim) * scaling * 0.2
                radius = 0.01
                self._draw_sphere(pos, radius, color=(0, 0, 0, 1))

    def draw_link_frames(self, link_ids=None, scaling=1.):
        """
        Draw frames of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get links associated to
                actuated joints.
            scaling (float): scaling factor
        """
        if link_ids is None:
            link_ids = [-1] + self.joints
        elif isinstance(link_ids, int):
            link_ids = [link_ids]

        for link in link_ids:
            if link in self.visual_shapes:
                if link == -1:
                    pos, orientation = self.get_base_pose()
                else:
                    pos = self.get_link_frame_world_positions(link)
                    orientation = self.get_link_frame_world_orientations(link)
                dim = self.visual_shapes[link]['dimensions']
                # radius = min(dim) * scaling * 0.2
                radius = 0.005 * scaling
                # self._draw_sphere(pos, radius, color=(0,0,0,1))
                # length = 4*radius
                length = 0.05 * scaling
                self._draw_frame(pos, orientation, radius, length)

    def draw_joint_frames(self, joint_ids=None):
        """
        Draw (actuated) joint frame
        """
        pass

    def draw_bounding_boxes(self, link_ids=None):
        """
        Draw bounding box around the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get links associated to
                actuated joints.
        """
        if link_ids is None:
            link_ids = [-1] + self.joints
        elif isinstance(link_ids, int):
            link_ids = [link_ids]

        for link in link_ids:
            if link in self.visual_shapes:
                aabb_min, aabb_max = self.sim.get_aabb(self.id, link)
                self._draw_debug_box(aabb_min, aabb_max)

    def draw3d_ellipsoid(self, position, orientation=(0., 0., 0., 1.), scale=(1., 1., 1.), color=(0, 1, 0, 0.7)):
        """
        Draw a 3D ellipsoid in the simulator.

        Warnings: Currently, PyBullet doesn't support to load an ellipsoid, so we load from a mesh file.

        Args:
            position (float[3]): position in the world space
            orientation (float[4]): orientation in the world space
            scale (float[3]): scale in the (x,y,z) directions
            color (float[4]): RGBA color

        Returns:
            int: id of the ellipsoid
        """
        filename = os.path.dirname(__file__) + '/meshes/ellipsoid.obj'
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_MESH, filename=filename, mesh_scale=scale,
                                                    rgba_color=color)
        ellipsoid = self.sim.create_body(mass=0., visual_shape_id=visual_shape, position=position,
                                         orientation=orientation)
        return ellipsoid

    def get_ellipsoid_orientation_and_scale(self, X):
        # compute evecs and singular values
        _, S, V = np.linalg.svd(X)

        # compute orientation of ellipsoid
        v1, v2 = V[0], V[1]  # 2 most important evecs
        pitch = -np.arctan2(v1[2], v1[0])
        yaw = np.arctan2(v1[1], v1[0])
        # roll = np.array([[np.cos(yaw), -np.sin(yaw), 0],
        #                  [np.sin(yaw), np.cos(yaw), 0],
        #                  [0, 0, 1]]).dot(np.array([0,1,0]))
        roll = np.arccos(v2.dot(np.array([-np.sin(yaw), np.cos(yaw), 0])))
        orientation = get_quaternion_from_rpy([roll, pitch, yaw])

        # evals, evecs = np.linalg.eigh(X)
        # evals, evecs = evals[::-1], evecs[:,::-1]
        # #S, orientation = np.sqrt(evals), self.angular_converter.convertFrom(quaternion.from_rotation_matrix(evecs.T))
        #
        # print(V[0])
        # print(V[1])
        # print(evecs[:,0])
        # print(evecs[:,1])
        # print(S[0])
        # print(np.sqrt(evals)[0])
        # raw_input('enter')

        # normalize singular values for scaling
        scale = S/np.sum(S)
        for i in range(len(S)):
            if S[i] < 0.005:  # 5mm
                scale[i] = 0.005

        return orientation, scale

    def draw_velocity_manipulability_ellipsoid(self, link_id, Jlin=None, JJT=None, color=(0, 1, 0, 0.7)):
        """
        evecs of JJ^T = directions
        singular values of JJ^T = dimensions

        Args:
            link_id (int): link id. This will be used to check where to draw the ellipsoid.
            J (float[3,N], None): linear Jacobian matrix. It doesn't need to be provided if `JJT` is given.
            JJT (float[3,3], None): if None, it will compute it using the provided linear Jacobian matrix.

        Returns:
            int: id of the visual ellipsoid
        """
        if JJT is None:
            if Jlin is None:
                raise ValueError("Please provide the linear Jacobian matrix")
            JJT = self.get_JJT(Jlin)

        orientation, scale = self.get_ellipsoid_orientation_and_scale(JJT)

        # load ellipsoid
        position = self.get_link_world_positions(link_id)
        self.draw3d_ellipsoid(position, orientation, scale=scale, color=color)

    def draw_force_manipulability_ellipsoid(self, link_id, J=None, JJT=None):
        """
        Kineto-statics duality: direction with good velocity manipulability is obtained a direction along which poor
        force manipulability is obtained.

        evecs((JJ^T)^{-1})

        Args:
            link_id:
            J:
            JJT:
        """
        pass

    def update_manipulability_ellipsoid(self, link_id, ellipsoid_id):
        """
        Update the position, orientation, and scaling of the given manipulability ellipsoid.

        Warnings: currently, the bullet simulator do not allow to update the scale, only the position and orientation.
        """
        pass
        # self.sim.reset_base_pose(ellipsoid_id, position, orientation)

    def remove_manipulability_ellipsoid(self, ellipsoid_id):
        """
        Remove the given ellipsoid manipulability ellipsoid.

        Args:
            ellipsoid_id (int): id of the ellipsoid to remove
        """
        self.sim.remove_body(ellipsoid_id)
