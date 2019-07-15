#!/usr/bin/env python
"""Define the robot class from which all the other robots inherit from.

The robot can access to the simulator as the `World`. If the `Robot` class defined here is the first layer in
the inheritance hierarchy/tree then in the second layer, you have `Manipulator`, `LeggedRobot`, `WheeledRobot`,
`UAV`, etc.

Dependencies:
- `pyrobolearn.simulators`
- `pyrobolearn.utils`
"""

import os
import time
import copy
import collections
# import rbdl
import numpy as np
# import quaternion

from pyrobolearn.utils.transformation import *
from pyrobolearn.utils.manifold_utils import tensor_matrix_product, symmetric_matrix_to_vector, logarithm_map, \
    distance_spd
from pyrobolearn.robots.base import ControllableBody


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse (general)", "Leonel Rozo (manipulability)",
               "Noemie Jaquier (manipulability)", "Songyan Xin (centroidal dynamics)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Robot(ControllableBody):
    r"""Robot class.

    This is the class that all robots should inherit from. It contains all the useful methods to operate the robot,
    and has been implemented such that it is very generic.

    References:
        - [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
        - [2] "Springer Handbook of Robotics", Siciliano et al., 2008
        - [3] "Rigid Body Dynamics Algorithms", Featherstone, 2008
        - [4] "Symbolic differentiation of the velocity mapping for a serial kinematic chain", Bruyninck et al.,
              Mechanism and Machine Theory. 1996
        - [5] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
              http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        - [6] "Whole-body cooperative balancing of humanoid robot using COG Jacobian", Sugihara et al., IROS, 2002
        - [7] "Geometry-aware Tracking of Manipulability Ellipsoids", Jaquier et al., R:SS, 2018
        - [8] "Improved computation of the humanoid centroidal dynamics and application for whole-body control",
              Wensing and Orin, 2016
        - [9] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Xin, 2018
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1., *args, **kwargs):
        """
        Initialize the robot.

        Args:
            simulator: reference to the simulator such that the robot can access it.
            urdf (str): path to the URDF/MJCF file.
            position (np.array[3]): initial position.
            orientation (np.array[4]): initial orientation represented as a quaternion (x,y,z,w).
            fixed_base (bool, None): if True, the base of the robot will be fixed.
            scale (float): scaling factor.
        """
        # check parameters
        if position is None:
            position = (0., 0., 0)
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
            self.id = self.sim.load_urdf(urdf, position, orientation, use_fixed_base=fixed_base, scale=scale)

        # self.sim.configure_debug_visualizer(self.sim.COV_ENABLE_RENDERING, 1)

        # save the input parameters
        self.urdf = urdf
        self.fixed_base = fixed_base
        self.scale = scale

        # rescale if specified
        if scale != 1.0:
            # we rescale manually the mass and inertia matrices of each link
            for link in range(self.num_links):
                info = self.sim.get_dynamics_info(self.id, link)
                mass, local_inertia_diagonal = info[0], np.array(info[2])
                mass *= scale ** 3      # because the density is unchanged when scaling
                local_inertia_diagonal *= scale ** 5   # 5 = 3+2; 3 is for the mass, and 2 is for the distance: I~mr^2
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

        # State of the robot
        self._prev_state = {}
        self._state = {}
        self._prev_jacobian = {}
        self._jacobian = {}

        # set useful variables
        self.joints = []  # non-fixed joint/link indices in the simulator
        self.joint_names = {}  # joint name to id in the simulator
        self.link_names = {}  # link name to id in the simulator
        self.end_effectors = []  # end effector indices
        self.end_effector_names = {}  # end effector name to id in the simulator

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

        # sensors and actuators
        self.sensors = []  # list of sensors
        self.actuators = []  # list of actuators

    #############
    # Operators #
    #############

    def __copy__(self):
        """Return a shallow copy of the robot. This can be overridden in the child class."""
        return self.__class__(simulator=self.simulator, urdf=self.urdf, position=self.position,
                              orientation=self.orientation, fixed_base=self.fixed_base, scale=self.scale)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the robot. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        simulator = copy.deepcopy(self.simulator, memo)
        urdf = copy.deepcopy(self.urdf)
        position = copy.deepcopy(self.position)
        orientation = copy.deepcopy(self.orientation)
        robot = self.__class__(simulator=simulator, urdf=urdf, position=position, orientation=orientation,
                               fixed_base=self.fixed_base, scale=self.scale)

        # TODO: copy sensors and actuators

        memo[self] = robot
        return robot

    ##############
    # Properties #
    ##############

    @property
    def num_dofs(self):
        """Return the number of degrees of freedom (DoFs); that is, if the base is not fixed, 6 (= 3 degrees for
        translation + 3 degrees for orientation) + the joints that are not fixed.
        """
        if self.fixed_base:
            return len(self.joints)
        return 6 + len(self.joints)

    @property
    def num_free_joints(self):
        """Return the number of joints that are not fixed."""
        return len(self.joints)

    ###########
    # Methods #
    ###########

    def step(self):
        """Perform a step."""
        # update previous and current states
        self._prev_state = self._state
        self._prev_jacobian = self._jacobian
        self._state = {}
        self._jacobian = {}

        # sense
        self.sense()

    def sense(self):
        """Run all the sensors."""
        for sensor in self.sensors:
            sensor()

    def act(self):
        """Run all the actuators."""
        for actuator in self.actuators:
            actuator()

    ########
    # Base #
    ########

    def get_base_pose(self, concatenate=False):
        """
        Get base position and orientation (expressed as a quaternion [x,y,z,w]) with respect to the world frame.

        Args:
            concatenate (bool): if we should concatenate or not the position and orientation. By default, it doesn't
                concatenate because some users might wish to change the orientation representation.

        Returns:
            if concatenate:
                np.array[7]: concatenated position and orientation
            else:
                np.array[3]: position
                np.array[4]: orientation (x, y, z, w)
        """
        pose = self.sim.get_base_pose(self.id)
        if concatenate:
            return np.concatenate((pose[0], pose[1]))
        return pose

    def get_base_position(self):
        """
        Return the base position.

        Returns:
            np.array[3]: base position.
        """
        return self.sim.get_base_position(self.id)

    def get_base_orientation(self):
        """
        Get the base orientation in the form of a quaternion (x, y, z, w).

        Returns:
            quaternion (np.array[4]): base orientation in the form of a quaternion (x, y, z, w).
        """
        return self.sim.get_base_orientation(self.id)

    def get_base_velocity(self, concatenate=True):
        """
        Return the base linear and angular velocities.

        Args:
            concatenate (bool): if we should concatenate or not the linear and angular velocities.

        Returns:
            if concatenate:
                np.array[6]: linear and angular velocities of the base
            else:
                np.array[3]: linear velocity of the base
                np.array[3]: angular velocity of the base
        """
        # check if cached
        if 'vel' in self._state:
            lin_vel, ang_vel = self._state['vel'][:2]
        else:
            lin_vel, ang_vel = self.sim.get_base_velocity(self.id)
            self._state['vel'] = lin_vel, ang_vel, time.time()

        if concatenate:
            return np.concatenate((lin_vel, ang_vel))
        return lin_vel, ang_vel

    def get_base_linear_velocity(self):
        """
        Return the linear velocity of the base.

        Returns:
            np.array[3]: linear velocity of the base
        """
        return self.sim.get_base_linear_velocity(self.id)

    def get_base_angular_velocity(self):
        """
        Return the angular velocity of the base.

        Returns:
            np.array[3]: angular velocity of the base
        """
        return self.sim.get_base_angular_velocity(self.id)

    def get_base_spatial_velocity(self):
        """
        Return the base spatial velocity (which is the concatenation of the angular and linear velocity).

        Returns:
            np.array[6]: spatial velocity
        """
        lin_vel, ang_vel = self.get_base_velocity(concatenate=False)
        return np.concatenate((ang_vel, lin_vel))

    def get_base_acceleration(self, concatenate=True):
        """
        Return the linear and angular acceleration of the base. Some simulators does not provide the accelerations.
        If that is the case, then we use finite difference to compute it (calling this the first time will return a
        zero vector for the linear and angular acceleration).

        Returns:
            if concatenate:
                np.array[6]: concatenation of the linear and angular acceleration
            else:
                np.array[3]: linear acceleration
                np.array[3]: angular acceleration
        """
        # check if cached
        if 'acc' in self._state:
            acc = self._state['acc'][:2]
        else:
            # if the simulator keep in memory the accelerations, return it
            if self.sim.supports_acceleration():
                acc = self.sim.get_base_acceleration(self.id)
            else:  # else, use finite difference

                # get current base velocity and time
                if 'vel' not in self._state:
                    self.get_base_velocity(concatenate=False)
                lin_vel, ang_vel, t = self._state['vel']

                # if we did not cache the previous base velocity
                if 'vel' not in self._prev_state:
                    acc = np.zeros(3), np.zeros(3)
                else:
                    # retrieve previous base velocity and time
                    lin_vel_prev, ang_vel_prev, t_prev = self._prev_state['vel']

                    # compute time difference
                    if self.sim.use_real_time():  # if the simulator is in real-time mode
                        dt = (t - t_prev)
                    else:  # if we are stepping in the simulator
                        dt = self.sim.timestep

                    pos = self.get_base_position()

                    # compute base acceleration
                    ang_acc = (ang_vel - ang_vel_prev) / dt
                    lin_acc = (lin_vel - lin_vel_prev) / dt
                    lin_acc += np.cross(ang_acc, pos) + np.cross(ang_vel, np.cross(ang_vel, pos))
                    acc = (lin_acc, ang_acc)

            self._state['acc'] = [acc[0], acc[1], time.time()]

        # if we need to concatenate the accelerations
        if concatenate:
            return np.concatenate((acc[0], acc[1]))
        return acc

    def get_base_spatial_acceleration(self):
        """
        Return the base spatial acceleration (which is the concatenation of the angular and linear acceleration).

        Returns:
            np.array[6]: spatial acceleration
        """
        lin_acc, ang_acc = self.get_base_acceleration(concatenate=False)
        return np.concatenate((ang_acc, lin_acc))

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
        # return self.floating_base
        return not self.fixed_base

    def has_fixed_base(self):
        """
        Return True if the robot has a fixed base.

        Returns:
            bool: True if the robot has a fixed base.
        """
        # return not self.has_floating_base()
        return self.fixed_base

    #######
    # CoM #
    #######

    def get_center_of_mass_position(self):
        """
        Return the center of mass position.

        Returns:
            np.array[3]: center of mass position
        """
        self.com = self.sim.get_center_of_mass_position(self.id)
        return self.com

    # alias
    get_com_position = get_center_of_mass_position

    def get_center_of_mass_velocity(self):
        """
        Return the center of mass velocity.

        Returns:
            np.array[3]: center of mass velocity
        """
        return self.sim.get_center_of_mass_velocity(self.id)

    # alias
    get_com_velocity = get_center_of_mass_velocity

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

    ########################
    # Joints (joint space) #
    ########################

    def get_joint_ids(self, joint=None):
        r"""
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
        r"""
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
                [13] np.array[3]:  joint axis in local frame (ignored for JOINT_FIXED)
                [14] np.array[3]:  joint position in parent frame
                [15] np.array[4]:  joint orientation in parent frame (x, y, z, w)
                [16] int:       parent link index, -1 for base

            if multiple joints: list of joint information (i.e. list of above)
        """
        if isinstance(joint_ids, int):
            return self.sim.get_joint_info(self.id, joint_ids)
        if joint_ids is None:
            joint_ids = self.joints
        return [self.sim.get_joint_info(self.id, joint_id) for joint_id in joint_ids]

    def get_joint_axes(self, joint_ids=None):
        r"""
        Get information about the given joint(s).

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the axis for all
                (actuated) joints.

        Returns:
            if 1 joint:
                np.array[3]: joint axis
            if multiple joint:
                [np.array[3]]: list of joint axis
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_axes(self.id, joint_ids)

    def get_q_indices(self, joint_ids=None):
        r"""
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
        r"""
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
        r"""
        Get the joint limits of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                np.array[2]: lower and upper limit
            if multiple joints:
                np.array[N,2]: lower and upper limit for each specified joint
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_limits(self.id, joint_ids)

    def get_joint_dampings(self, joint_ids=None):
        r"""
        Get the damping coefficient of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: damping coefficient of the given joint
            if multiple joints:
                np.array[N]: damping coefficient for each specified joint
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_dampings(self.id, joint_ids)

    def get_joint_frictions(self, joint_ids=None):
        r"""
        Get the friction coefficient of the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: friction coefficient of the given joint
            if multiple joints:
                np.array[N]: friction coefficient for each specified joint
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_frictions(self.id, joint_ids)

    def get_joint_max_forces(self, joint_ids=None):
        r"""
        Get the maximum force that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: maximum force [N]
            if multiple joints:
                np.array[N]: maximum force for each specified joint [N]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_max_forces(self.id, joint_ids)

    def get_joint_max_velocities(self, joint_ids=None):
        r"""
        Get the maximum velocity that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: maximum velocity [rad/s]
            if multiple joints:
                np.array[N]: maximum velocities for each specified joint [rad/s]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_max_velocities(self.id, joint_ids)

    def get_joint_names(self, joint_ids=None):
        r"""
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
        r"""
        Get the state of the given joint(s).

        Args:
            joint_ids (int, int[N], None): id of the joint, or list of joint ids. If None, get the state of all
                (actuated) joints.

        Returns:
            for 1 joint:
                float: joint position [rad]
                float: joint velocity [rad/s]
                np.array[6]: joint reaction forces [fx,fy,fz,mx,my,mz]
                float: applied joint motor torque (during the last step)
            for multiple joints: list of each joint state
        """
        if isinstance(joint_ids, int):
            return self.sim.get_joint_state(self.id, joint_ids)
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_states(self.id, joint_ids)

    def get_joint_positions(self, joint_ids=None):
        r"""
        Get the position of the given joint(s).

        See Also: :func:`~Robot.get_augmented_joint_positions`.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the position of all (actuated)
                joints.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[N]: joint positions [rad]
        """
        # check if cached
        if 'q' in self._state:
            # get cached joint positions
            q = self._state['q'][0]
        else:
            # get joint positions and cache it
            q = self.sim.get_joint_positions(self.id, self.joints)
            self._state['q'] = [q, time.time()]

        # return joint positions
        if joint_ids is None:
            return q
        return q[self.get_q_indices(joint_ids)]

    def get_augmented_joint_positions(self, joint_ids=None):
        r"""
        Get the augmented joint position vector of the specified joint(s). If the robot has a floating base, the first
        6 joints are the 3D world position and orientation (expressed as roll-pitch-yaw angles) of the robot base.
        If the robot has a fixed base, this is the same as calling :func:`~Robot.get_joint_positions`.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the position of all (actuated)
                joints.

        Returns:
            if 1 joint:
                float, np.array[6+1]: joint position(s) [rad]
            if multiple joints:
                np.array[N], np.array[6+N]: joint positions [rad]
        """
        q = self.get_joint_positions(joint_ids=joint_ids)
        if self.has_fixed_base():
            return q
        pose = self.get_base_pose(concatenate=False)
        pos, rpy = pose[0], get_rpy_from_quaternion(pose[1])
        return np.concatenate((np.concatenate((pos, rpy)), np.asarray(q).reshape(-1)))

    def get_joint_velocities(self, joint_ids=None):
        r"""
        Get the velocity of the given joint(s).

        See Also: :func:`~Robot.get_augmented_joint_velocities`.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the velocity of all (actuated)
                joints.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.array[N]: joint velocities [rad/s]
        """
        # check if cached
        if 'dq' in self._state:
            # get cached joint velocities
            dq = self._state['dq'][0]
        else:
            # get joint velocities and cache it
            dq = self.sim.get_joint_velocities(self.id, self.joints)
            self._state['dq'] = [dq, time.time()]

        # return joint velocities
        if joint_ids is None:
            return dq
        return dq[self.get_q_indices(joint_ids)]

    def get_augmented_joint_velocities(self, joint_ids=None):
        r"""
        Get the augmented joint velocity vector of the specified joint(s). If the robot has a floating base, the first
        6 joints are the 3D world linear and angular velocities of the robot base. If the robot has a fixed base, this
        is the same as calling :func:`~Robot.get_joint_velocities`.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the velocity of all (actuated)
                joints.

        Returns:
            if 1 joint:
                float, np.array[6+1]: joint velocity [rad/s]
            if multiple joints:
                np.array[N], np.array[6+N]: joint velocities [rad/s]
        """
        dq = self.get_joint_velocities(joint_ids=joint_ids)
        if self.has_fixed_base():
            return dq
        velocity = self.get_base_velocity(concatenate=True)
        return np.concatenate((velocity, np.asarray(dq).reshape(-1)))

    # def get_joint_accelerations(self, body_id, joint_ids, q=None, dq=None):
    #     """
    #     Get the acceleration of the specified joint(s). This is carried out by first getting the joint torques, then
    #     performing forward dynamics to get the joint accelerations from the joint torques.
    #
    #     Args:
    #         body_id (int): unique body id.
    #         joint_ids (int, list of int): joint id, or list of joint ids.
    #         q (list of int, None): all the joint positions. If None, it will compute it.
    #         dq (list of int, None): all the joint velocities. If None, it will compute it.
    #
    #     Returns:
    #         if 1 joint:
    #             float: joint acceleration [rad/s^2]
    #         if multiple joints:
    #             np.array[N]: joint accelerations [rad/s^2]
    #     """
    #     # check joint id
    #     if joint_ids is None:
    #         joint_ids = self.joints
    #
    #     # if simulator supports accelerations
    #     if self.sim.supports_acceleration():
    #         return self.sim.get_joint_accelerations()
    #
    #     # get the torques
    #     torques = self.get_joint_torques(joint_ids)
    #
    #     # compute the accelerations
    #     accelerations = self.calculate_forward_dynamics(torques)
    #
    #     # return the specified accelerations
    #     q_idx = self.get_q_indices(joint_ids)
    #     return accelerations[q_idx]

    def get_joint_accelerations(self, joint_ids=None):  # TODO: fix this!!
        r"""
        Get the acceleration of the specified joint(s). If the simulator doesn't provide the joint accelerations, this
        is computed using finite difference :math:`\ddot{q}(t) = \frac{\dot{q}(t) - \dot{q}(t-dt)}{dt}`.

        Warnings: if we use finite difference, note that the first time this method is called, it will return a zero
            vector because we do not have previous joint velocities (i.e. :math:`\dot{q}(t-dt)`) yet.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the acceleration of all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: joint acceleration [rad/s^2]
            if multiple joints:
                np.array[N]: joint accelerations [rad/s^2]
        """
        # check if cached
        if 'ddq' in self._state:
            ddq = self._state[0]
            if joint_ids is None:
                return ddq
            return ddq[self.get_q_indices(joint_ids)]

        # check joint id
        was_none = False
        if joint_ids is None:
            was_none = True
            joint_ids = self.joints

        # if simulator supports accelerations
        if self.sim.supports_acceleration():
            return self.sim.get_joint_accelerations(self.id, joint_ids)

        # else, use finite difference

        # get current joint velocities and time
        if 'dq' in self._state:
            dq, t = self._state['dq']
        else:
            dq, t = self.get_joint_velocities(), time.time()
            self._state['dq'] = [dq, t]

        # if we did not cache the previous joint velocities
        if 'dq' not in self._prev_state:
            self._state['ddq'] = [np.zeros(len(self.joints)), t]
            if isinstance(joint_ids, int):
                return 0
            return np.zeros(len(joint_ids))

        # retrieve previous joint velocities and time
        dq_prev, t_prev = self._prev_state['dq']

        # compute time difference
        if self.sim.use_real_time():  # if the simulator is in real-time mode
            dt = (t - t_prev)
        else:  # if we are stepping in the simulator
            dt = self.sim.timestep

        # compute joint accelerations using finite difference, and cache it
        ddq = (dq - dq_prev) / dt
        self._state['ddq'] = [ddq, t]

        # return joint accelerations
        if was_none:
            return ddq
        q_idx = self.get_q_indices(joint_ids)
        return ddq[q_idx]

    def get_augmented_joint_accelerations(self, joint_ids=None):
        r"""
        Get the augmented joint acceleration vector of the specified joint(s). If the robot has a floating base, the
        first 6 joints are the 3D world linear and angular accelerations of the robot base. If the robot has a fixed
        base, this is the same as calling :func:`~Robot.get_joint_accelerations`.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get the acceleration of all
                (actuated) joints.

        Returns:
            if 1 joint:
                float, np.array[6+1]: joint acceleration [rad/s^2]
            if multiple joints:
                np.array[N], np.array[6+N]: joint accelerations [rad/s^2]
        """
        ddq = self.get_joint_accelerations(joint_ids=joint_ids)
        if self.has_fixed_base():
            return ddq
        acceleration = self.get_base_acceleration(concatenate=True)
        return np.concatenate((acceleration, np.asarray(ddq).reshape(-1)))

    def get_joint_reaction_forces(self, joint_ids=None):
        r"""
        Return the joint reaction forces at the given joint. Note that the torque sensor must be enabled, otherwise
        it will always return [0,0,0,0,0,0].

        Args:
            joint_ids (int, int[N], None): unique id of the joint, or list of joint ids. If None, get the joint
                reaction forces of all (actuated) joints.

        Returns:
            if 1 joint:
                np.array[6]: joint reaction force (fx,fy,fz,mx,my,mz) [N,Nm]
            if multiple joints:
                np.array[N,6]: joint reaction forces [N, Nm]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_reaction_forces(self.id, joint_ids)

    def get_joint_torques(self, joint_ids=None):
        r"""
        Get the applied torque on the given joint(s).

        Args:
            joint_ids (int, int[N], None): id of the joint, or list of joint ids. If None, get the joint torques of
                all (actuated) joints.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.array[N]: torques associated to the given joints [Nm]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_torques(self.id, joint_ids)

    def get_joint_powers(self, joint_ids=None):
        r"""
        Return the applied power at the given joint(s). Power = torque * velocity.

        Args:
            joint_ids (int, int[N], None): id of the joint, or list of joint ids. If None, get the joint powers of
                all (actuated) joints.

        Returns:
            if 1 joint:
                float: joint power [W]
            if multiple joints:
                np.array[N]: power at each joint [W]
        """
        if joint_ids is None:
            joint_ids = self.joints
        return self.sim.get_joint_powers(self.id, joint_ids)

    # TODO: max_velocities and forces
    def set_joint_positions(self, positions, joint_ids=None, kp=None, kd=None, velocities=None, forces=None):
        r"""
        Set the position of the given joint(s) (using position control).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get all the actuated joints.
            positions (float, np.array[N]): desired position, or list of desired positions [rad]
            velocities (float, np.array[N], None): desired velocity, or list of desired velocities [rad/s]
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            forces (float, np.array[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
        """
        if joint_ids is None:
            joint_ids = self.joints
        self.sim.set_joint_positions(self.id, joint_ids, positions, velocities=velocities, kps=kp, kds=kd,
                                     forces=forces)

    # TODO: max_velocities and forces
    def set_joint_velocities(self, velocities, joint_ids=None, forces=None, max_velocity=None):
        r"""
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, get all the actuated joints.
            velocities (float, np.array[N]): desired velocity, or list of desired velocities [rad/s]
            forces (float, np.array[N], None, bool): maximum motor torques / forces. If True, it will apply the
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
        r"""
        Set the acceleration of the given joint(s) (using force control). This is achieved by performing inverse
        dynamic which given the joint accelerations compute the joint torques to be applied.

        Args:
            accelerations (float, np.array[N]): desired joint acceleration, or list of desired joint accelerations
                [rad/s^2]
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

    def set_joint_torques(self, torques=None, joint_ids=None):
        r"""
        Set the torque to the given joint(s) (using force/torque control).

        Args:
            torques (float, np.array[N], None): desired torque(s) to apply to the joint(s) [N]. If None, it will apply
                a torque of 0 to the given joint(s).
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will set the joint torques to
                all (actuated) joints.
        """
        if isinstance(joint_ids, int):
            if torques is None:
                torques = 0
        else:
            if joint_ids is None:
                joint_ids = self.joints
            if not isinstance(joint_ids, collections.Iterable):
                raise TypeError("Expecting jointId to be a tuple, list, or numpy array, got instead "
                                "{}".format(type(joint_ids)))
            if torques is None:
                torques = [0] * len(joint_ids)
            elif isinstance(torques, (int, float)):
                torques = [torques] * len(joint_ids)

        self.sim.set_joint_motor_control(self.id, joint_ids, self.sim.TORQUE_CONTROL, forces=torques)

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
                positions (float, np.array[N]) (optional): target position of the joint (in position control) [rad]
                velocities (float, np.array[N]) (optional): target velocity of the joint (in position/velocity
                    control) [rad/s]
                forces (float, np.array[N]) (optional): in position/velocity control, this is the maximum force used
                    to reach the target value. In torque control, this is the force/torque to be applied.
                kp (float, np.array[N]) (optional): position gain :math:`Kp`
                kd (float, np.array[N]) (optional): velocity gain :math:`Kd`
                maxVelocity (float, np.array[N]) (optional): in position control, this limits the velocity to a maximum.
        """
        self.sim.set_joint_motor_control(self.id, joint_ids, control_mode, **kwargs)

    def disable_motor(self, joint_ids=None):
        r"""
        Disable the motor associated with the given joint(s).

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, it will disable the motors of
                all actuated joints.
        """
        if joint_ids is None:
            joint_ids = self.joints
        self.sim.set_joint_motor_control(self.id, joint_ids, self.sim.VELOCITY_CONTROL, forces=[0] * len(joint_ids))

    def reset_joint_states(self, q=None, dq=None, joint_ids=None):
        r"""
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
        r"""
        Return the joint positions for the home position defined by the user. This method has to be overwritten in
        the child class.
        """
        return np.zeros(self.num_actuated_joints)

    def set_home_joint_positions(self):
        r"""
        Set the joints to their home position defined by the user.
        """
        joint_positions = self.get_home_joint_positions()
        if joint_positions is not None:
            self.reset_joint_states(joint_positions)

    def move_home_joint_positions(self):
        r"""
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

    def get_joint_configurations(self, name=None):
        """
        If no name is specified, return the list of possible joint configurations. If a name is specified, it returns
        the corresponding joint ids and positions to move the robot to.

        This method has to be implemented in the child class.

        Args:
            name (str, None): name of the joint configuration to move the robot to.

        Returns:
            if name is None:
                list:
                    str: name of each joint configuration.
            else:
                np.array[M]: joint ids to move.
                np.array[M]: joint positions.
        """
        pass

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
                [0] np.array[3]: Cartesian position of center of mass
                [1] np.array[4]: Cartesian orientation of center of mass
                [2] np.array[3]: local position offset of inertial frame (CoM) expressed in the URDF link frame
                [3] np.array[4]: local orientation (quat. [x,y,z,w]) offset of the inertial frame expressed in URDF
                    link frame
                [4] np.array[3]: world position of the URDF link frame
                [5] np.array[4]: world orientation of the URDF link frame
                [6] np.array[3]: Cartesian world linear velocity
                [7] np.array[3]: Cartesian world angular velocity
            if multiple links: list of above
        """
        if isinstance(link_ids, int):  # one link
            return self.sim.get_link_state(self.id, link_ids, compute_velocity=compute_link_velocity,
                                           compute_forward_kinematics=compute_forward_kinematics)
        if link_ids is None:
            link_ids = self.joints
        return self.sim.get_link_states(self.id, link_ids, compute_velocity=compute_link_velocity,
                                        compute_forward_kinematics=compute_forward_kinematics)

    def get_link_local_position(self, link_ids=None):
        """
        Get the local position offset of the inertial frame (CoM) of the specified links expressed in the URDF link
        frame.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the local position of all
                links associated to actuated joints.

        Returns:
            if 1 link:
                np.array[3]: local position offset of inertial frame (CoM) expressed in the URDF link frame
            if multiple links: list of above
        """
        if isinstance(link_ids, int):
            return self.get_link_states(self, link_ids, False, False)[2]
        return [state[2] for state in self.get_link_states(self, link_ids, False, False)]

    def get_link_local_orientations(self, link_ids=None):
        """
        Get the local orientation offset of the inertial frame (CoM) of the specified links expressed in the URDF link
        frame.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the local position of all
                links associated to actuated joints.

        Returns:
            if 1 link:
                np.array[4]: local orientation (quaternion [x,y,z,w]) offset of inertial frame (CoM) expressed in the
                    URDF link frame
            if multiple links: list of above
        """
        if isinstance(link_ids, int):
            return self.get_link_states(self, link_ids, False, False)[3]
        return [state[3] for state in self.get_link_states(self, link_ids, False, False)]

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
        if link_ids is None:
            link_ids = self.joints
        return self.sim.get_link_names(self.id, link_ids)

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
                np.array[N]: mass of each link
        """
        if isinstance(link_ids, int):
            return self.sim.get_dynamics_info(self.id, link_ids)[0]
        if link_ids is None:
            link_ids = list(range(self.num_links))
        return np.array([self.sim.get_dynamics_info(self.id, link)[0] for link in link_ids])

    def get_link_frames(self, link_ids=None, flatten=False):
        r"""
        Return the link world frame position(s) and orientation(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the frame position of all
                links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.array[3]: the link frame position in the world space
                np.array[4]: Cartesian orientation of the link frame [x,y,z,w]
            if multiple links:
                np.array[N*3], np.array[N,3]: link frame position of each link in world space
                np.array[N*4], np.array[N,4]: orientation of each link frame [x,y,z,w]

        """
        return self.get_link_world_frame_positions(link_ids, flatten), self.get_link_world_frame_orientations(link_ids,
                                                                                                              flatten)

    def get_link_world_frame_positions(self, link_ids=None, flatten=False):
        r"""
        Return the frame position (in the Cartesian world space coordinates) of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the frame position of all
                links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.array[3]: the link frame position in the world space
            if multiple links:
                np.array[N*3], np.array[N,3]: link frame position of each link in world space
        """
        if isinstance(link_ids, int):
            return np.array(self.sim.get_link_state(self.id, link_ids)[4])
        if link_ids is None:
            link_ids = self.joints
        pos = np.array([self.sim.get_link_state(self.id, link)[4] for link in link_ids])
        if flatten:
            return pos.reshape(-1)  # 1D array
        return pos  # 2D array

    def get_link_world_frame_orientations(self, link_ids=None, flatten=False):
        r"""
        Return the frame orientation (in the Cartesian world space) of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the frame orientation of
                all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array of float numbers instead of an array of quaternion

        Returns:
            if 1 link:
                np.array[4]: Cartesian orientation of the link frame [x,y,z,w]
            if multiple links:
                np.array[N*4], np.array[N,4]: orientation of each link frame [x,y,z,w]
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
                np.array[3]: the link CoM position in the world space
            if multiple links:
                np.array[N*3], np.array[N,3]: CoM position of each link in world space
        """
        # check if cached
        if 'link_pos' in self._state:
            pos = self._state['link_pos'][0]  # (N,6)
        else:
            links = list(range(self.num_links))
            pos = self.sim.get_link_world_positions(body_id=self.id, link_ids=links)  # (N,6)
            self._state['link_pos'] = [pos, time.time()]

        # if one link
        if isinstance(link_ids, int):
            if link_ids == -1:
                return self.get_base_position()
            return pos[link_ids]

        # if multiple links
        if link_ids is None:
            link_ids = self.joints
        pos = pos[link_ids]

        # if we need to flatten
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
                np.array[3]: the link CoM position
            if multiple links:
                np.array[N*3], np.array[N,3]: CoM position of each link
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
                np.array[4]: Cartesian orientation of the link CoM [x,y,z,w]
            if multiple links:
                np.array[N*4], np.array[N,4]: CoM orientation of each link [x,y,z,w]
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
                np.array[4]: Cartesian orientation of the link CoM [x,y,z,w]
            if multiple links:
                np.array[N*4], np.array[N,4]: CoM orientation of each link [x,y,z,w]
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
                np.array[3]: linear velocity of the link in the Cartesian world space
            if multiple links:
                np.array[N*3], np.array[N,3]: linear velocity of each link
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
                np.array[3]: angular velocity of the link in the Cartesian world space
            if multiple links:
                np.array[N*3], np.array[N,3]: angular velocity of each link
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
                np.array[6]: linear and angular velocity of the link in the Cartesian world space
            if multiple links:
                np.array[N*6], np.array[N,6]: linear and angular velocity of each link
        """
        # check if cached
        if 'link_vel' in self._state:
            velocities = self._state['link_vel'][0]
        else:
            links = list(range(self.num_links))
            velocities = self.sim.get_link_world_velocities(body_id=self.id, link_ids=links)
            self._state['link_vel'] = [velocities, time.time()]

        # if one link, compute the linear and angular velocity of that link
        if isinstance(link_ids, int):
            if link_ids == -1:
                return self.get_base_velocity(concatenate=True)
            return velocities[link_ids]

        # if multiple links, compute the linear and angular velocity of each link
        if link_ids is None:
            link_ids = self.joints
        velocities = velocities[link_ids]

        # if we need to flatten the velocities (N, 6) --> (N*6,)
        if flatten:
            return velocities.reshape(-1)  # 1d array
        return velocities  # 2D array

    def get_spatial_link_world_velocities(self, link_ids=None, flatten=False):
        r"""
        Return the spatial link world velocities which is the concatenation of the angular and linear velocities.
        The difference with :func:`~get_link_world_velocities` is that this one returns the concatenation of the
        linear and angular velocities, instead of first the angular and then the linear velocities. So, it is just
        the order of concatenation that is different. See :func:`~get_link_world_velocities` for more information.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the angular and linear
                velocities of all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.array[6]: angular and linear velocity of the link in the Cartesian world space
            if multiple links:
                np.array[N*6], np.array[N,6]: angular and linear velocity of each link
        """
        velocities = self.get_link_world_velocities(link_ids=link_ids, flatten=False)

        if len(velocities.shape) == 1:
            velocities = np.concatenate((velocities[3:], velocities[:3]))
        else:
            velocities = np.concatenate((velocities[:, 3:], velocities[:, :3]))

        # if we need to flatten the velocities (N, 6) --> (N*6,)
        if flatten:
            return velocities.reshape(-1)  # 1d array

        return velocities  # 2D array

    # alias
    get_link_twist = get_spatial_link_world_velocities

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
                np.array[3]: the linear velocity of the given link wrt to the other link
            if multiple links:
                np.array[N*3], np.array[N,3]: linear velocity of each link wrt to the other link(s)
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
                np.array[3]: the angular velocity of the given link wrt to the other link
            if multiple links:
                np.array[N*3], np.array[N,3]: angular velocity of each link wrt to the other link(s)
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
                np.array[6]: the linear and angular velocity of the given link wrt to the other link
            if multiple links:
                np.array[N*6], np.array[N,6]: linear and angular velocity of each link wrt to the other link(s)
        """
        v1 = self.get_link_world_velocities(link_ids, flatten=False)
        v0 = self.get_base_velocity() if wrt_link_id is None or wrt_link_id == -1 \
                else self.get_link_world_velocities(wrt_link_id, flatten=False)
        v = (v1 - v0)
        if flatten:
            return v.reshape(-1)
        return v

    # def get_link_linear_accelerations(self, link_ids=None):
    #     pass
    #
    # def get_link_angular_accelerations(self, link_ids=None):
    #     pass
    #
    # def get_link_accelerations(self, link_ids=None):
    #     r"""
    #
    #     Args:
    #         link_ids:
    #
    #     Returns:
    #
    #     """
    #     pass

    def get_link_world_accelerations(self, link_ids=None, flatten=True):
        r"""
        Return the linear and angular accelerations (expressed in the Cartesian world space coordinates) for the given
        link(s).

        From [1], the acceleration of a link can be computed from the previous link acceleration in a recursive form:

        .. math:: \pmb{a}_i = \pmb{a}_{i-1} + \pmb{s}_i \ddot{q}_i + \pmb{v}_i \cross \pmb{s}_i \dot{q}_i,

        where :math:`\pmb{a}_i = [\dot{\pmb{\omega}}_i^\top, \dot{\pmb{v}}_{O,i}^\top]^\top \in \mathbb{R}^6` is the
        spatial acceleration of the link, :math:`i`, :math:`\pmb{s}_i \in \mathbb{R}^6` represents the joint motion
        axis fixed in link :math:`i`, :math:`\ddot{q}_i \in \mathbb{R}` is the joint acceleration associated with link
        :math:`i`, :math:`\pmb{v_i} = [\pmb{\omega}_i^\top, \pmb{v}_{O,i}^\top] \in \mathbb{R}^6` is the spatial
        velocity of the link :math:`i`, and :math:`\cross` is the spatial cross product.

        It can also be computed from the base to the link:

        .. math::

            \pmb{a}_i &= \sum_{j=1}^i \pmb{s}_j \ddot{q}_j + \pmb{v}_j \cross \pmb{s}_j \dot{q}_j \\
                &= \sum_{j=1}^i \pmb{s}_j \ddot{q}_j + \sum_{k=1}^{j-1} \pmb{s}_k \cross \pmb{s}_j \dot{q}_j \dot{q}_k.

        As well as using the Jacobian (and its derivative):

        .. math::

            \pmb{a}_i &= \frac{d}{dt} \pmb{v}_i \\
                      &= \frac{d}{dt} \pmb{J}_i(q) \dot{\pmb{q}}
                      &= \pmb{J}_i(q) \ddot{\pmb{q}} + \dot{\pmb{J}}_i(q) \dot{\pmb{q}},

        where :math:`\pmb{J}_i(q) \in \mathbb{R}^{6 \cross N}` is the Jacobian for body/link :math:`i` at the current
        configuration (i.e. joint positions) :math:`\pmb{q}` (the Jacobian is the concatenation of the angular and
        linear Jacobian), :math:`\dot{\pmb{q}}` and :math:`\ddot{\pmb{q}}` are the current joint velocity and
        acceleration vectors respectively, and :math:`\dot{\pmb{J}}_i(q)` is the time derivative of the Jacobian for
        body/link :math:`i`.

        Warnings: if the simulator provides accelerations, we return this one. If not, we compute the link world
            accelerations using finite difference.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the linear and angular
                accelerations of all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.array[6]: linear and angular acceleration of the link in the Cartesian world space
            if multiple links:
                np.array[N*6], np.array[N,6]: linear and angular acceleration of each link

        References:
            - [1] "Rigid Body Dynamics Algorithms" (chap 2.11), Featherstone, 2008
        """
        # check if cached
        if 'link_acc' in self._state:
            acc = self._state['link_acc'][0]
        else:
            # all link indices
            links = list(range(self.num_links))

            # if the simulator keep in memory the accelerations, return it
            if self.sim.supports_acceleration():
                acc = self.sim.get_link_world_accelerations(self.id, link_ids=links)  # (N,6)
            else:  # else, use finite difference

                # get current link world velocities and time
                if 'link_vel' not in self._state:
                    self.get_link_world_velocities(flatten=False)
                vel, t = self._state['link_vel']  # (N,6)

                # if we did not cache the previous base velocity
                if 'link_vel' not in self._prev_state:
                    acc = np.zeros(len(self.joints), 6)  # (N,6)
                else:
                    # retrieve previous link world velocities and time
                    vel_prev, t_prev = self._prev_state['link_vel']  # (N,6)

                    # compute time difference
                    if self.sim.use_real_time():  # if the simulator is in real-time mode
                        dt = (t - t_prev)
                    else:  # if we are stepping in the simulator
                        dt = self.sim.timestep

                    # get current link positions
                    pos = self.get_link_world_positions(link_ids=links, flatten=False)  # (N,3)

                    # separate linear and angular velocities
                    lin_vel, ang_vel = vel[:3], vel[3:]  # (N,3)
                    lin_vel_prev, ang_vel_prev = vel_prev[:3], vel_prev[3:]  # (N,3)

                    # compute base acceleration
                    ang_acc = (ang_vel - ang_vel_prev) / dt
                    lin_acc = (lin_vel - lin_vel_prev) / dt
                    lin_acc += np.cross(ang_acc, pos) + np.cross(ang_vel, np.cross(ang_vel, pos))
                    acc = np.hstack((lin_acc, ang_acc))  # (N,6)

            self._state['link_acc'] = [acc, time.time()]

        # if one link
        if isinstance(link_ids, int):
            if link_ids == -1:
                return self.get_base_acceleration(concatenate=True)
            return acc[link_ids]
        # if multiple links
        if link_ids is None:
            link_ids = self.joints
        acc = acc[link_ids]

        # if we need to concatenate the accelerations
        if flatten:
            return acc.reshape(-1)  # (N*6,)
        return acc

    def get_spatial_link_world_acceleration(self, link_ids=None, flatten=True):
        r"""
        Return the spatial link world accelerations which is the concatenation of the angular and linear accelerations.
        The difference with :func:`~get_link_world_accelerations` is that this one returns the concatenation of the
        linear and angular accelerations, instead of first the angular and then the linear accelerations. So, it is
        just the order of concatenation that is different. See :func:`~get_link_world_accelerations` for more
        information.

        From [1], the acceleration of a link can be computed from the previous link acceleration in a recursive form:

        .. math:: \pmb{a}_i = \pmb{a}_{i-1} + \pmb{s}_i \ddot{q}_i + \pmb{v}_i \cross \pmb{s}_i \dot{q}_i,

        where :math:`\pmb{a}_i = [\dot{\pmb{\omega}}_i^\top, \dot{\pmb{v}}_{O,i}^\top]^\top \in \mathbb{R}^6` is the
        spatial acceleration of the link, :math:`i`, :math:`\pmb{s}_i \in \mathbb{R}^6` represents the joint motion
        axis fixed in link :math:`i`, :math:`\ddot{q}_i \in \mathbb{R}` is the joint acceleration associated with link
        :math:`i`, :math:`\pmb{v_i} = [\pmb{\omega}_i^\top, \pmb{v}_{O,i}^\top] \in \mathbb{R}^6` is the spatial
        velocity of the link :math:`i`, and :math:`\cross` is the spatial cross product.

        It can also be computed from the base to the link:

        .. math::

            \pmb{a}_i &= \sum_{j=1}^i \pmb{s}_j \ddot{q}_j + \pmb{v}_j \cross \pmb{s}_j \dot{q}_j \\
                &= \sum_{j=1}^i \pmb{s}_j \ddot{q}_j + \sum_{k=1}^{j-1} \pmb{s}_k \cross \pmb{s}_j \dot{q}_j \dot{q}_k.

        As well as using the Jacobian (and its derivative):

        .. math::

            \pmb{a}_i &= \frac{d}{dt} \pmb{v}_i \\
                      &= \frac{d}{dt} \pmb{J}_i(q) \dot{\pmb{q}}
                      &= \pmb{J}_i(q) \ddot{\pmb{q}} + \dot{\pmb{J}}_i(q) \dot{\pmb{q}},

        where :math:`\pmb{J}_i(q) \in \mathbb{R}^{6 \cross N}` is the Jacobian for body/link :math:`i` at the current
        configuration (i.e. joint positions) :math:`\pmb{q}` (the Jacobian is the concatenation of the angular and
        linear Jacobian), :math:`\dot{\pmb{q}}` and :math:`\ddot{\pmb{q}}` are the current joint velocity and
        acceleration vectors respectively, and :math:`\dot{\pmb{J}}_i(q)` is the time derivative of the Jacobian for
        body/link :math:`i`.

        Warnings: if the simulator provides accelerations, we return this one. If not, we compute the link world
            accelerations using finite difference.

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the angular and linear
                accelerations of all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                np.array[6]: angular and linear acceleration of the link in the Cartesian world space
            if multiple links:
                np.array[N*6], np.array[N,6]: angular and linear acceleration of each link
        """
        accelerations = self.get_link_world_accelerations(link_ids=link_ids, flatten=False)

        if len(accelerations.shape) == 1:
            accelerations = np.concatenate((accelerations[3:], accelerations[:3]))
        else:
            accelerations = np.concatenate((accelerations[:, 3:], accelerations[:, :3]))

        # if we need to flatten the accelerations (N, 6) --> (N*6,)
        if flatten:
            return accelerations.reshape(-1)  # 1d array

        return accelerations  # 2D array

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
                    np.array[3]: contact position on A (in Cartesian world coordinates)
                    np.array[3]: contact position on B (in Cartesian world coordinates)
                    np.array[3]: contact normal on B pointing towards A
                    float: contact distance (positive for separation and negative for penetration)
                    float: normal force applied during the last simulation step
            if multiple links: list of above
        """
        if isinstance(link_ids, int):
            return self.sim.get_contact_points(body1=self.id, link1_id=link_ids)
        return [self.sim.get_contact_points(body1=self.id, link1_id=link) for link in link_ids]

    def get_link_local_inertia(self, link_ids=None):
        """
        Return the local inertia (diagonal) of the given link(s).

        Args:
            link_ids (int, int[N], None): link id, or list of desired link ids. If None, get the inertia of all the
                links (even of fixed links).

        Returns:
            if 1 link:
                np.array[3]: local inertia (diagonal vector) of the given link
            else:
                np.array[N]: mass of each link
        """
        if isinstance(link_ids, int):
            return self.sim.get_dynamics_info(body_id=self.id, link_id=link_ids)[2]
        if link_ids is None:
            link_ids = list(range(self.num_links))
        return np.array([self.sim.get_dynamics_info(self.id, link)[2] for link in link_ids])

    def set_link_positions(self, link_ids, positions, orientations=None):
        """
        Set the position(s) of the given link(s) using inverse kinematics (IK).

        Warnings: be careful that at the end we get joint position(s) using IK, and thus if you are trying to set
        the position of multiple links that share some joints, you will get positions that are inconsistents.

        Args:
            link_ids (int, int[N]): link id, or list of desired link ids.
            positions (np.array[3], [np.array[3]], np.array[N,3]): desired link position(s).
            orientations (np.array[4], list of np.array[4], np.array[N,4]): desired link orientation(s) (expressed as
                quaternions [x,y,z,w])
        """
        # TODO: think when setting the position of multiple links where some joints are shared (need to use the
        #  null-space)
        pass

    def set_link_velocities(self, link_ids, velocities):
        # TODO: think when setting the position of multiple links where some joints are shared (need to use the
        #  null-space)
        pass

    def set_link_forces(self, link_ids, forces):
        # TODO: think when setting the position of multiple links where some joints are shared (need to use the
        #  null-space)
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

            self.end_effectors = list(end_effectors.keys())
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
            position (np.array[3]): position vector
            orientation (np.array[4], np.array[3,3], np.array[3]): orientation (expressed as a quaternion [x,y,z,w],
                3x3 rotation matrix, or roll-pitch-yaw angles).

        Returns:
            np.array[4,4]: homogeneous matrix
        """
        return get_homogeneous_transform(position, orientation)

    ##############
    # Kinematics #
    ##############

    # TODO: allow to slice the Jacobian to only get what interests the user
    def get_jacobian(self, link_id, q=None, local_position=None):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q)^T, J_{ang}(q)^T]^T`, such that:

        .. math:: v = [\dot{p}^T, \omega^T]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            link_id (int): link id.
            q (np.array[N], None): joint positions of size N, where N is the number of DoFs. If None, it will compute q
                based on the current joint positions.
            local_position (None, np.array[3]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.array[6,N], np.array[6,(6+N)]: full geometric (linear and angular) Jacobian matrix. The number of
                columns depends if the base is fixed or floating.
        """
        if q is None:
            q = self.get_joint_positions()
        else:
            if len(q) != len(self.joints):
                raise ValueError("The length of q ({}) is different from the number of DoFs"
                                 " ({}).".format(len(q), len(self.joints)))

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
            q (np.array[N]): joint positions of size N, where N is the number of DoFs. If None, it will compute q based
                on the current joint positions.
            local_position: the point on the specified link to compute the Jacobian (in link local coordinates around
                its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.array[3,N], np.array[3,(6+N)]: full linear geometric Jacobian matrix. The number of columns depends if
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
            q (np.array[N]): joint positions of size N, where N is the number of DoFs. If None, it will compute q based
                on the current joint positions.
            local_position: the point on the specified link to compute the Jacobian (in link local coordinates around
                its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.array[3,N], np.array[3,(6+N)]: full angular geometric Jacobian matrix. The number of columns depends if
                the base is fixed or floating.
        """
        return self.get_jacobian(link_id, q, local_position)[3:]

    def get_spatial_jacobian(self, link_id, q=None, local_position=None):
        r"""
        Return the spatial Jacobian which is the vertical concatenation of the angular and linear Jacobian matrices.
        The difference with :func:`~get_jacobian` is that this one returns the concatenation of the linear and
        angular Jacobian matrices, instead of first the angular and then the linear Jacobian matrices. So, it is just
        the order of concatenation that is different. See :func:`~get_jacobian` for more information.

        Args:
            link_id (int): link id.
            q (np.array[N], None): joint positions of size N, where N is the number of DoFs. If None, it will compute q
                based on the current joint positions.
            local_position (None, np.array[3]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.array[6,N], np.array[6,(6+N)]: spatial Jacobian matrix. The number of columns depends if the base is
                fixed or floating.
        """
        jacobian = self.get_jacobian(link_id=link_id, q=q, local_position=local_position)
        return np.vstack((jacobian[3:], jacobian[:3]))

    @staticmethod
    def get_jacobian_derivative_rpy_to_angular_velocity(rpy_angle):
        r"""
        Return the Jacobian that maps RPY angle rates to angular velocities, i.e. :math:`\omega = T(\phi) \dot{\phi}`.

        Warnings: :math:`T` is singular when the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}`

        Args:
            rpy_angle (np.array[3]): RPY Euler angles [rad]

        Returns:
            np.array[3,3]: Jacobian matrix that maps RPY angle rates to angular velocities.
        """
        r, p, y = rpy_angle
        T = np.array([[1., 0., np.sin(p)],
                      [0., np.cos(r), -np.cos(p) * np.sin(r)],
                      [0., np.sin(r), np.cos(p) * np.cos(r)]])
        return T

    @staticmethod
    def get_jacobian_derivative_zyz_to_angular_velocity(zyz_angle):
        r"""
        Return the Jacobian that maps ZYZ angle rates to angular velocities, i.e. :math:`\omega = T(\phi) \dot{\phi}`.

        Warnings: :math:`T` is singular when the angle associated with `Y` is :math:`0` or :math:`\pi`.

        Args:
            zyz_angle (np.array[3]): ZYZ Euler angles [rad]

        Returns:
            np.array[3,3]: Jacobian matrix that maps ZYZ angle rates to angular velocities.
        """
        z, y = zyz_angle[:2]
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
            jacobian (np.array[6,N], np.array[6,6+N]): full geometric Jacobian.
            rpy_angle (np.array[3]): RPY Euler angles

        Returns:
            np.array[6,N], np.foat[6,(6+N)]: the full analytical Jacobian. The number of columns depends if the base
                is fixed or floating.
        """
        T = self.get_jacobian_derivative_rpy_to_angular_velocity(rpy_angle)
        Tinv = np.linalg.inv(T)
        Ja = np.vstack((np.hstack((np.identity(3), np.zeros((3, 3)))),
                        np.hstack((np.zeros((3, 3)), Tinv)))).dot(jacobian)
        return Ja

    @staticmethod
    def compute_jacobian_joint_derivative(jacobian):
        r"""
        Compute the derivative of the Jacobian wrt joint values (hybrid Jacobian representation; i.e. the base frame
        is the reference frame and the origin of the end-effector frame is located at the velocity reference point on
        the end-effector. Other representations include the body-fixed and inertial representations, see [1] (sec 3.1)
        for more information). The computation is based on [1].

        .. math:: \frac{d}{dq} J(q)

        Args:
            jacobian (np.array[6,N], np.array[6,6+N]): jacobian matrix J.

        Returns:
            np.array[6,N,N]: derivative of the Jacobian wrt joint values (dJ/dq)

        References:
            - [1] "Symbolic differentiation of the velocity mapping for a serial kinematic chain", Bruyninck et al.,
              Mechanism and Machine Theory. 1996
        """
        nb_rows = jacobian.shape[0]  # task space dim.
        nb_cols = jacobian.shape[1]  # joint space dim.

        # compute Jgrad
        J_grad = np.zeros((nb_rows, nb_cols, nb_cols))
        for i in range(nb_cols):
            for j in range(nb_cols):
                J_i, J_j = jacobian[:, i], jacobian[:, j]
                if j < i:
                    # J_grad[0:3, i, j] = np.cross(J_j[3:6], J_i[0:3])   # Slow implementation
                    J_grad[0, i, j] = J_j[4] * J_i[2] - J_j[5] * J_i[1]
                    J_grad[1, i, j] = J_j[5] * J_i[0] - J_j[3] * J_i[2]
                    J_grad[2, i, j] = J_j[3] * J_i[1] - J_j[4] * J_i[0]
                    # J_grad[3:6, i, j] = np.cross(J_j[3:6], J_i[3:6])  # Slow implementation
                    J_grad[3, i, j] = J_j[4] * J_i[5] - J_j[5] * J_i[4]
                    J_grad[4, i, j] = J_j[5] * J_i[3] - J_j[3] * J_i[5]
                    J_grad[5, i, j] = J_j[3] * J_i[4] - J_j[4] * J_i[3]
                elif j > i:
                    # J_grad[0:3, i, j] = -np.cross(J_j[0:3], J_i[3:6])  # Slow implementation
                    J_grad[0, i, j] = - J_j[1] * J_i[5] + J_j[2] * J_i[4]
                    J_grad[1, i, j] = - J_j[2] * J_i[3] + J_j[0] * J_i[5]
                    J_grad[2, i, j] = - J_j[0] * J_i[4] + J_j[1] * J_i[3]
                else:
                    # J_grad[0:3, i, j] = np.cross(J_i[3:6], J_i[0:3])  # Slow implementation
                    J_grad[0, i, j] = J_i[4] * J_i[2] - J_i[5] * J_i[1]
                    J_grad[1, i, j] = J_i[5] * J_i[0] - J_i[3] * J_i[2]
                    J_grad[2, i, j] = J_i[3] * J_i[1] - J_i[4] * J_i[0]

        return J_grad

    @staticmethod
    def compute_jacobian_time_derivative(prev_jacobian, curr_jacobian, dt):
        r"""
        Compute the Jacobian time derivative :math:`\dot{J}(q)` using finite difference:

        .. math:: \dot{J}(q) \sim \frac{ J(q(t)) - J(q(t-dt)) }{dt}

        Args:
            prev_jacobian (np.array[6,N], np.array[6,(6+N)]): previous Jacobian.
            curr_jacobian (np.array[6,N], np.array[6,(6+N)]): current Jacobian.
            dt (float): time difference (should be bigger than 0).

        Returns:
            np.array[6,N], np.array[6,(6+N)]: time derivative of the Jacobian matrix. The number of columns depends
                if the base is fixed or floating.
        """
        return (curr_jacobian - prev_jacobian) / dt

    def get_jacobian_time_derivative(self, link_id, local_position=None):
        r"""
        Get the Jacobian time derivative :math:`\dot{J}_i(q)` of the specified link :math:`i` using finite difference:

        .. math:: \dot{J}(q) \sim \frac{ J(q(t)) - J(q(t-dt)) }{dt}

        Warnings: Note that we keep in memory the previous jacobian matrix (associated to the given parameters), so
        calling this method the first time will just return a zero matrix.

        Args:
            link_id (int): link id.
            local_position (None, np.array[3]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.array[6,N], np.array[6,(6+N)]: time derivative of the Jacobian matrix (which is the concatenation of
                the linear and angular parts). The number of columns depends if the base is fixed or floating.
        """
        # convert type if necessary
        local_position = None if local_position is None else tuple(local_position)

        # compute key for jacobian dict
        key = (link_id, local_position)

        # check if cached
        if 'dJ' in self._jacobian and key in self._jacobian['dJ']:
            return self._jacobian['dJ'][key][0]

        # get current jacobian and time
        if key in self._jacobian:
            jacobian, t = self._jacobian[key]
        else:
            jacobian, t = self.get_jacobian(link_id=link_id, local_position=local_position)
            self._jacobian[key] = [jacobian, t]

        # check if we cached the previous jacobian
        if key not in self._prev_jacobian:
            dJ = np.zeros((6, self.num_dofs))
            self._jacobian.setdefault('dJ', {})[key] = dJ
            return dJ

        # retrieve previous jacobian and time
        jacobian_prev, t_prev = self._prev_jacobian[key]

        # compute time difference
        if self.sim.use_real_time():  # if the simulator is in real-time mode
            dt = (t - t_prev)
        else:  # if we are stepping in the simulator
            dt = self.sim.timestep

        # compute the Jacobian time derivative using finite difference, and cache it
        dJ = (jacobian - jacobian_prev) / dt
        self._jacobian.setdefault('dJ', {})[key] = dJ

        return dJ

    # alias
    get_Jdot = get_jacobian_time_derivative

    def get_spatial_jacobian_time_derivative(self, link_id, local_position=None):
        r"""
        Get the spatial Jacobian time derivative :math:`\dot{J}_i(q)` of the specified link :math:`i` using finite
        difference:

        .. math:: \dot{J}(q) \sim \frac{ J(q(t)) - J(q(t-dt)) }{dt}

        Warnings: Note that we keep in memory the previous jacobian matrix (associated to the given parameters), so
        calling this method the first time will just return a zero matrix.

        Compared to :func:`~get_jacobian_time_derivative`, the Jacobian matrix is here the concatenation of the angular
        part followed by the linear part, instead of the opposite.

        Args:
            link_id (int): link id.
            local_position (None, np.array[3]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            np.array[6,N], np.array[6,(6+N)]: time derivative of the spatial Jacobian matrix. The number of columns
                depends if the base is fixed or floating.
        """
        jacobian = self.get_jacobian_time_derivative(link_id=link_id, local_position=local_position)
        return np.vstack((jacobian[3:], jacobian[:3]))

    def get_center_of_mass_jacobian(self, q=None):
        r"""
        Compute the Jacobian for the center of mass of the robot. This method was coded based on the C++ code
        provided in the `ModelInterface` class from ADVR Humanoids repository. A useful reference to check for this
        method is [1].

        Args:
            q (np.array[N]): joint positions of size N, where N is the number of DoFs. If None, it will compute q
                based on the current joint positions.

        Returns:
            np.array[6,N]: CoM Jacobian

        References:
            - [1] "Whole-body cooperative balancing of humanoid robot using COG Jacobian", Sugihara et al., IROS, 2002
        """
        # check if cached
        if 'Jcom' in self._jacobian:
            return self._jacobian['Jcom']

        # Get current joint position
        if q is None:
            q = self.get_joint_positions()
        else:
            if len(q) != len(self.joints):
                raise ValueError("The length of q ({}) is different from the number of DoFs"
                                 " ({}).".format(len(q), len(self.joints)))

        # Robot total mass
        robot_mass = 0

        # initialize center of mass jacobian: J_com
        # if self.has_fixed_base():
        Jcom = np.zeros((6, self.num_dofs))
        # else:
        #     Jcom = np.zeros((6, self.num_dofs + 6))

        # calculate the CoM jacobian
        for link_id in range(self.num_links):
            Jlink = self.get_jacobian(link_id, q)
            link_mass = self.get_link_masses(link_id)
            Jcom += link_mass * Jlink
            robot_mass += link_mass

        Jcom /= robot_mass

        # cache it (for later)
        self._jacobian['Jcom'] = Jcom

        return Jcom

    # alias
    get_com_jacobian = get_center_of_mass_jacobian

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
            rpy_angle (np.array[3]): RPY Euler angles [rad]
            dRPY (np.array[3]): time derivative of RPY Euler angles [rad/s]

        Returns:
            np.array[3]: angular velocities [rad/s]
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
            rpy_angle (np.array[3]): RPY Euler angles [rad]
            angular_velocity (np.array[3]): angular velocities [rad/s]

        Returns:
            np.array[3]: time derivative of RPY Euler angles [rad/s]

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
            jacobian (np.array[D,N]): Jacobian matrix

        Returns:
            np.array[D,D]: :math:`JJ^T`
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
            jacobian (np.array[D,N]): Jacobian matrix
            damping_factor (float): damping factor

        Returns:
            np.array[N,D]: DLS inverse matrix
        """
        J, k = jacobian, damping_factor
        return J.T.dot(np.linalg.inv(J.dot(J.T) + k**2 * np.identity(J.shape[0])))

    @staticmethod
    def get_pinv_jacobian(jacobian):
        r"""
        Return the right pseudo-inverse of the jacobian, i.e. :math:`J^\dagger = J^T(JJ^T)^{-1}`.

        Args:
            jacobian (np.array[D,N]): Jacobian matrix

        Returns:
            np.array[N,N]: right pseudo-inverse of the Jacobian
        """
        return np.linalg.pinv(jacobian)

    def get_null_space_projector(self, jacobian):
        r"""
        The null space projector :math:`P` is the matrix that projects any vectors to the null space of :math:`J`.
        This is given by: :math:`P = (I - J^\dagger J)`, where :math:`J^\dagger = J^T(JJ^T)^{-1}` is the right
        pseudo-inverse of the jacobian :math:`J`. This is notably used to perform inverse kinematics, where
        :math:`\dot{q} = J^\dagger v + P \dot{q}_0` with :math:`\dot{q}_0` representing arbitrary joint velocities.

        Args:
            jacobian (np.array[D,N]): Jacobian matrix

        Returns:
            np.array[N,N]: null space projector matrix
        """
        J = jacobian
        I = np.identity(J.shape[1])
        return I - self.get_pinv_jacobian(J).dot(J)

    def compute_manipulability_measure(self, jacobian):
        r"""
        Compute the manipulability measure `w(q) = sqrt( det(J(q)J(q)^T) )`. This is useful to get a general sense
        about the manipulation ability of the manipulator. This term, for instance, vanishes at singular
        configurations (see [1]).

        Args:
            jacobian (np.array[D,N]): Jacobian matrix

        Returns:
            float: manipulability measure :math:`w(q)`

        References:
            - [1] "Robotics: Modelling, Planning and Control" (chap 3.5 and 3.9), Siciliano et al., 2010
        """
        return np.sqrt(np.linalg.det(self.get_JJT(jacobian)))

    def compute_velocity_manipulability_ellipsoid(self, jacobian):
        r"""
        Compute the velocity manipulability ellipsoid (matrix) as `M = J(q)J(q)^T`.

        Args:
            jacobian (np.array[D,N]): Jacobian matrix

        Returns:
            np.array[D,D]: velocity manipulability

        References:
            - [1] "Robotics: Modelling, Planning and Control" (section 3.9), Siciliano et al., 2010
        """
        return self.get_JJT(jacobian)

    def compute_force_manipulability_ellipsoid(self, jacobian):
        r"""
        Compute the force manipulability ellipsoid (matrix) as `M = (J(q)J(q)^T)^-1`.

        Args:
            jacobian (np.array[D,N]): Jacobian matrix

        Returns:
            np.array[D,D]: force manipulability

        References:
            - [1] "Robotics: Modelling, Planning and Control" (section 3.9), Siciliano et al., 2010
        """
        return np.linalg.inv(self.get_JJT(jacobian))

    @staticmethod
    def in_singular_configuration(jacobian):
        r"""
        Return True if we are in a singular configuration.

        Singularities are interesting because (see [1]):
        - they represent configurations where the mobility of the manipulator is reduced
        - infinite solutions to the IK problem may exist
        - around them, small velocities in the task/operational space may cause large velocities in the joint space

        Args:
            jacobian (np.array[D,N]): Jacobian matrix

        Returns:
            bool: True if in a singular configuration

        References:
            - [1] "Robotics: Modelling, Planning, and Control" (chap 3.3), Siciliano et al., 2010
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
            jacobian (np.array[3,N], np.array[6,N]): Jacobian matrix
            velocity (np.array[3], np.array[6]): linear and/or angular velocities

        Returns:
            np.array[N]: joint velocities
        """
        Jpinv = self.get_pinv_jacobian(jacobian)
        return Jpinv.dot(velocity)

    @staticmethod
    def get_cartesian_velocities_from_joint_velocities(jacobian, dq):
        r"""
        Return the Cartesian velocities :math:`v = [\dot{p}, \omega]^T` where :math:`\dot{p}` and :math:`\omega`
        are the linear and angular velocities, respectively.

        .. math:: v = J(q) \dot{q}

        Args:
            jacobian (np.array[D,N], np.array[D,N]): Jacobian matrix
            dq (np.array[N]): joint velocities

        Returns:
            np.array[6]: Cartesian linear and angular velocities
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
            position (np.array[3]): target position of the end effector (its link coordinate, not center of mass
                coordinate!). By default this is in Cartesian world space, unless you provide `q_curr` joint angles.
            orientation (np.array[4]): target orientation in Cartesian world space, quaternion [x,y,w,z]. If not
                specified, pure position IK will be used.
            lower_limits (np.array[N], list of N floats): lower joint limits. Optional null-space IK.
            upper_limits (np.array[N], list of N floats): upper joint limits. Optional null-space IK.
            joint_ranges (np.array[N], list of N floats): range of value of each joint.
            rest_poses (np.array[N], list of N floats): joint rest poses. Favor an IK solution closer to a given rest
                pose.
            joint_dampings (np.array[N], list of N floats): joint damping factors. Allow to tune the IK solution using
                joint damping factors.
            solver (int): p.IK_DLS (=0) or p.IK_SDLS (=1), Damped Least Squares or Selective Damped Least Squares, as
                described in the paper by Samuel Buss "Selectively Damped Least Squares for Inverse Kinematics".
            q_curr (np.array[N]): list of joint positions. By default PyBullet uses the joint positions of the body.
                If provided, the target_position and targetOrientation is in local space!
            max_iters (int): maximum number of iterations. Refine the IK solution until the distance between target
                and actual end effector position is below this threshold, or the `max_iters` is reached.
            threshold (float): residual threshold. Refine the IK solution until the distance between target and actual
                end effector position is below this threshold, or the `max_iters` is reached.

        Returns:
            np.array[M]: joint positions (for each actuated joint).
        """
        # calculate joint positions solving IK and return them
        return self.sim.calculate_inverse_kinematics(self.id, link_id, position=position, orientation=orientation,
                                                     lower_limits=lower_limits, upper_limits=upper_limits,
                                                     joint_ranges=joint_ranges, rest_poses=rest_poses,
                                                     joint_dampings=joint_dampings, max_iters=max_iters,
                                                     threshold=threshold)

    def calculate_inverse_differential_kinematics_velocity_manipulability(self, jacobian,
                                                                          target_velocity_manipulability, Km):
        r"""
        Compute the inverse differential kinematics for velocity manipulability; it will return a joint velocity for
        all the actuated joints [1].

        Args:
            jacobian (np.array[D,N]): jacobian matrix
            target_velocity_manipulability (np.array[D,D]): target velocity manipulability
            Km (float[,]): Proportional gain for manipulability error

        Returns:
            np.array[N]: joint velocities
            float: minimum of eigenvalues of the velocity manip. Jacobian
            float: Distance between desired and current manip. ellipsoids

        References:
            - [1] "Geometry-aware Tracking of Manipulability Ellipsoids", Jaquier et al., R:SS, 2018
        """
        num_task_vars = np.size(target_velocity_manipulability, 0)

        # Compute manipulability error
        velocity_manip = self.compute_velocity_manipulability_ellipsoid(jacobian)
        Me = logarithm_map([target_velocity_manipulability], velocity_manip[0:num_task_vars, 0:num_task_vars])[0]
        # print("Me: {}".format(Me))
        distance = distance_spd(target_velocity_manipulability, velocity_manip[0:num_task_vars, 0:num_task_vars])
        # print("SPD distance: {}".format(distance))

        Jm_red = self.compute_velocity_manipulability_jacobian(jacobian, num_task_vars)
        # print("Jm: {}".format(Jm_red))

        # Matrix singularity robustness
        U, S, Vh = np.linalg.svd(Jm_red)
        damping = 1E-2 if np.min(S) < 1E-2 else 1E-8

        dq = np.dot(self.get_damped_least_squares_inverse(Jm_red, damping), np.dot(Km, symmetric_matrix_to_vector(Me)))

        return dq, np.min(S), distance

    def compute_velocity_manipulability_jacobian(self, jacobian, num_task_vars):
        r"""
        Compute the velocity manipulability Jacobian [1].

        Args:
            jacobian (np.array[D,N]): jacobian matrix
            num_task_vars (int): number of task variables (usually 3 or 6)

        Returns:
            np.array[(num_task_vars * num_task_vars + num_task_vars) / 2, N]: manipulability jacobian matrix

        References:
            - [1] "Geometry-aware Tracking of Manipulability Ellipsoids", Jaquier et al., R:SS, 2018
        """
        num_dofs = jacobian.shape[1]

        # Jtot = sc.linalg.block_diag(*jacobian)
        # print("Jtot: {}".format(Jtot))

        # Compute derivative of Jacobian wrt joint angles
        J_grad = self.compute_jacobian_joint_derivative(jacobian)

        # for i in range(J_grad.shape[2]):
        #    print("dJ/dq_{}: {}".format(i, J_grad[:, :, i]))

        # Manipulability Jacobian
        Jm = tensor_matrix_product(J_grad, jacobian, 1) + \
             tensor_matrix_product(np.transpose(J_grad, [1, 0, 2]), jacobian, 0)
        # for i in range(Jm.shape[2]):
        #    print("Jm_{}: {}".format(i, Jm[:, :, i]))
        # Jm = Jm[num_task_vars, num_task_vars, :]

        # Manipulability Jacobian in matrix form (Mandel notation)
        # num_vars = len(num_task_vars)
        Jm_red = np.zeros((int((num_task_vars * num_task_vars + num_task_vars) / 2), np.sum(num_dofs)))
        # print("Jm_red.shape: {}".format(Jm_red.shape))

        for i in range(Jm.shape[2]):
            Jm_red[:, i] = symmetric_matrix_to_vector(Jm[0:num_task_vars, 0:num_task_vars, i])

        return Jm_red

    # def hard_priorities(self, jacobians, task_velocities, method='backtrack'):
    #     r"""
    #     Return dq.
    #
    #     Args:
    #         jacobians:
    #         task_velocities:
    #         method: 'successive', 'augmented', 'backtrack'.
    #
    #     Returns:
    #
    #     """
    #     pass

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
            q (np.array[M]): joint positions
            dq (np.array[M]): joint velocities
            des_ddq (np.array[M]): desired joint accelerations

        Returns:
            np.array[M]: joint torques computed using the rigid-body equation of motion

        References:
            - [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            - [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            - [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            - [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
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
            q (np.array[M]): joint positions
            dq (np.array[M]): joint velocities
            torques (np.array[M]): desired joint torques

        Returns:
            np.array[M]: joint accelerations computed using the rigid-body equation of motion

        References:
            - [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            - [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            - [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            - [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                  http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        # if the joint velocities and positions are not provided, read them
        if dq is None:
            dq = self.get_joint_velocities()
        if q is None:
            q = self.get_joint_positions()

        # compute and return joint accelerations
        torques = np.array(torques)
        if not self.fixed_base:  # if floating base
            torques = np.concatenate((np.zeros(6), torques))
        Hinv = np.linalg.inv(self.get_mass_matrix(q))
        C = self.calculate_inverse_dynamics(np.zeros(len(q)), dq=dq, q=q)
        if np.any(np.equal(C, None)):
            C = np.zeros(len(torques))
        acc = Hinv.dot(torques - C)
        if not self.fixed_base:  # if floating base
            return acc[6:]
        return acc

    def get_mass_matrix(self, q=None, q_idx=None):
        r"""
        Return the mass/inertia matrix :math:`H(q)`.

        Warnings: If the base is floating, it will return a [6+N,6+N] inertia matrix, where N is the number of actuated
            joints. If the base is fixed, it will return a [N,N] inertia matrix

        Args:
            q (np.array[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            q_idx (slice, None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N).

        Returns:
            np.array[N,N], np.array[6+N,6+N], np.array[M,M]: inertia matrix
        """
        if q is None:
            q = self.get_joint_positions()
        else:
            if len(q) != self.num_free_joints:  # self.num_dofs:
                raise ValueError("All the joint positions need to be given to this method. You can then slice the"
                                 "inertia matrix afterward.")

        # TODO: we need to get all the joints (even the fixed ones) --> need to test
        # make sure that we have all the joints even the fixed ones
        q_aug = np.zeros(self.num_joints)
        q_aug[self.joints] = q

        if q_idx is None:
            return np.array(self.sim.calculate_mass_matrix(self.id, q_aug))
        return np.array(self.sim.calculate_mass_matrix(self.id, q_aug))[q_idx, q_idx]

    # alias
    get_inertia_matrix = get_mass_matrix

    def compute_inertia_joint_derivative(self, jacobian):
        r"""
        Compute the derivative of the Inertia matrix H(q) wrt joint values q.
        The computation is based on [1].

        Args:
            jacobian (np.array[D,N]): Jacobian matrix

        Returns:
            float[N,N,N]: derivative of the inertia matrix wrt joint values (dH/dq)
        """
        nb_rows = jacobian.shape[0]  # task space dim.
        nb_cols = jacobian.shape[1]  # joint space dim.
        nb_links = self.num_links

        # initialize variables
        dHdq = np.zeros((nb_cols, nb_cols, nb_cols))
        Jlinks = np.zeros((nb_rows, nb_cols, nb_links))
        dJlink_dq = np.zeros((nb_rows, nb_cols, nb_cols, nb_links))  # 4D array to store derivatives of link Jacobians

        # Compute derivatives for link Jacobians
        qt = self.get_joint_positions()
        for linkId in range(nb_links):
            Jlinks[:, :, linkId] = self.get_jacobian(linkId, qt)[:, 0:nb_cols]  # Jacobian for robot link
            dJlink_dq[:, :, :, linkId] = self.compute_jacobian_joint_derivative(Jlinks[:, :, linkId])  # Derivative of J

        for n in range(nb_cols):
            for linkId in range(nb_links):
                mass_i = self.get_link_masses(linkId)
                # Create generalized inertia matrix for link
                Hi = np.diag(self.get_link_local_inertia(linkId))  # Local inertia
                Mi = np.vstack((np.hstack((mass_i*np.eye(3), mass_i*np.zeros((3, 3)))),
                                np.hstack((mass_i*np.zeros((3, 3)), Hi))))
                dHdq[:, :, n] += np.dot(np.dot(dJlink_dq[:, :, n, linkId].T, Mi), Jlinks[:, :, linkId]) + \
                                 np.dot(np.dot(Jlinks[:, :, linkId].T, Mi), dJlink_dq[:, :, n, linkId])

        return dHdq

    @staticmethod
    def get_cartesian_inertia_matrix(H=None, Ja=None):
        r"""
        Return the cartesian inertia matrix.

        .. math:: H_{x}(q) = J_{a}^{-T}(q) H(q) J_{a}^{-1}(q)

        where :math:`H(q)` is the joint inertia matrix, :math:`J_{a}` is the analytical Jacobian, i.e. it
        respects the relation :math:`\dot{x} = [\dot{p} \dot{\phi}]^T = J_{a}(q) \dot{q}` where :math:`\phi` are the
        Euler angles. This is different from the geometric Jacobian :math:`J` which respects
        :math:`v = [\dot{p} \omega]^T = J(q) \dot{q}`, where :math:`\omega` are the angular velocities.

        Args:
            H (np.array[N,N], None): Joint inertia matrix. If None, it will be computed here (the q's then need to be
                provided).
            Ja (np.array[6,N], None): Analytical Jacobian. If None, it will be computed here (the q's then need to be
                provided and the link_id

        Returns:
            np.array[6,6]: Cartesian inertia matrix
        """
        Ja_inv = np.linalg.inv(Ja)
        return Ja_inv.T.dot(H).dot(Ja_inv)

    def get_kinetic_energy(self, q=None, dq=None, q_idx=None):
        r"""
        Return the kinetic energy due to the movement of the specified joint(s).

        .. math:: T(q,\dot{q}) = \frac{1}{2} \dot{q}^T H(q) \dot{q}

        Args:
            q (np.array[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (np.array[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
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
        r"""
        Return the potential energy due to gravity.

        .. math:: V(q) = - \sum_{i=1}^N m_{l_i} g^T p_{l_i}

        where :math:`l_i` represents the link `i`, :math:`m_l` is the mass of the link, :math:`g` is the gravity
        vector, and :math:`p_l` is the position of the link.

        Args:
            q (np.array[N], None): joint positions of size N, where N is the total number of DoFs. THIS IS CURRENTLY
                NOT USED, as we can get the link positions from the simulator (instead of using forward kinematics).
            q_idx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.
            g (np.array[3]): gravity vector.

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
        r"""
        Return the potential energy of the system.

        WARNING: Note that we currently assume rigid body systems (thus rigid links). With this assumption, the
        potential energy is only due to gravitational forces. So, for now this is just an alias to
        `get_gravity_potential_energy`.

        Args:
            q (np.array[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (np.array[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).
            q_idx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: potential energy
        """
        return self.get_gravity_potential_energy(q, q_idx)

    def get_lagrangian(self, q=None, dq=None, q_idx=None):
        r"""
        Return the Lagrangian evaluate at the given configuration.

        .. math:: L(q, \dot{q}) = T(q, \dot{q}) - V(q)

        where :math:`T` and :math:`V` are the kinetic and potential energy respectively.

        Args:
            q (np.array[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (np.array[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).
            q_idx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: value of the Lagrangian
        """
        T = self.get_kinetic_energy(q=q, dq=dq, q_idx=q_idx)
        V = self.get_potential_energy(q=q, q_idx=q_idx)
        return T - V

    @staticmethod
    def get_joint_torques_from_cartesian_wrench(jacobian, wrench):
        r"""
        Return the joint torques from the given Cartesian wrench (=force and torque) using the provided Jacobian.

        .. math:: \tau = J^T(q) f

        where :math:`\tau` are the joint torques, :math:`f` is the wrench vector (i.e. it contains the forces/torques
        applied at the link), and :math:`J` is the geometric Jacobian.

        Args:
            jacobian (np.array[3,N], np.array[6,N]): jacobian matrix.
            wrench (np.array[3], np.array[6]): wrench applied to the link (point) associated to the given jacobian.

        Returns:
            np.array[N]: joint torques [Nm]
        """
        return jacobian.T.dot(wrench)

    @staticmethod
    def get_cartesian_wrench_from_joint_torques(jacobian, torques):
        r"""
        Return the Cartesian wrench (=force and torque) from the given joint torques using the provided Jacobian.

        .. math:: f = J(J^TJ)^{-1} \tau

        where :math:`\tau` are the joint torques, :math:`f` is the wrench vector (i.e. it contains the forces/torques
        applied at the link), and :math:`J` is the geometric Jacobian.

        Args:
            jacobian (np.array[6,N]): jacobian matrix.
            torques (np.array[N]): torques.

        Returns:
            np.array[6]: forces and torques (=wrench) in the Cartesian world space [N,Nm]
        """
        J = jacobian
        return J.dot(np.linalg.inv(J.T.dot(J))).dot(torques)

    def enable_coriolis_and_gravity_compensation(self, enable=True):
        """
        Enable the gravity and Coriolis compensation when applying torques. This will automatically compute these
        terms and add them automatically to the given torques when using torque control.

        Args:
            enable (bool): If True, enable the gravity and Coriolis compensation when applying torques.
        """
        self.coriolis_and_gravity_compensation = enable

    def get_coriolis_and_gravity_compensation_torques(self, q=None, dq=None, q_idx=None):
        r"""
        Return the torques that need to be applied to the robot joints such that it compensates for gravity and
        Coriolis effects.

        From the equations of motion:

        .. math:: H(q) \ddot{q} + C(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F,

        we can see that if we set :math:`F` and :math:`\ddot{q}` to 0, then we have:

        .. math:: \tau = C(q,\dot{q}) \dot{q} + g(q).

        These are the torques that need to be applied to the robot joints to compensate for gravity and Coriolis
        effects.

        Args:
            q (np.array[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (np.array[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            q_idx (int[M], None): slice the torques at the given q indices (0 < M <= N).

        Returns:
            np.array[M]: joint torques to be applied [Nm]
        """
        if q is None:
            q = self.get_joint_positions()
        if dq is None:
            dq = self.get_joint_velocities()

        ddq = np.zeros(len(self.joints))

        if q_idx is None:
            return self.sim.calculate_inverse_dynamics(self.id, q, dq, ddq)
        return self.sim.calculate_inverse_dynamics(self.id, q, dq, ddq)[q_idx]

    # alias
    get_nonlinear_effects = get_coriolis_and_gravity_compensation_torques

    def get_gravity_compensation_torques(self, q=None, q_idx=None):
        r"""
        Return the torques that need to be applied to the robot joints such that it compensates for gravity.

        From the equations of motion:

        .. math:: H(q) \ddot{q} + C(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F,

        we can see that if we set :math:`F, \dot{q}, \ddot{q}` to 0, then we have:

        .. math:: \tau = g(q).

        These are the torques that need to be applied to the robot joints to compensate for gravity.

        Args:
            q (np.array[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            q_idx (int[M], None): slice the torques at the given q indices (0 < M <= N).

        Returns:
            np.array[M]: joint torques to be applied [Nm]
        """
        if q is None:
            q = self.get_joint_positions()
        dq = np.zeros(len(q))
        return self.get_coriolis_and_gravity_compensation_torques(q, dq, q_idx)

    def get_coriolis_torques(self, q=None, dq=None, q_idx=None):
        r"""
        Return the torques that need to be applied to the robot joints such that it compensates for Coriolis effects
        in the absence of gravity, i.e. :math:`\tau = C(q,\dot{q}) \dot{q}`

        From the equations of motion:

        .. math:: H(q) \ddot{q} + C(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F,

        we can see that if we set :math:`F` and :math:`\ddot{q}` to 0, then we have:

        .. math:: \tau_1 = C(q,\dot{q}) \dot{q} + g(q),

        and if additionally, we set :math:`\dot{q}` to 0, then we have:

        .. math:: \tau_2 = g(q).

        We can then get :math:`C(q,\dot{q}) \dot{q} = \tau_1 - \tau_2`.

        Args:
            q (np.array[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (np.array[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            q_idx (int[M], None): slice the torques at the given q indices (0 < M <= N).

        Returns:
            np.array[M]: joint torques to be applied [Nm]
        """
        if q is None:
            q = self.get_joint_positions()
        if dq is None:
            dq = self.get_joint_velocities()

        tau1 = self.get_coriolis_and_gravity_compensation_torques(q=q, dq=dq, q_idx=q_idx)
        tau2 = self.get_gravity_compensation_torques(q=q, q_idx=q_idx)

        return tau1 - tau2

    def apply_coriolis_and_gravity_compensation(self, q=None, dq=None, q_idx=None, external_torques=0.):
        r"""
        Apply Coriolis and Gravity Compensation; set the torques using torque control.

        The torques are given by:

        .. math::  \tau = C(q,\dot{q}) \dot{q} + g(q).

        Args:
            q (np.array[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (np.array[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            q_idx (int[M], None): slice the torques at the given q indices (0 < M <= N).
            external_torques (np.array[M], float): external torques to be applied.
        """
        joint_ids = self.joints if q_idx is None else self.joints[q_idx]
        torques = self.get_coriolis_and_gravity_compensation_torques(q, dq, q_idx)
        self.set_joint_torques(torques=torques + external_torques, joint_ids=joint_ids)

    # TODO: finish to implement the method + think about multiple links + think about dimensions
    def get_active_compliant_torques(self, q=None, dq=None, q_idx=None, jacobian=None, link_velocity=None,
                                     link_id=None, kd=60):
        r"""
        Return the torques that need to be applied to enable active compliance. This is done by enabling Coriolis
        and gravity compensation along with a damping force projected from the Cartesian space to the joint space.

        The torques to be applied are given by:

        .. math::  \tau = C(q,\dot{q}) \dot{q} + g(q) + J^T F

        where :math:`F = - D v` with :math:`v` are the Cartesian velocities, and :math:`D` is the damping factor.

        Args:
            q (np.array[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (np.array[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            q_idx (int[M], None): slice the torques at the given q indices (0 < M <= N).
            jacobian (np.array[6,N], np.array[6,6+N]): Jacobian matrix.
            link_velocity (np.array[6]): linear and angular velocity of the link in the Cartesian world space

        Returns:
            np.array[M]: joint torques to be applied [Nm]
        """
        if q is None:
            q = self.get_joint_positions()
        if dq is None:
            dq = self.get_joint_velocities()
        if jacobian is None:
            jacobian = self.get_jacobian(link_id, q)
        if link_velocity is None:
            link_velocity = self.get_link_world_velocities(link_id)
        if isinstance(kd, int):
            kd = kd * np.identity(6)

        torques = self.get_coriolis_and_gravity_compensation_torques(q, dq, q_idx)
        torques += jacobian.T.dot(-kd * link_velocity)
        return torques

    # TODO: finish to implement the method
    def apply_active_compliance(self, q=None, dq=None, q_idx=None, external_torques=0.):
        r"""
        Apply active compliance; this is done by enabling Coriolis and gravity compensation along with a damping
        force projected from the Cartesian space to the joint space.

        Args:
            q (np.array[N], None): all the joint positions. If None, it will get the current joint positions of all the
                joints. However, note that if you already got the joint positions in your code,
                it is better to pass them to this method for performance.
            dq (np.array[N], None): all the joint velocities. If None, it will get the current joint velocities of
                all the joints.
            q_idx (int[M], None): slice the torques at the given q indices (0 < M <= N).
            external_torques (float, np.array[M]): external torques.
        """
        joint_id = self.joints if q_idx is None else self.joints[q_idx]
        torques = self.get_active_compliant_torques(q, dq, q_idx)
        self.set_joint_torques(torques + external_torques, joint_id)

    # def get_impedance_torques(self, jacobian, xdes=0, x=0, dx=0, dxdes=0, ddx=0, ddxdes=0, Km=1, Dm=0.01):
    #     r"""
    #     Return the impedance torques.
    #
    #     .. math:: F_{a} = H_m (\ddot{x} - \ddot{x}_d) + D_m (\dot{x} - \dot{x}_d) + K_m (x - x_d)
    #
    #     Args:
    #
    #
    #     Returns:
    #         np.array[N]: impedance torques
    #
    #     References:
    #         - [1] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
    #           http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
    #     """
    #     pass

    def get_attractor_torques(self, x_des, jacobian, link_id=None, q=None, dq=None, x=None, dx=None, K=5, D=0.1):
        r"""
        The torques to be applied (using impedance control with an attractor point) are given by:

        .. math::  \tau = C(q,\dot{q}) \dot{q} + g(q) + J^T F

        where :math:`F = K(x_d - x) - D v` with :math:`x` and :math:`v` are the Cartesian position and velocities,
        and :math:`K` and :math:`D` are the stiffness and damping factor, respectively.

        Args:
            q (np.array[N]): joint positions
            dq (np.array[N]): joint velocities
            x_des (np.array[3]): desired position of the link
            x (np.array[3]): cartesian world position of the link
            dx (np.array[3]): cartesian world linear velocity of the link
            jacobian (np.array[3,N]): linear jacobian associated to the link
            K (float, np.array[3,3]): proportional gain scalar or matrix
            D (float, np.array[3,3]): derivative gain scalar or matrix

        Returns:
            np.array[N]: torques to apply

        References:
            - [1] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
              http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        # check arguments
        if q is None:
            q = self.get_joint_positions()
        if dq is None:
            dq = self.get_joint_velocities()
        if link_id is not None:
            if x is None:
                x = self.get_link_world_positions(link_id)
            if dx is None:
                dx = self.get_link_world_linear_velocities(link_id)
            if jacobian is None:
                jacobian = self.get_jacobian(link_id, q=q)

        # coriolis and gravity torques
        torques = self.get_coriolis_and_gravity_compensation_torques(q=q, dq=dq)

        # impedance control: attractor point
        forces = K.dot(x_des - x) - D.dot(dx)
        torques += jacobian.T.dot(forces)

        return torques

    @staticmethod
    def compute_dynamic_manipulability_ellipsoid(jacobian, inertia):
        r"""
        Compute the dynamic manipulability ellipsoid (matrix) as `M = J(q)H(q)^{-1} (J(q)H(q)^{-1})^T`.

        Args:
            jacobian (np.array[D,N]): Jacobian matrix
            inertia (np.array[N,N]): inertia matrix in joint space

        Returns:
            np.array[D,D]: dynamic manipulability
        """
        epsilon = jacobian.dot(np.linalg.inv(inertia))
        return epsilon.dot(epsilon.T)

    def calculate_inverse_differential_kinematics_dynamic_manipulability(self, jacobian, inertia,
                                                                         target_dynamic_manipulability, Km):
        """
        Compute the inverse differential kinematics for dynamic manipulability; it will return a joint velocity for all
        the actuated joints.

        Args:
            jacobian (np.array[D,N]): Jacobian matrix
            inertia (np.array[N,N]): inertia matrix
            target_dynamic_manipulability (np.array[D,D]): target dynamic manipulability
            Km (float[,]): Proportional gain for manipulability error

        Returns:
            np.array[N]: joint velocities
            float: minimum of eigenvalues of the dynamic manip. Jacobian
            float: Distance between desired and current manip. ellipsoids
        """
        num_task_vars = np.size(target_dynamic_manipulability, 0)

        # Compute manipulability error
        dynamic_manip = self.compute_dynamic_manipulability_ellipsoid(jacobian, inertia)
        Me = logarithm_map([target_dynamic_manipulability[0:num_task_vars, 0:num_task_vars]],
                           dynamic_manip[0:num_task_vars, 0:num_task_vars])[0]
        # print("Me: {}".format(Me))
        distance = distance_spd(target_dynamic_manipulability[0:num_task_vars, 0:num_task_vars],
                                dynamic_manip[0:num_task_vars, 0:num_task_vars])
        print("SPD distance: {}".format(distance))

        Jm_red = self.compute_dynamic_manipulability_jacobian(jacobian, inertia, num_task_vars)
        # print("Jm: {}".format(Jm_red))

        # Matrix singularity robustness
        U, S, Vh = np.linalg.svd(Jm_red)
        damping = 1E-2 if np.min(S) < 1E-2 else 1E-8

        dq = np.dot(self.get_damped_least_squares_inverse(Jm_red, damping), np.dot(Km, symmetric_matrix_to_vector(Me)))

        return dq, np.min(S), distance

    def compute_dynamic_manipulability_jacobian(self, jacobian, inertia, num_task_vars):
        r"""
        Compute the dynamic manipulability Jacobian.

        Args:
            jacobian (np.array[D,N]): Jacobian matrix
            inertia (np.array[N,N]): inertia matrix
            num_task_vars (int): number of task variables (usually 3 or 6)

        Returns:
            np.array[(num_task_vars * num_task_vars + num_task_vars) / 2, N]: manipulability jacobian matrix
        """
        num_dofs = jacobian.shape[1]

        # Compute derivative of Jacobian wrt joint angles
        J_grad = self.compute_jacobian_joint_derivative(jacobian)
        # Compute derivative of Inertia matrix wrt joint angles
        H_grad = self.compute_inertia_joint_derivative(jacobian)
        #
        L = np.dot(jacobian, np.linalg.inv(inertia))

        # for i in range(J_grad.shape[2]):
        #    print("dJ/dq_{}: {}".format(i, J_grad[:, :, i]))
        # for i in range(H_grad.shape[2]):
        #    print("dH/dq_{}: {}".format(i, H_grad[:, :, i]))

        # Dynamic manipulability Jacobian
        Lgrad = tensor_matrix_product(J_grad, np.linalg.inv(inertia), 1) - \
                tensor_matrix_product(tensor_matrix_product(H_grad, L, 0), np.linalg.inv(inertia), 1)
        Jm = tensor_matrix_product(np.transpose(Lgrad[:, :], [1, 0, 2]), L, 0) + tensor_matrix_product(Lgrad, L, 1)
        # for i in range(Jm.shape[2]):
        #    print("Jm_{}: {}".format(i, Jm[:, :, i]))
        # Jm = Jm[num_task_vars, num_task_vars, :]

        # # Manipulability Jacobian in matrix form (Mandel notation)
        # num_vars = len(num_task_vars)
        Jm_red = np.zeros((int((num_task_vars * num_task_vars + num_task_vars) / 2), np.sum(num_dofs)))
        # print("Jm_red.shape: {}".format(Jm_red.shape))

        for i in range(Jm.shape[2]):
            Jm_red[:, i] = symmetric_matrix_to_vector(Jm[0:num_task_vars, 0:num_task_vars, i])

        return Jm_red

    def get_centroidal_dynamics(self, q=None, dq=None, inertia=None):
        r"""
        Compute the centroidal momentum dynamics based on [1]. "The centroidal momentum of a rigid-body system
        consists of its net linear momentum as well as its net angular momentum about its center of mass (CoM)" [1]

        The centroidal momentum, which is the sum of all body spatial momenta computed with respect to the CoM, is
        given by:

        .. math:: h_G = A_G \dot{q},

        where :math:`h_G \in \mathbb{R}^6` is the centroidal momentum, :math:`A_G \in \mathbb{R}^{6 \times (n+6)}` is
        the centroidal momentum matrix (CMM), and :math:`\dot{q}` are the system's generalized velocities. The CMM is
        related to the joint space inertia matrix (see code). This centroidal momentum collects the system linear and
        angular momentum together.

        The centroidal dynamics are then given by the equation:

        .. math:: \dot{h}_G = A_G \ddot{q} + \dot{A}_G \dot{q}.

        The centroidal dynamics :math:`\dot{h}_G` are then linked to external forces on the system by:

        .. math:: \dot{h}_G = f_G^{net},

        where :math:`f_G^{net}` is the net external wrench applied on the robot expressed at the CoM. This last term
        includes for instance the gravity force and ground reaction forces.

        Warnings: this currently does not work with a fixed base.

        Args:
            q (np.array[N], None): joint positions of size N, where N is the total number of free joints. If None, it
                will get the current joint positions (but note that this could lead to a decrease of performance).
            dq (np.array[N], None): joint velocities of size N, where N is the total number of free joints. If None, it
                will get the current joint velocities (but note that this could lead to a decrease of performance).
            inertia (np.array[6+N,6+N]): inertia matrix. If None, it will get the current inertia matrix
                (but note that this could lead to a decrease of performance if you have already computed it).

        Returns:
            np.array[6, N+6]: the centroidal momentum matrix :math:`A_G`
            np.array[6]: the centroidal dynamics velocity-dependent bias vector :math:`\dot{A}_G \dot{q}`

        Raises:
            RuntimeError: if the robot has not a floating base (i.e. it has a fixed base).

        References:
            - [1] "Improved computation of the humanoid centroidal dynamics and application for whole-body control",
              Wensing and Orin, 2016
            - [2] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis: sec 2.1.5 and 3.1.3), Xin,
              2018
        """
        # check if floating base
        if self.fixed_base:
            raise RuntimeError("You can not get the centroidal dynamics for a body with a fixed base; need to be a "
                               "floating base.")

        # get number of DoFs
        N, n = self.num_dofs, self.num_dofs - 6  # N=n+6

        # check arguments
        if q is None:
            q = self.get_joint_positions()  # shape = (N,)
        if dq is None:
            dq = self.get_joint_velocities()  # shape = (N,)
        if inertia is None:
            inertia = self.get_mass_matrix(q)  # shape = (N,N)
        H = inertia

        # Coriolis term: C(q, dq) * dq
        C_dq = self.get_coriolis_torques(q=q, dq=dq)  # shape = (N,)

        # U_1
        U_1 = np.hstack((np.identity(6), np.zeros((6, n))))  # shape = (6,n+6)

        # R_01(q_1)
        R_01 = get_matrix_from_quaternion(self.get_base_orientation()).T  # shape = (3,3)

        # Phi_1: this is a matrix transfer generalized velocity of floating base to spatial velocity defined in local
        # frame
        Phi_1 = np.vstack((np.hstack((np.zeros((3, 3)), np.identity(3))),
                           np.hstack((R_01.T, np.zeros((3, 3))))))  # shape = (6,6)

        # Psi_1
        Psi_1 = np.linalg.inv(Phi_1)  # shape = (6,6)

        # some other computations
        H11 = U_1.dot(H).dot(U_1.T)  # shape = (6,6)
        I1C = Psi_1.T.dot(H11).dot(Psi_1)  # shape = (6,6)
        M = I1C[6 - 1, 6 - 1]
        p1G = (1.0 / M) * np.array([I1C[3 - 1, 5 - 1], I1C[1 - 1, 6 - 1], I1C[2 - 1, 4 - 1]])  # shape = (3,)
        X_iG_T = np.vstack((np.hstack((R_01, R_01.dot(skew_matrix(p1G).T))),
                            np.hstack((np.zeros((3, 3)), R_01))))  # shape = (6,6)

        # compute centroidal momentum matrix and the dot product between the derivative of this centroidal momentum
        # matrix with the generalized velocities vector
        A_G = X_iG_T.dot(Psi_1.T).dot(U_1).dot(H)  # shape = (6,n+6)
        A_Gd_dq = X_iG_T.dot(Psi_1.T).dot(U_1).dot(C_dq)  # shape = (6,)

        return A_G, A_Gd_dq

    def get_centroidal_momentum(self, q=None, dq=None, inertia=None):
        r"""
        Return the centroidal momentum which consists of the net linear and angular momentum about the rigid-body's
        center of mass (CoM). This is thus the sum of all body spatial momenta computed with respect to the CoM,
        given by:

        .. math:: h_G = A_G \dot{q},

        where :math:`h_G \in \mathbb{R}^6` is the centroidal momentum, :math:`A_G \in \mathbb{R}^{6 \times (n+6)}` is
        the centroidal momentum matrix (CMM), and :math:`\dot{q}` are the system's generalized velocities. The CMM is
        related to the joint space inertia matrix (see code). This centroidal momentum collects the system linear and
        angular momentum together.

        Args:
            q (np.array[N], None): joint positions of size N, where N is the total number of free joints. If None, it
                will get the current joint positions (but note that this could lead to a decrease of performance).
            dq (np.array[N], None): joint velocities of size N, where N is the total number of free joints. If None, it
                will get the current joint velocities (but note that this could lead to a decrease of performance).
            inertia (np.array[6+N,6+N]): inertia matrix. If None, it will get the current inertia matrix
                (but note that this could lead to a decrease of performance if you have already computed it).

        Returns:
            np.array[6]: the centroidal momentum

        Raises:
            RuntimeError: if the robot has not a floating base (i.e. it has a fixed base).

        References:
            - [1] "Improved computation of the humanoid centroidal dynamics and application for whole-body control",
                  Wensing and Orin, 2016
            - [2] "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis: sec 2.1.5 and 3.1.3), Xin,
                  2018
        """
        if dq is None:
            dq = self.get_joint_velocities()  # shape = (N,)
        A_G, A_Gd_dq = self.get_centroidal_dynamics(q, dq, inertia)
        return A_G.dot(dq)

    @staticmethod
    def get_centroidal_momentum_singular_values(A_G):
        r"""
        Return the singular values of the centroidal momentum matrix.

        Args:
            A_G (np.array[6, N+6]): centroidal momentum matrix

        Returns:
            np.array[6]: singular values
        """
        u, s, vh = np.linalg.svd(A_G, full_matrices=True)
        return s

    @staticmethod
    def get_centroidal_momentum_orientation_and_scale(A_G):
        r"""
        Return the orientation and scale of the centroidal momentum matrix ellipsoid.

        Args:
            A_G (np.array[6, N+6]): centroidal momentum matrix

        Returns:
            np.array[4]: orientation (expressed as a quaternion [x,y,z,w])
            float: scale
        """
        u, scale, vh = np.linalg.svd(A_G, full_matrices=True)
        quaternion = get_quaternion_from_matrix(u)
        return quaternion, scale

    ######################
    # Symbolic Equations #
    ######################

    # def get_symbolic_equations_of_motion(self):
    #     r"""
    #     This returns the symbolic equation of motions of the robot (using the URDF). Internally, this used the
    #     `sympy.mechanics` module.
    #
    #     Returns:
    #
    #     References:
    #         - [1] `sympy.mechanics`: http://docs.sympy.org/latest/modules/physics/mechanics/index.html
    #         - [2] https://github.com/pydy/pydy-tutorial-human-standing
    #         - [3] https://github.com/pydy/pydy/tree/master/examples
    #     """
    #     pass
    #
    # def linearize_equations_of_motion(self, point=None):
    #     r"""
    #     Linearize the equation of motions around the given point. That is, instead of having :math:`\dot{x} = f(x,u)`
    #     where :math:`f` is in general a non-linear function, linearize it around a certain point.
    #
    #     .. math:: \dot{x} = A x + B u
    #
    #     where :math:`x` is the state vector, :math:`u` is the control input vector, and :math:`A` and :math:`B` are
    #     the matrices.
    #
    #     Args:
    #         point:
    #
    #     Returns:
    #         np.array[M,M]: :math:`A` matrix, where M is the size of the state vector
    #         np.array[M,N]: :math:`B` matrix, where N is the size of the input vector
    #
    #     References:
    #         - [1] "State-Space Representation of LTI Systems", Rowell, 2002 (handout):
    #               http://web.mit.edu/2.14/www/Handouts/StateSpace.pdf
    #         - [2] "Time-Domain Solution of LTI State Equations", Rowell, 2002 (handout):
    #               http://web.mit.edu/2.14/www/Handouts/StateSpaceResponse.pdf
    #         - [3] `sympy.mechanics`: http://docs.sympy.org/latest/modules/physics/mechanics/index.html
    #     """
    #     pass

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

    #########
    # Debug #
    #########

    @staticmethod
    def _get_joint_type_str(idx):
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

    # TODO: move these functions elsewhere

    # def plot_joint_positions(self, joint_ids=None):
    #     pass
    #
    # def plot_joint_velocities(self, joint_ids=None):
    #     pass
    #
    # def plot_joint_accelerations(self, joint_ids=None):
    #     pass
    #
    # def plot_com_position(self):
    #     pass
    #
    # def plot_com_velocity(self):
    #     pass
    #
    # def plot_com_acceleration(self):
    #     pass
    #
    # def plot_cartesian_positions(self, link_ids=None):
    #     pass
    #
    # def plot_cartesian_velocities(self, link_ids=None):
    #     pass
    #
    # def plot_cartesian_accelerations(self, link_ids=None):
    #     pass

    ########
    # draw # # WARNING: ALL THE FOLLOWING METHODS NEED A SIMULATOR IN WHICH TO RUN #
    ########

    # TODO: move these functions elsewhere

    def _draw_sphere(self, position, radius=0.1, color=(1, 1, 1, 1)):
        visual = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
        body = self.sim.create_body(mass=0, visual_shape_id=visual, position=position)
        return body

    def _draw_cylinder(self, position, orientation, radius=1., height=1., color=(1, 1, 1, 1)):
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

    def compute_and_draw_com_position(self, radius=0.05, color=(1, 0, 0, 0.8)):
        """
        Compute the CoM and draw it as a sphere in the simulator.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot.
            color (tuple/list of 4 float): rgba color of the sphere. By default, it is red.

        Returns:
            np.array[3]: center of mass
        """
        self.get_center_of_mass_position()
        self.draw_com_position(radius=radius, color=color)
        return self.com

    def draw_com_position(self, radius=0.05, color=(1, 0, 0, 0.8)):
        """
        Draw the CoM in the simulator.

        WARNING: `get_center_of_mass_position()` must be called before calling this method. Otherwise, check the other
        method `compute_and_draw_com_position()`.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot
            color (tuple/list of 4 float): rgba color of the sphere. By default it is red.
        """
        if self.com_visual is None:  # create visual shape if not already created
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
            np.array[3], None: position of the projected CoM, or None if it couldn't project the CoM
        """
        com = self.get_center_of_mass_position()
        object_id, _, _, hit_position, _ = self.sim.ray_test(com, com - np.array([0., 0., max_depth]))[0]
        if object_id >= 0:  # if there is a collision
            return hit_position  # = projected com
        else:
            return None

    def compute_and_draw_projected_com_position(self, radius=0.05, color=(0, 0, 1, 0.8)):
        """
        Compute and draw the projected center of mass.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot
            color (tuple/list of 4 float): rgba color of the sphere. By default it is blue.

        Returns:
            np.array[3], None: position of the projected CoM, or None if it couldn't project the CoM
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

    # def draw_projected_com(self, radius=0.05, color=(1,0,0,1)):
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
            # if link in self.visual_shapes:
            if link == -1:
                position, orientation = self.get_base_pose()
            else:
                # position = self.get_link_world_frame_positions(link)
                # orientation = self.get_link_world_frame_orientations(link)
                position = self.get_link_world_positions(link)
                orientation = self.get_link_world_orientations(link)

            # dim = self.visual_shapes[link]['dimensions']
            # radius = min(dim) * scaling * 0.2
            radius = 0.005 * scaling
            # self._draw_sphere(position, radius, color=(0,0,0,1))
            # length = 4*radius
            length = 0.05 * scaling
            self._draw_frame(position, orientation, radius, length)

    def draw_joint_frames(self, joint_ids=None, scaling=1.):
        """
        Draw the specified actuated joint frames.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, draw all the actuated joint frames.
            scaling (float): scaling factor
        """
        # check joints argument
        if joint_ids is None:
            joint_ids = self.joints
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        elif not isinstance(joint_ids, collections.Iterable):
            raise TypeError("Expecting the given 'joint_ids' to be None, an int, or a list of int, instead got: "
                            "{}".format(joint_ids))

        positions = self.get_link_world_frame_positions(joint_ids)
        orientations = self.get_link_world_frame_orientations(joint_ids)

        # draw each joint axis
        for joint, position, orientation in zip(joint_ids, positions, orientations):
            radius = 0.005 * scaling
            length = 0.05 * scaling
            self._draw_frame(position, orientation, radius, length)

    def draw_joint_axes(self, joint_ids=None, scaling=1.):
        """
        Draw the specified actuated joint axes.

        Args:
            joint_ids (int, int[N], None): joint id, or list of joint ids. If None, draw all the actuated joint axes.
            scaling (float): scaling factor
        """
        # check joints argument
        if joint_ids is None:
            joint_ids = self.joints
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        elif not isinstance(joint_ids, collections.Iterable):
            raise TypeError("Expecting the given 'joint_ids' to be None, an int, or a list of int, instead got: "
                            "{}".format(joint_ids))

        positions = self.get_link_world_frame_positions(joint_ids)
        orientations = self.get_link_world_frame_orientations(joint_ids)
        joint_axes = self.get_joint_axes(joint_ids)

        # draw each joint axis
        for joint, position, orientation, axis in zip(joint_ids, positions, orientations, joint_axes):
            radius = 0.008 * scaling
            length = 0.08 * scaling
            R = get_matrix_from_quaternion(orientation)
            y = R.dot(length / 2. * axis) + position
            qx = np.array([0.707, 0, 0, 0.707])  # 90deg around x

            # draw joint axis cylinder
            self._draw_cylinder(y, get_quaternion_product(orientation, qx), radius, length, color=(1, 1, 0, 1))

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
            position (np.array[3]): position in the world space
            orientation (np.array[4]): orientation in the world space
            scale (list/tuple of 3 float): scale in the (x,y,z) directions
            color (list/tuple of 4 float): RGBA color

        Returns:
            int: id of the ellipsoid
        """
        filename = os.path.dirname(__file__) + '/meshes/ellipsoid.obj'
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_MESH, filename=filename, mesh_scale=scale,
                                                    rgba_color=color)
        ellipsoid = self.sim.create_body(mass=0., visual_shape_id=visual_shape, position=position,
                                         orientation=orientation)
        return ellipsoid

    @staticmethod
    def get_ellipsoid_orientation_and_scale(X):
        r"""
        Get ellipsoid's orientation and scale.

        Args:
            X (np.array): 2D matrix

        Returns:
            np.array[4]: orientation (expressed as a quaternion [x,y,z,w])
            float: scale
        """
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
        # S, orientation = np.sqrt(evals), get_quaternion_from_matrix(evecs.T))
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

    def draw_ellipsoid_from_matrix(self, ellipsoid, position, color=(0, 1, 0, 0.7)):
        r"""
        Draw the manipulability ellipsoid at the specified link position (provided by the link id); the directions of
        the ellipsoid are given by the eigenvectors of the ellipsoid matrix :math:`evecs(E)` and the dimension scales
        are given by the singular values of :math:`\sigma(E)` where :math:`E` is the ellipsoid matrix.

        Args:
            ellipsoid (np.array): ellipsoid matrix (on which SVD will be performed to get the directions
            position (np.array): cartesian world position to draw the ellipsoid
            color (tuple of 4 float): RGBA color (each channel is between 0 and 1)

        Returns:
            int: id of the visual ellipsoid
        """
        orientation, scale = self.get_ellipsoid_orientation_and_scale(ellipsoid)
        return self.draw3d_ellipsoid(position, orientation, scale=scale, color=color)

    def draw_velocity_manipulability_ellipsoid(self, link_id, linear_jacobian=None, JJT=None, color=(0, 1, 0, 0.7)):
        r"""
        Draw the velocity manipulability ellipsoid using the linear jacobian; the directions of the ellipsoid are
        given by the eigenvectors :math:`evecs(JJ^T)` and the dimension scales are given by the singular values of
        :math:`JJ^T` where :math:`J` is the linear jacobian.

        Args:
            link_id (int): link id. This will be used to check where to draw the ellipsoid.
            linear_jacobian (np.array[3,N], None): linear Jacobian matrix. It doesn't need to be provided if `JJT` is
                given.
            JJT (np.array[3,3], None): if None, it will compute it using the provided linear Jacobian matrix.
            color (tuple of 4 float): RGBA color (each channel is between 0 and 1)

        Returns:
            int: id of the visual ellipsoid

        References:
            - [1] "Robotics: Modelling, Planning and Control" (section 3.9), Siciliano et al., 2010
        """
        if JJT is None:
            if linear_jacobian is None:
                raise ValueError("Please provide the linear Jacobian matrix")
            JJT = self.get_JJT(linear_jacobian)

        position = self.get_link_world_positions(link_id)
        return self.draw_ellipsoid_from_matrix(JJT, position=position, color=color)

    def draw_force_manipulability_ellipsoid(self, link_id, linear_jacobian=None, JJT=None, color=(0, 0, 1, 0.7)):
        r"""
        Draw the force manipulability ellipsoid using the linear jacobian; the directions of the ellipsoid are
        given by the eigenvectors :math:`evecs((JJ^T)^{-1})` and the dimension scales are given by the singular values
        of :math:`(JJ^T)^{-1}` where :math:`J` is the linear jacobian.

        Kineto-statics duality: direction with good velocity manipulability is obtained a direction along which poor
        force manipulability is obtained.

        Args:
            link_id (int): link id. This will be used to check where to draw the ellipsoid.
            linear_jacobian (np.array[3,N], None): linear Jacobian matrix. It doesn't need to be provided if `JJT` is
                given.
            JJT (np.array[3,3], None): if None, it will compute it using the provided linear Jacobian matrix.
            color (tuple of 4 float): RGBA color (each channel is between 0 and 1)

        Returns:
            int: id of the visual ellipsoid
        """
        if JJT is None:
            if linear_jacobian is None:
                raise ValueError("Please provide the linear Jacobian matrix")
            JJT = self.get_JJT(linear_jacobian)

        position = self.get_link_world_positions(link_id)
        return self.draw_ellipsoid_from_matrix(np.linalg.inv(JJT), position=position, color=color)

    def update_manipulability_ellipsoid(self, link_id, ellipsoid_id, ellipsoid, color=(0, 1, 0, 0.7)):
        """
        Update the position, orientation, and scaling of the given manipulability ellipsoid.

        Warnings: currently, the bullet simulator do not allow to update the scale, only the position and orientation.

        Args:
            link_id (int): link id. This will be used to check where to draw the ellipsoid.
            ellipsoid_id (int): id of the ellipsoid to update.
            ellipsoid (np.array): manipulability ellipsoid matrix
            color (tuple of 4 float): RGBA color (each channel is between 0 and 1)

        Returns:
            int: id of the new visual ellipsoid
        """
        # orientation, scale = self.get_ellipsoid_orientation_and_scale(ellipsoid)
        # position = self.get_link_world_positions(link_id)
        # self.sim.reset_base_pose(ellipsoid_id, position, orientation)
        self.remove_manipulability_ellipsoid(ellipsoid_id)
        position = self.get_link_world_positions(link_id)
        return self.draw_ellipsoid_from_matrix(ellipsoid, position=position, color=color)

    def remove_manipulability_ellipsoid(self, ellipsoid_id):
        """
        Remove the given ellipsoid manipulability ellipsoid.

        Args:
            ellipsoid_id (int): id of the ellipsoid to remove
        """
        self.sim.remove_body(ellipsoid_id)
