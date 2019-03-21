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
import quaternion
import collections
import os

from pyrobolearn.utils.converter import NumpyListConverter, QuaternionListConverter


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Robot(object):
    r"""Robot class.

    This is the class that all robots should inherit from. It contains all the useful methods to operate the robot,
    and has been implemented such that it is very generic.
    """

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.5), init_orient=(0, 0, 0, 1),
                 useFixedBase=False, scaling=1., *args, **kwargs):
        """
        Initialize the robot.

        Args:
            simulator: reference to the simulator such that the robot can access it.
            urdf_path (str): path to the urdf/mjcf file
            init_pos (float[3]): initial position
            init_orient (float[4]): initial orientation represented as a quaternion (x,y,z,w)
            useFixedBase (bool): if True, the base of the robot will be fixed
            scaling (float): scaling factor.
        """
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 1.5)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        self.init_position = init_pos
        self.init_orientation = init_orient

        # set the simulator
        self.sim = simulator
        # self.name = urdf_path.split('/')[-1].split('.urdf')[0]

        # load the robot
        # self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_RENDERING, 0)
        if urdf_path[-3:] == 'xml' or urdf_path[-4:] == 'mjcf':
            self.id = self.sim.loadMJCF(urdf_path)[0]  # assume the first entity is the robot
        else:  # if urdf_path[-4:] == 'urdf':
            self.id = self.sim.loadURDF(urdf_path, init_pos, init_orient, useFixedBase=useFixedBase,
                                        globalScaling=scaling)

        # self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_RENDERING, 1)

        if scaling != 1.0:
            # we rescale manually the mass and inertia matrices of each link
            for link in range(self.getNumberOfLinks()):
                info = self.sim.getDynamicsInfo(self.id, link)
                mass, localInertiaDiagonal = info[0], np.array(info[2])
                mass *= scaling**3      # because the density is unchanged when scaling
                localInertiaDiagonal *= scaling**5   # 5 = 3+2; 3 is for the mass, and 2 is for the distance: I~mr^2
                self.sim.changeDynamics(self.id, link, mass=mass, localInertiaDiagonal=localInertiaDiagonal)

        # set useful variables
        self.joints = []  # non-fixed joint/link indices in the simulator
        self.joint_names = {}  # joint name to id in the simulator
        self.link_names = {}  # link name to id in the simulator
        self.end_effectors = []  # end effector indices
        self.end_effector_names = {}  # end effector name to id in the simulator
        self.com = None  # center of mass
        self.actuators = []  # list of actuators
        self.sensors = []  # list of sensors

        for joint in range(self.getNumberOfJoints()):
            # Get joint info
            jnt = self.sim.getJointInfo(self.id, joint)
            self.joint_names[jnt[1]] = jnt[0]
            self.link_names[jnt[12]] = jnt[0]

            if jnt[2] != self.sim.JOINT_FIXED:  # if not a fixed joint
                self.joints.append(jnt[0])

        # set automatically the end-effectors
        self._setEndEffectors()

        # visual debug: sliders and drawing
        self.joint_sliders = {}
        self.comVisual = None
        self.projectedCoMVisual = None

        # Converters
        self.linear_converter = NumpyListConverter()
        self.angular_converter = QuaternionListConverter(convention=1)

        # other variables
        self.coriolisAndGravityCompensation = False
        self.floating_base = self._checkFloatingBase()

        # remember visual shapes
        # warning: the length of the returned list might be different from the number of links, because some links
        # don't have any visual shapes
        visualShapes = self.sim.getVisualShapeData(self.id)
        self.visualShapes = {shape[1]: {'dimensions': shape[3], 'color': list(shape[-1])} for shape in visualShapes}

        # symbolic equations
        self.symbols = None

        self._mass = None

        # init joint positions
        self.init_joint_positions = self.getJointPositions()
        self.joint_limits = self.getJointLimits()

        # Gains
        self.kp, self.kd = None, None

    def __repr__(self):
        """
        Return the name of the class.

        Returns:
            str: name of the class
        """
        return self.__class__.__name__

    def _convert_to_quat(self, quat, format='xyzw'):
        """
        Convert quaternion from tuple/list/array to np.quaternion (w,x,y,z).
        In pybullet, the quaternions are in the format (x,y,z,w).
        :param quat_xyzw: tuple/list/array representing the quaternion
        :param format: describe the format of the given quaternion
        :return: np.quaternion
        """
        if format == 'xyzw':
            return np.quaternion(quat[3], *quat[:3])
        elif format == 'wxyz':
            return np.quaternion(*quat)
        else:
            raise NotImplementedError('Unknown format for the given quaternion.')

    ##############
    # Properties #
    ##############

    @property
    def mass(self):
        if not self._mass: # compute and cache the mass
            self._mass = self.getTotalMass()
        return self._mass

    ################
    # General Info #
    ################

    def getNumberOfDoFs(self):
        """
        Return the number of degrees of freedom (i.e. the number of joints that are not fixed)

        Returns:
            int: the number of degrees of freedom
        """
        return len(self.joints)

    def getTotalMass(self):
        """
        Return the total mass of the robot (=sum of all mass links).

        Returns:
            float: total mass of the robot [kg]
        """
        return np.sum(self.getLinkMasses([-1] + list(range(self.getNumberOfLinks()))))

    ########
    # Base #
    ########

    def getBaseId(self):
        """
        Return the base id.

        Returns:
            int: base id.
        """
        return -1

    def getBaseName(self):
        """
        Return the base name.

        Returns:
            str: base name
        """
        return self.sim.getBodyInfo(self.id)[0]

    def getBasePositionAndOrientation(self, convert_to_numpy_quaternion=True):
        """
        Get base position and orientation with respect to the world frame.

        Returns:
            float[3]: position
            np.quaternion: orientation
        """
        pos, orientation = self.sim.getBasePositionAndOrientation(self.id)
        if convert_to_numpy_quaternion:
            orientation = self._convert_to_quat(orientation)
        else:
            orientation = np.array(orientation)
        return np.array(pos), orientation

    def getBasePosition(self):
        """
        Return the base position.

        Returns:
            float[3]: base position.
        """
        return np.array(self.sim.getBasePositionAndOrientation(self.id)[0])

    def getBaseOrientation(self, convert_to_numpy_quaternion=True):
        """
        Get the base orientation.

        Returns:
            quaternion (float[4]): base orientation in the form of a quaternion.
        """
        if convert_to_numpy_quaternion:
            return self._convert_to_quat(self.sim.getBasePositionAndOrientation(self.id)[1])
        return np.array(self.sim.getBasePositionAndOrientation(self.id)[1])

    def getBaseVelocity(self, concatenate=True):
        """
        Return the base linear and angular velocities.

        Returns:
            float[6]: linear and angular velocities of the base
        """
        lin_vel, ang_vel = self.sim.getBaseVelocity(self.id)
        if concatenate:
            return np.array(lin_vel + ang_vel)
        return np.array(lin_vel), np.array(ang_vel)

    def getBaseLinearVelocity(self):
        """
        Return the linear velocity of the base.

        Returns:
            float[3]: linear velocity of the base
        """
        return np.array(self.sim.getBaseVelocity(self.id)[0])

    def getBaseAngularVelocity(self):
        """
        Return the angular velocity of the base.

        Returns:
            float[3]: angular velocity of the base
        """
        return np.array(self.sim.getBaseVelocity(self.id)[1])

    def _checkFloatingBase(self):
        """
        Return True if the robot has a floating base (i.e. floating root link). Otherwise, it is a fixed base.

        Returns:
            bool: True if the robot has a floating base.
        """
        # # We used the fact if the robot has a floating base then the base velocity can be close to 0, but never
        # # completely equal to 0, unless the base is fixed
        # return np.all(np.zeros(6) == self.getBaseVelocity())

        # We check by computing the Jacobian (hopefully this only needs to be done once)
        if not self.joints:
            return False
        linkId = self.joints[0]
        J = self.calculateJacobian(linkId)
        # if floating base then the Jacobian will also include columns corresponding to the root link DoFs, while
        # with a fixed base, it will only have columns associated with the joints.
        if J.shape[1] > len(self.joints):
            return True
        return False

    def hasFloatingBase(self):
        """
        Return True if the robot has a floating base (i.e. floating root link). Otherwise, it is a fixed base.

        Returns:
            bool: True if the robot has a floating base.
        """
        return self.floating_base

    def hasFixedBase(self):
        """
        Return True if the robot has a fixed base.

        Returns:
            bool: True if the robot has a fixed base.
        """
        return not self.hasFloatingBase()

    #######
    # CoM #
    #######

    def getCoMPosition(self):
        """
        Return the center of mass position.

        Returns:
            float[3]: center of mass position
        """
        linkIds = list(range(self.getNumberOfLinks()))
        pos = self.getLinkWorldPositions(linkId=linkIds, flatten=False)
        mass = self.getLinkMasses(linkId=linkIds)

        self.com = np.sum(pos.T * mass, axis=1) / np.sum(mass)

        return self.com

    def getCoMVelocity(self):
        """
        Return the center of mass velocity.

        Returns:
            float[3]: center of mass velocity
        """
        linkIds = list(range(self.getNumberOfLinks()))
        vel = self.getLinkWorldVelocities(linkId=linkIds, flatten=False)
        mass = self.getLinkMasses(linkId=linkIds)

        com = np.sum(vel.T * mass, axis=1) / np.sum(mass)

        return com

    def getLinearMomentum(self):
        """
        Compute the linear momentum around the center of mass.

        .. math:: p = mv

        where :math:`p` is the linear momentum, :math:`m` is the total mass, and :math:`v` is the velocity.

        Returns:
            np.array[3]: linear momentum
        """
        return self.mass * self.getBaseLinearVelocity()

    def getAngularMomentum(self):
        """
        Compute the angular momentum around the center of mass.

        .. math:: h = I\omega

        where :math:`h` is the angular momentum (based on the world origin), :math:`I` is the moment of inertia,
        and :math:`\omega` is the angular velocity.

        Returns:
            np.array[3]: angular momentum
        """
        pass

    def getCentroidalDynamics(self, q=None, dq=None):
        """
        Compute the centroidal momentum dynamics based on [1]. "The centroidal momentum of a rigid-body system
        consists of its net linear momentum as well as its net angular momentum about its center of mass (CoM)" [1]

        #TODO: add math

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (float[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).

        Returns:
            np.array[6, N+6]: centroidal momentum matrix :math:`A_G`
            np.array[6]: the dot product between the derivative of the centroidal momentum matrix with the
                generalized velocities vector. That is, :math:`\dot{A}_G \dot{q}`

        References:
            [1] "Improved computation of the humanoid centroidal dynamics and application for whole-body control",
                Wensing and Orin, 2016
        """
        pass

    ########################
    # Joints (joint space) #
    ########################

    def getNumberOfJoints(self):
        """
        Return the total number of joints (including fixed-ones).

        Returns:
            int: the total number of joints (all types included)
        """
        return self.sim.getNumJoints(self.id)

    def getNumberOfActuatedJoints(self):
        """
        Return the total number of actuated joints.

        Returns:
            int: the total number of actuated joints
        """
        return len(self.joints)

    def getJointIds(self, joint=None):
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

        def getIndex(jnt):
            if isinstance(jnt, str):
                return self.joint_names[jnt]
            elif isinstance(jnt, int):
                return self.joints[jnt]
            else:
                raise TypeError("Incorrect type")

        # list of joints
        if isinstance(joint, collections.Iterable) and not isinstance(joint, str):
            return [getIndex(jnt) for jnt in joint]

        # one joint
        return getIndex(joint)

    def getJointInfo(self, jointId=None):
        """
        Get information about the given joint(s).

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
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
                [15] float[4]:  joint orientation in parent frame
                [16] int:       parent link index, -1 for base

            if multiple joints: list of joint information (i.e. list of above)
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)
        if jointId is None:
            jointId = self.joints
        return [self.sim.getJointInfo(self.id, jnt) for jnt in jointId]

    def getJointAxis(self, jointId=None):
        """
        Get information about the given joint(s).

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the axis for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float[3]: joint axis
            if multiple joint:
                [float[3]]: list of joint axis
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)[-4]
        if jointId is None:
            jointId = self.joints
        return [self.sim.getJointInfo(self.id, jnt)[-4] for jnt in jointId]

    # TODO: to check
    def getQIndex(self, jointId=None):
        """
        Get the corresponding q index of the given joint(s).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the q indices for all
                (actuated) joints.

        Returns:
            if 1 joint:
                int: q index
            if multiple joints:
                int[N]: q indices
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)[3] - 7 # TODO: check for 7
        if jointId is None:
            jointId = self.joints
        return np.array([self.sim.getJointInfo(self.id, jnt)[3] for jnt in jointId]) - 7 # TODO: check for 7

    def getDQIndex(self, jointId=None):
        """
        Get the corresponding dq index of the given joint(s).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the dq indices for all
                (actuated) joints.

        Returns:
            if 1 joint:
                int: dq index
            if multiple joints:
                int[N]: dq indices
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)[4]
        if jointId is None:
            jointId = self.joints
        return [self.sim.getJointInfo(self.id, jnt)[4] for jnt in jointId]

    def getJointTypes(self, jointId=None, convert_to_string=True):
        """
        Get the joint type as a string or integer.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.
            convert_to_string (bool): if True, it will return the joint type in a readable string format

        Returns:
            if 1 joint:
                str/int: the name of the joint type, or the flag associated with it.
            if multiple joints: list of above
        """
        if isinstance(jointId, int):
            if convert_to_string:
                return self.getJointTypeStr(self.sim.getJointInfo(self.id, jointId)[2])
            return self.sim.getJointInfo(self.id, jointId)[2]
        if jointId is None:
            jointId = self.joints
        if convert_to_string:
            return [self.getJointTypeStr(self.sim.getJointInfo(self.id, jnt)[2]) for jnt in jointId]
        return [self.sim.getJointInfo(self.id, jnt)[2] for jnt in jointId]

    def getJointTypeStr(self, idx):
        """
        Return the joint type as a string based on the flag.

        Args:
            idx (int): flag for the type of joint

        Returns:
            str: name of the joint type
        """
        return ['revolute', 'prismatic', 'spherical', 'planar', 'fixed', 'point2point', 'gear'][idx]

    def _isJointType(self, jointId=None, jointType=0):
        """
        Return True if the given joint(s) are revolute.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.
            jointType (int): flag for the type of joint. 0 = revolute, 1 = prismatic, 2 = spherical, 3 = planar,
                4 = fixed, 5 = point2point, 6 = gear.

        Returns:
            bool, bool[N]: list of booleans. True if the joint(s) are revolute.
        """
        if isinstance(jointId, int):
            return (self.sim.getJointInfo(self.id, jointId)[2] == jointType)
        if jointId is None:
            jointId = self.joints
        return [(self.sim.getJointInfo(self.id, joint)[2] == jointType) for joint in jointId]

    def isRevoluteJoint(self, jointId=None):
        """
        Return True if the given joint(s) are revolute.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            bool, bool[N]: list of booleans. True if the joint(s) are revolute.
        """
        return self._isJointType(jointId, jointType=self.sim.JOINT_REVOLUTE)

    def isPrismaticJoint(self, jointId=None):
        """
        Return True if the given joint(s) are prismatic.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            bool, bool[N]: list of booleans. True if the joint(s) are prismatic.
        """
        return self._isJointType(jointId, jointType=self.sim.JOINT_PRISMATIC)

    def isSphericalJoint(self, jointId=None):
        """
        Return True if the given joint(s) are spherical.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            bool, bool[N]: list of booleans. True if the joint(s) are spherical.
        """
        return self._isJointType(jointId, jointType=self.sim.JOINT_SPHERICAL)

    def isPlanarJoint(self, jointId=None):
        """
        Return True if the given joint(s) are planar.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            bool, bool[N]: list of booleans. True if the joint(s) are planar.
        """
        return self._isJointType(jointId, jointType=self.sim.JOINT_PLANAR)

    def isFixedJoint(self, jointId=None):
        """
        Return True if the given joint(s) are fixed.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            bool, bool[N]: list of booleans. True if the joint(s) are fixed.
        """
        return self._isJointType(jointId, jointType=self.sim.JOINT_FIXED)

    def isPoint2PointJoint(self, jointId=None):
        """
        Return True if the given joint(s) are point 2 point.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            bool, bool[N]: list of booleans. True if the joint(s) are point-2-point.
        """
        return self._isJointType(jointId, jointType=self.sim.JOINT_POINT2POINT)

    def isGearJoint(self, jointId=None):
        """
        Return True if the given joint(s) are gear.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            bool, bool[N]: list of booleans. True if the joint(s) are gear.
        """
        return self._isJointType(jointId, jointType=self.sim.JOINT_GEAR)

    def getJointLimits(self, jointId=None):
        """
        Get the joint limits of the given joint(s).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float[2]: lower and upper limit
            if multiple joints:
                float[N,2]: lower and upper limit for each specified joint
        """
        if isinstance(jointId, int):
            return np.array(self.sim.getJointInfo(self.id, jointId)[8:10])
        if jointId is None:
            jointId = self.joints
        return np.array([self.sim.getJointInfo(self.id, jnt)[8:10] for jnt in jointId])

    def getJointDampings(self, jointId=None):
        """
        Get the damping coefficient of the given joint(s).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: damping coefficient of the given joint
            if multiple joints:
                float[N]: damping coefficient for each specified joint
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)[6]
        if jointId is None:
            jointId = self.joints
        return np.array([self.sim.getJointInfo(self.id, jnt)[6] for jnt in jointId])

    def getJointFrictions(self, jointId=None):
        """
        Get the friction coefficient of the given joint(s).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: friction coefficient of the given joint
            if multiple joints:
                float[N]: friction coefficient for each specified joint
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)[7]
        if jointId is None:
            jointId = self.joints
        return np.array([self.sim.getJointInfo(self.id, jnt)[7] for jnt in jointId])

    def getJointMaxForces(self, jointId=None):
        """
        Get the maximum force that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: maximum force [N]
            if multiple joints:
                float[N]: maximum force for each specified joint [N]
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)[10]
        if jointId is None:
            jointId = self.joints
        return np.array([self.sim.getJointInfo(self.id, jnt)[10] for jnt in jointId])

    def getJointMaxVelocities(self, jointId=None):
        """
        Get the maximum velocity that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, return the information for all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: maximum velocity [rad/s]
            if multiple joints:
                float[N]: maximum velocities for each specified joint [rad/s]
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)[11]
        if jointId is None:
            jointId = self.joints
        return np.array([self.sim.getJointInfo(self.id, jnt)[11] for jnt in jointId])

    def getJointNames(self, jointId=None):
        """
        Return the name of the given joint(s).

        Args:
            jointId (int, int[N]): joint id, or list of joint ids. If None, get the name of all (actuated) joints.

        Returns:
            if 1 joint:
                str: name of the joint
            if multiple joints:
                str[N]: name of each joint
        """
        if isinstance(jointId, int):
            return self.sim.getJointInfo(self.id, jointId)[1]
        if jointId is None:
            jointId = self.joints
        return [self.sim.getJointInfo(self.id, joint)[1] for joint in jointId]

    def getJointStates(self, jointId=None):
        """
        Get the state of the given joint(s).

        Args:
            jointId (int, int[N], None): id of the joint, or list of joint ids. If None, get the state of all
                (actuated) joints.

        Returns:
            for 1 joint:
                float: joint position [rad]
                float: joint velocity [rad/s]
                float[6]: joint reaction forces [fx,fy,fz,mx,my,mz]
                float: applied joint motor torque (during the last step)
            for multiple joints: list of each joint state
        """
        if isinstance(jointId, int):
            return self.sim.getJointState(self.id, jointId)
        if jointId is None:
            jointId = self.joints
        return self.sim.getJointStates(self.id, jointId)

    def getJointPositions(self, jointId=None):
        """
        Get the position of the given joint(s).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, get the position of all (actuated)
                joints.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.float[N]: joint positions [rad]
        """
        if isinstance(jointId, int):
            return self.sim.getJointState(self.id, jointId)[0]
        if jointId is None:
            jointId = self.joints
        return np.array([state[0] for state in self.sim.getJointStates(self.id, jointId)])

    def getJointVelocities(self, jointId=None):
        """
        Get the velocity of the given joint(s).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, get the velocity of all (actuated)
                joints.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.float[N]: joint velocities [rad/s]
        """
        if isinstance(jointId, int):
            return self.sim.getJointState(self.id, jointId)[1]
        if jointId is None:
            jointId = self.joints
        return np.array([state[1] for state in self.sim.getJointStates(self.id, jointId)])

    def getJointAccelerations(self, jointId=None):
        """
        Get the acceleration at the given joint(s). This is carried out by first getting the joint torques, then
        performing forward dynamics to get the joint accelerations from the joint torques.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, get the acceleration of all
                (actuated) joints.

        Returns:
            if 1 joint:
                float: joint acceleration [rad/s^2]
            if multiple joints:
                np.float[N]: joint accelerations [rad/s^2]
        """
        # check joint id
        if jointId is None:
            jointId = self.joints

        # get the torques
        torques = self.getJointTorques(jointId)

        # compute the accelerations
        accelerations = self.calculateForwardDynamics(torques)

        # return the specified accelerations
        q_idx = self.getQIndex(jointId)
        return accelerations[q_idx]

    def getJointReactionForces(self, jointId=None):
        """
        Return the joint reaction forces at the given joint. Note that the torque sensor must be enabled, otherwise
        it will always return [0,0,0,0,0,0].

        Args:
            jointId (int, int[N], None): unique id of the joint, or list of joint ids. If None, get the joint reaction
                forces of all (actuated) joints.

        Returns:
            if 1 joint:
                np.float[6]: joint reaction force (fx,fy,fz,mx,my,mz) [N,Nm]
            if multiple joints:
                np.float[N,6]: joint reaction forces [N, Nm]
        """
        if isinstance(jointId, int):
            return np.array(self.sim.getJointState(self.id, jointId)[2])
        if jointId is None:
            jointId = self.joints
        return np.array([state[2] for state in self.sim.getJointStates(self.id, jointId)])

    def getJointTorques(self, jointId=None):
        """
        Get the applied torque on the given joint(s). "This is the motor torque applied during the last stepSimulation.
        Note that this only applies in VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the
        applied joint motor torque is exactly what you provide, so there is no need to report it separately." (from
        the 'pybullet user guide')

        Args:
            jointId (int, int[N], None): id of the joint, or list of joint ids. If None, get the joint torques of
                all (actuated) joints.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.float[N]: torques associated to the given joints [Nm]
        """
        if isinstance(jointId, int):
            return self.sim.getJointState(self.id, jointId)[3]
        if jointId is None:
            jointId = self.joints
        return np.array([state[3] for state in self.sim.getJointStates(self.id, jointId)])

    def getJointPowers(self, jointId, torque=None):
        """
        Return the applied power at the given joint(s). Power = torque * velocity.

        Args:
            jointId (int, int[N]): joint id, or list of joint ids
            torque (float, float[N]): torques to apply to the joint(s). This has to be provided if we are doing
                TORQUE_CONTROL.

        Returns:
            if 1 joint:
                float: joint power [W]
            if multiple joints:
                np.float[N]: power at each joint [W]
        """
        if isinstance(torque, (list, tuple)):
            torque = np.array(torque)
        elif torque is None:
            torque = self.getJointTorques(jointId)
        velocity = self.getJointVelocities(jointId)
        return torque * velocity

    # TODO: desVel, maxVel, and maxTorque
    def setJointPositions(self, position, jointId=None, kp=None, kd=None, velocity=None, maxTorque=None):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, get all the actuated joints.
            position (float, float[N]): desired position, or list of desired positions [rad]
            velocity (float, float[N], None): desired velocity, or list of desired velocities [rad/s]
            kp (float, float[N], None): position gain(s)
            kd (float, float[N], None): velocity gain(s)
            maxTorque (float, float[N], None): maximum motor torques
        """
        if isinstance(jointId, int):
            kwargs = {}
            if kp is not None:
                kwargs['positionGain'] = kp
            if kd is not None:
                kwargs['velocityGain'] = kd
            if velocity is not None:
                kwargs['targetVelocity'] = velocity
            if maxTorque is not None:
                kwargs['force'] = maxTorque
            self.sim.setJointMotorControl2(self.id, jointId, self.sim.POSITION_CONTROL, targetPosition=position,
                                           **kwargs)
        else:
            if jointId is None:
                jointId = self.joints
            kwargs = {}
            if kp is not None:
                if isinstance(kp, (float, int)):
                    kp = kp * np.ones(len(jointId))
                kwargs['positionGains'] = kp
            if kd is not None:
                if isinstance(kd, (float, int)):
                    kd = kd * np.ones(len(jointId))
                kwargs['velocityGains'] = kd
            # qIdx = self.getQIndex(jointId)
            # print("pos: ", position)
            # print(self.joint_limits[qIdx, 0], self.joint_limits[qIdx, 1])
            # TODO: the following clip causes an error... Check Minitaur...
            # position = np.clip(position, self.joint_limits[qIdx, 0], self.joint_limits[qIdx, 1])
            # kp = kp.tolist()
            # kd = kd.tolist()
            # print("pos: ", position)
            # print("kp: ", kp)
            # print("kd: ", kd)
            if velocity is not None:
                if isinstance(velocity, (float, int)):
                    velocity = velocity * np.ones(len(jointId))
                kwargs['targetVelocities'] = velocity
            if maxTorque is not None:
                if isinstance(maxTorque, (float, int)):
                    maxTorque = maxTorque * np.ones(len(jointId))
                kwargs['forces'] = maxTorque
            self.sim.setJointMotorControlArray(self.id, jointId, self.sim.POSITION_CONTROL, targetPositions=position,
                                               **kwargs)

    # TODO: maxVel and maxTorque
    def setJointVelocities(self, velocity, jointId=None, maxVelocity=True, maxTorque=True):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, get all the actuated joints.
            velocity (float, float[N]): desired velocity, or list of desired velocities [rad/s]
            maxVelocity (bool): if True, it will make sure that the given velocity(ies) are below their authorized
                maximum value(s) (inferred from the URDF, or set previously by the user). If you already did the check
                outside the method or if you don't want limits, set this variable to False.
            maxTorque (bool, float, float[N]): maximum motor torques
        """
        if isinstance(jointId, int):
            self.sim.setJointMotorControl2(self.id, jointId, self.sim.VELOCITY_CONTROL, targetVelocity=velocity)
        else:
            if jointId is None:
                jointId = self.joints
            self.sim.setJointMotorControlArray(self.id, jointId, self.sim.VELOCITY_CONTROL, targetVelocities=velocity)

    # TODO: maxAccel and maxTorque
    def setJointAccelerations(self, acceleration, jointId=None, maxAcceleration=True):
        """
        Set the acceleration of the given joint(s) (using force control). This is achieved by performing inverse
        dynamic which given the joint accelerations compute the joint torques to be applied.

        Args:
            acceleration (float, float[N]): desired joint acceleration, or list of desired joint accelerations [rad/s^2]
            jointId (int, int[N], None): joint id, or list of joint ids. If None, get all the actuated joints.
            maxAcceleration (bool): if True, it will make sure that the given acceleration(s) are below their
                authorized maximum value(s). If you already did the check outside the method or if you don't want
                limits, set this variable to False.
        """
        # check joint ids
        if jointId is None:
            jointId = self.joints
        elif isinstance(jointId, int):
            jointId = [jointId]
        if isinstance(acceleration, (int, float)):
            acceleration = [acceleration]
        if len(acceleration) != len(jointId):
            raise ValueError("Expecting the desired accelerations to be of the same size as the number of joints; "
                             "{} != {}".format(len(acceleration), len(jointId)))

        # if joint accelerations vector is not the same size as the actuated joints
        if len(acceleration) != len(self.joints):
            q_idx = self.getQIndex(jointId)
            acc = np.zeros(len(self.joints))
            acc[q_idx] = acceleration
            acceleration = acc

        # compute joint torques from Inverse Dynamics
        torques = self.calculateInverseDynamics(acceleration)

        # get corresponding torques
        if len(torques) != len(jointId):
            q_idx = self.getQIndex(jointId)
            torques = torques[q_idx]

        # print("Robot - torques {} for joints {}".format(torques, jointId))

        # set the joint torques
        self.setJointTorques(torques, jointId)

    def setJointTorques(self, torque=None, jointId=None):
        """
        Set the torque to the given joint(s) (using force/torque control).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, it will set the joint torques to
                all (actuated) joints.
            torque (float, float[N], None): desired torque(s) to apply to the joint(s) [N]. If None, it will apply
                a torque of 0 to the given joint(s).
        """
        if isinstance(jointId, int):
            if torque is None:
                torque = 0
            self.sim.setJointMotorControl2(self.id, jointId, self.sim.TORQUE_CONTROL, force=torque)
        else:
            if jointId is None:
                jointId = self.joints
            if not isinstance(jointId, collections.Iterable):
                raise TypeError("Expecting jointId to be a tuple, list, or numpy array, got instead "
                                "{}".format(type(jointId)))
            if torque is None:
                torque = [0]*len(jointId)
            elif isinstance(torque, (int, float)):
                torque = [torque]*len(jointId)
            self.sim.setJointMotorControlArray(self.id, jointId, self.sim.TORQUE_CONTROL, forces=torque)

    def setJointMotorControl(self, jointId, **kwargs):
        """
        Set joint motor control.

        In position control:
        .. math:: error = Kp (x_{des} - x) + Kd (\dot{x}_{des} - \dot{x})

        In velocity control:
        .. math:: error = \dot{x}_{des} - \dot{x}

        Note that the maximum forces and velocities are not automatically used for the different control schemes.

        Args:
            jointId (int, int[N]): joint id, or list of joint ids
            kwargs:
                controlMode (int): sim.VELOCITY_CONTROL (=0), sim.TORQUE_CONTROL (=1), sim.POSITION_CONTROL (=2)
                targetPosition (float, float[N]) (optional): target position of the joint (in position control) [rad]
                targetVelocity (float, float[N]) (optional): target velocity of the joint (in position/velocity
                    control) [rad/s]
                force (float, float[N]) (optional): in position/velocity control, this is the maximum force used
                    to reach the target value. In torque control, this is the force/torque to be applied.
                positionGain (float, float[N]) (optional): position gain :math:`Kp`
                velocityGain (float, float[N]) (optional): velocity gain :math:`Kd`
                maxVelocity (float, float[N]) (optional): in position control, this limits the velocity to a maximum.
        """
        if isinstance(jointId, int):
            self.sim.setJointMotorControl2(self.id, jointId, **kwargs)
        else:
            self.sim.setJointMotorControlArray(self.id, jointId, **kwargs)

    def disableMotor(self, jointId=None):
        """
        Disable the motor associated with the given joint(s).

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, it will disable the motors of
                all actuated joints.
        """
        if isinstance(jointId, int):
            self.sim.setJointMotorControl2(self.id, jointId, self.sim.VELOCITY_CONTROL, force=0)
        else:
            if jointId is None:
                jointId = self.joints
            self.sim.setJointMotorControlArray(self.id, jointId, self.sim.VELOCITY_CONTROL, forces=[0] * len(jointId))

    def resetJointStates(self, q=None, dq=None, jointIds=None):
        """
        Reset the state of the robot.

        Warnings: This is only valid in the simulator, and note that calling this method overrides all physics
        simulation.
        """
        # check jointIds
        if not jointIds:
            jointIds = self.joints
        if isinstance(jointIds, int):
            jointIds = [jointIds]

        # check q
        if q is None:
            q = np.zeros(len(jointIds))
        elif isinstance(q, (int, float)):
            q = [q]
        else:
            if len(q) != len(jointIds):
                raise ValueError("The number of joint ids does not match up with the number of q's")

        # check dq
        if dq is None:
            dq = np.zeros(len(jointIds))
        elif isinstance(dq, (int, float)):
            dq = [dq]
        else:
            if len(dq) != len(jointIds):
                raise ValueError("The number of joint ids does not match with the number of dq's")

        # reset the joint state
        for jointId, p, v in zip(jointIds, q, dq):
            self.sim.resetJointState(self.id, jointId, p, v)

    def getHomeJointPositions(self):
        """
        Return the joint positions for the home position defined by the user. This method has to be overwritten in
        the child class.
        """
        return np.zeros(self.getNumberOfActuatedJoints())

    def setJointHomePositions(self):
        """
        Set the joints to their home position defined by the user.
        """
        jointPositions = self.getHomeJointPositions()
        if jointPositions is not None:
            self.resetJointStates(jointPositions)

    def moveJointHomePositions(self):
        """
        Move the joints to their home position defined by the user. This method can be overwritten in the child
        class.

        The difference between this method and the `setJointHomePosition` is that the latter directly (re)set the
        joints to their home position while this one moves the joints to their home position.
        """
        jointPositions = self.getHomeJointPositions()
        if jointPositions is not None:
            self.setJointPositions(jointPositions)

    def setJointInitPositions(self):
        self.setJointPositions(self.init_joint_positions)

    ##################################
    # Links (task/operational space) #
    ##################################

    def getNumberOfLinks(self):
        """
        Return the total number of links even the ones associated to fixed-joints.

        Returns:
            int: the number of links
        """
        return self.getNumberOfJoints()

    def getNumberOfActuatedLinks(self):
        """
        Return the number of links associated to actuated joints.

        Returns:
            int: the number of links
        """
        return self.getNumberOfActuatedJoints()

    def getLinkIds(self, link=None):
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

        def getIndex(lnk):
            if isinstance(lnk, str):
                return self.link_names[lnk]
            elif isinstance(lnk, int):
                return self.joints[lnk]
            else:
                raise TypeError("Incorrect type")

        # list of links
        if isinstance(link, collections.Iterable) and not isinstance(link, str):
            return [getIndex(lnk) for lnk in link]

        # one link
        return getIndex(link)

    def getParentLinkIds(self, linkId=None):
        """
        Return the parent link of the given link(s)

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the state of all links
                associated to actuated joints.

        Returns:
            if 1 link:
                int: link id
            if multiple links:
                int[N]: link ids
        """
        if isinstance(linkId, int):
            return self.sim.getJointInfo(self.id, linkId)[-1]
        if linkId is None:
            linkId = self.joints
        return [self.sim.getJointInfo(self.id, link)[-1] for link in linkId]

    def getChainLinkIds(self, toLinkId, fromLinkId=None):
        """
        Return the link ids that constitute the chain(s) that go(es) from `fromLinkId` to `toLinkId`.

        Args:
            toLinkId (int, int[M]): link id(s) that end(s) the chain(s).
            fromLinkId (int, int[M], None): link id(s) that start(s) the chain(s). `fromLinkId` has to be a parent or
                ancestor of the `toLinkId`. If None, it will return the chain going from the base to the `toLinkId`.

        Returns:
            int[N], [int[N]]: chain(s) containing the link ids.
        """
        if fromLinkId is None:
            if isinstance(toLinkId, collections.Iterable):
                fromLinkId = [-1] * len(toLinkId)
            else:
                fromLinkId = -1

        def get_chain(toLink, fromLink):
            chain = [toLink]
            for linkId in chain:
                linkId = self.getParentLinkIds(linkId)
                chain.append(linkId)
                if linkId == fromLinkId:
                    break
            return chain[::-1]

        if isinstance(toLinkId, int):
            return get_chain(toLinkId, fromLinkId)
        else:
            return [get_chain(toLink, fromLink) for toLink, fromLink in zip(toLinkId, fromLinkId)]

    def getLinkStates(self, linkId=None, computeLinkVelocity=True, computeForwardKinematics=True):
        """
        Return the state of the given link(s).

        Warning: note that we do not convert the data here.

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the state of all links
                associated to actuated joints.
            computeLinkVelocity (bool): if True, the Cartesian world velocity will be computed and returned.
            computeForwardKinematics (bool): if True, the Cartesian world position/orientation will be recomputed
                using forward kinematics.

        Returns:
            if 1 link:
                [0] float[3]: Cartesian position of center of mass
                [1] float[4]: Cartesian orientation of center of mass
                [2] float[3]: local position offset of inertial frame (CoM) expressed in the URDF link frame
                [3] float[4]: local orientation (quat. [x,y,z,w]) offset of the inertial frame expressed in URDF link
                    frame
                [4] float[3]: world position of the URDF link frame
                [5] float[4]: world orientation of the URDF link frame
                [6] float[3]: Cartesian world linear velocity
                [7] float[3]: Cartesian world angular velocity
            if multiple links: list of above
        """
        if isinstance(linkId, int): # one link
            return self.sim.getLinkState(self.id, linkId, computeLinkVelocity=computeLinkVelocity,
                                         computeForwardKinematics=computeForwardKinematics)
        if linkId is None:
            linkId = self.joints
        return [self.sim.getLinkState(self.id, link, computeLinkVelocity=computeLinkVelocity,
                                      computeForwardKinematics=computeForwardKinematics) for link in linkId]

    def getLinkNames(self, linkId=None):
        """
        Return the name of the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the name of all links
                associated to actuated joints.

        Returns:
            if 1 link:
                str: link name
            if multiple links:
                str[N]: link names
        """
        if isinstance(linkId, int):
            return self.sim.getJointInfo(self.id, linkId)[12]
        if linkId is None:
            linkId = self.joints
        return [self.sim.getJointInfo(self.id, link)[12] for link in linkId]

    def getLinkMasses(self, linkId=None):
        """
        Return the mass of the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the mass of all the links
                (even of fixed links).

        Returns:
            if 1 link:
                float: mass of the given link
            else:
                float[N]: mass of each link
        """
        if isinstance(linkId, int):
            return self.sim.getDynamicsInfo(self.id, linkId)[0]
        if linkId is None:
            linkId = list(range(self.getNumberOfLinks()))
        return np.array([self.sim.getDynamicsInfo(self.id, link)[0] for link in linkId])

    def getLinkFrames(self, linkId=None, flatten=False):
        """
        Return the link frame position and orientation (expressed in the world space).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the frame position of all
                links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[3]: the link frame position in the world space
                quaternion: Cartesian orientation of the link frame (w,x,y,z)
            if multiple links:
                float[Nx3], float[N,3]: link frame position of each link in world space
                float[Nx4], quaternion[N]: orientation of each link frame (w,x,y,z)

        """
        return self.getLinkFrameWorldPositions(linkId, flatten), self.getLinkFrameWorldOrientations(linkId, flatten)

    def getLinkFrameWorldPositions(self, linkId=None, flatten=False):
        """
        Return the frame position (in the Cartesian world space coordinates) of the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the frame position of all
                links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[3]: the link frame position in the world space
            if multiple links:
                float[Nx3], float[N,3]: link frame position of each link in world space
        """
        if isinstance(linkId, int):
            return np.array(self.sim.getLinkState(self.id, linkId)[4])
        if linkId is None:
            linkId = self.joints
        pos = np.array([self.sim.getLinkState(self.id, link)[4] for link in linkId])
        if flatten:
            return pos.reshape(-1)  # 1d array
        return pos  # 2D array

    def getLinkFrameWorldOrientations(self, linkId=None, flatten=False):
        """
        Return the frame orientation (in the Cartesian world space) of the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the frame orientation of
                all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array of float numbers instead of an array of quaternion

        Returns:
            if 1 link:
                quaternion: Cartesian orientation of the link frame (w,x,y,z)
            if multiple links:
                float[Nx4], quaternion[N]: orientation of each link frame (w,x,y,z)
        """
        if isinstance(linkId, int):
            return self._convert_to_quat(self.sim.getLinkState(self.id, linkId)[5])
        if linkId is None:
            linkId = self.joints
        orientation = np.array([self.angular_converter.convertTo(self.sim.getLinkState(self.id, link)[5])
                                for link in linkId])
        if flatten:
            return quaternion.as_float_array(orientation).reshape(-1)
        return orientation  # array of quaternions

    def getLinkWorldPositions(self, linkId=None, flatten=True):
        """
        Return the CoM position (in the Cartesian world space coordinates) of the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the position of all links
                associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[3]: the link CoM position in the world space
            if multiple links:
                float[Nx3], float[N,3]: CoM position of each link in world space
        """
        if isinstance(linkId, int):
            if linkId == -1:
                return self.getBasePosition()
            return np.array(self.sim.getLinkState(self.id, linkId)[0])
        if linkId is None:
            linkId = self.joints
        pos = np.array([self.sim.getLinkState(self.id, link)[0] for link in linkId])
        if flatten:
            return pos.reshape(-1) # 1d array
        return pos # 2D array

    def getLinkPositions(self, linkId=None, wrtLinkId=None, flatten=True):
        """
        Return the link CoM position wrt the position of another link. By default, it is the base.

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the position of all links
                associated to actuated joints.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the position wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[3]: the link CoM position
            if multiple links:
                float[Nx3], float[N,3]: CoM position of each link
        """
        p1 = self.getLinkWorldPositions(linkId, flatten=False)
        p0 = self.getBasePosition() if wrtLinkId is None or wrtLinkId == -1 \
                else self.getLinkWorldPositions(wrtLinkId, flatten=False)
        p = (p1 - p0)
        if flatten:
            return p.reshape(-1)
        return p

    def getLinkWorldOrientations(self, linkId=None, flatten=True):
        """
        Return the CoM orientation (in the Cartesian world space) of the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the orientation of all links
                associated to actuated joints.
            flatten (bool): if True, it will return a 1D array of float numbers instead of an array of quaternion

        Returns:
            if 1 link:
                quaternion: Cartesian orientation of the link CoM (w,x,y,z)
            if multiple links:
                float[Nx4], quaternion[N]: CoM orientation of each link (w,x,y,z)
        """
        if isinstance(linkId, int):
            return self._convert_to_quat(self.sim.getLinkState(self.id, linkId)[1])
        if linkId is None:
            linkId = self.joints
        orientation = np.array([self.angular_converter.convertTo(self.sim.getLinkState(self.id, link)[1])
                                for link in linkId])
        if flatten:
            return quaternion.as_float_array(orientation).reshape(-1)
        return orientation # array of quaternions

    def getLinkOrientations(self, linkId=None, wrtLinkId=None, flatten=True):
        """
        Return the link CoM orientation wrt the orientation of another link. By default, it is the base.

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the orientation of all links
                associated to actuated joints.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the orientation wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                quaternion: Cartesian orientation of the link CoM (w,x,y,z)
            if multiple links:
                float[Nx4], quaternion[N]: CoM orientation of each link (w,x,y,z)
        """
        q1 = self.getLinkWorldOrientations(linkId)
        if wrtLinkId is None or wrtLinkId == -1:
            q0 = self.getBaseOrientation().inverse()
        else:
            if isinstance(wrtLinkId, int):
                q0 = self.getLinkWorldOrientations(wrtLinkId).inverse()
            else:
                q0 = np.array([self.getLinkWorldOrientations(link).inverse() for link in wrtLinkId])

        q = q0 * q1
        if flatten:
            quaternion.as_float_array(q).reshape(-1)
        return q

    def getLinkWorldLinearVelocities(self, linkId=None, flatten=True):
        """
        Return the linear velocity of the link(s) expressed in the Cartesian world space coordinates.

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the linear velocities of
                all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[3]: linear velocity of the link in the Cartesian world space
            if multiple links:
                float[Nx3], float[N,3]: linear velocity of each link
        """
        if isinstance(linkId, int):
            return np.array(self.sim.getLinkState(self.id, linkId, computeLinkVelocity=1)[6])
        if linkId is None:
            linkId = self.joints
        vel = np.array([self.sim.getLinkState(self.id, link, computeLinkVelocity=1)[6] for link in linkId])
        if flatten:
            return vel.reshape(-1)  # 1d array
        return vel  # 2D array

    def getLinkWorldAngularVelocities(self, linkId=None, flatten=True):
        """
        Return the angular velocity of the link(s) in the Cartesian world space coordinates.

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the angular velocities of
                all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[3]: angular velocity of the link in the Cartesian world space
            if multiple links:
                float[Nx3], float[N,3]: angular velocity of each link
        """
        if isinstance(linkId, int):
            return np.array(self.sim.getLinkState(self.id, linkId, computeLinkVelocity=1)[7])
        if linkId is None:
            linkId = self.joints
        vel = np.array([self.sim.getLinkState(self.id, link, computeLinkVelocity=1)[7] for link in linkId])
        if flatten:
            return vel.reshape(-1)  # 1d array
        return vel  # 2D array

    def getLinkWorldVelocities(self, linkId=None, flatten=True):
        """
        Return the linear and angular velocities (expressed in the Cartesian world space coordinates) for the given
        link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the linear and angular
                velocities of all links associated to actuated joints.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[6]: linear and angular velocity of the link in the Cartesian world space
            if multiple links:
                float[Nx6], float[N,6]: linear and angular velocity of each link
        """
        if isinstance(linkId, int):
            lin_vel, ang_vel = self.sim.getLinkState(self.id, linkId, computeLinkVelocity=1)[6:8]
            return np.array(lin_vel + ang_vel)
        if linkId is None:
            linkId = self.joints
        vel = []
        for link in linkId:
            lin_vel, ang_vel = self.sim.getLinkState(self.id, link, computeLinkVelocity=1)[6:8]
            vel.append(lin_vel + ang_vel)
        vel = np.array(vel)
        if flatten:
            return vel.reshape(-1)  # 1d array
        return vel  # 2D array

    def getLinkLinearVelocities(self, linkId=None, wrtLinkId=None, flatten=True):
        """
        Return the linear velocity of the given link(s) wrt the other specified link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the linear velocity of
                all links associated to actuated joints.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the linear velocity wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[3]: the linear velocity of the given link wrt to the other link
            if multiple links:
                float[Nx3], float[N,3]: linear velocity of each link wrt to the other link(s)
        """
        v1 = self.getLinkWorldLinearVelocities(linkId, flatten=False)
        v0 = self.getBaseLinearVelocity() if wrtLinkId is None or wrtLinkId == -1 \
                else self.getLinkWorldLinearVelocities(wrtLinkId, flatten=False)
        v = (v1 - v0)
        if flatten:
            return v.reshape(-1)
        return v

    def getLinkAngularVelocities(self, linkId=None, wrtLinkId=None, flatten=True):
        """
        Return the angular velocity of the given link(s) wrt to the other specified link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the angular velocity of
                all links associated to actuated joints.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the angular velocity wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[3]: the angular velocity of the given link wrt to the other link
            if multiple links:
                float[Nx3], float[N,3]: angular velocity of each link wrt to the other link(s)
        """
        w1 = self.getLinkWorldAngularVelocities(linkId, flatten=False)
        w0 = self.getBaseAngularVelocity() if wrtLinkId is None or wrtLinkId == -1 \
                else self.getLinkWorldAngularVelocities(wrtLinkId, flatten=False)
        w = (w1 - w0)
        if flatten:
            return w.reshape(-1)
        return w

    def getLinkVelocities(self, linkId=None, wrtLinkId=None, flatten=True):
        """
        Return the linear and angular velocity of the given link(s) wrt to the other specified link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get the angular velocity of
                all links associated to actuated joints.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the angular velocity wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 link:
                float[6]: the linear and angular velocity of the given link wrt to the other link
            if multiple links:
                float[Nx6], float[N,6]: linear and angular velocity of each link wrt to the other link(s)
        """
        v1 = self.getLinkWorldVelocities(linkId, flatten=False)
        v0 = self.getBaseVelocity() if wrtLinkId is None or wrtLinkId == -1 \
                else self.getLinkWorldVelocities(wrtLinkId, flatten=False)
        v = (v1 - v0)
        if flatten:
            return v.reshape(-1)
        return v

    def getLinkLinearAccelerations(self, linkId=None):
        raise NotImplementedError

    def getLinkAngularAccelerations(self, linkId=None):
        raise NotImplementedError

    def getLinkAccelerations(self, linkId=None):
        raise NotImplementedError

    def getLinkContacts(self, linkId):
        """
        Check if the given link(s) is/are in contact with something in the environment, and return all the contact
        points involving the given robot link(s).

        Warnings: note that in reality, you can't know if your link(s) is/are in contact with an object unless there
        is a sensor attached to it. However, this can be useful in simulation to optimize, for instance, trajectories.

        Args:
            linkId (int, int[N]): link id, or list of desired link ids.

        Returns:
            if 1 link:
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
            if multiple links: list of above
        """
        if isinstance(linkId, int):
            return self.sim.getContactPoints(bodyA=self.id, linkIndexA=linkId)
        return [self.sim.getContactPoints(bodyA=self.id, linkIndexA=link) for link in linkId]

    def setLinkPositions(self, linkId, position, orientation=None):
        """
        Set the position(s) of the given link(s) using inverse kinematics (IK).

        Warnings: be careful that at the end we get joint position(s) using IK, and thus if you are trying to set
        the position of multiple links that share some joints, you will get positions that are inconsistents.

        Args:
            linkId (int, int[N]): link id, or list of desired link ids.
            position (float[3], [float[3]], float[N,3]):
            orientation (float[4], [float[4]], float[N,4]):
        """
        pass

    #################
    # End-Effectors #  # same interface than Links (but easier to manipulate) #
    #################

    def _setEndEffectors(self):
        """
        Set automatically the end-effector ids and names based on the URDF. Here, all the leaves of the robot
        kinematic tree will be considered as end-effectors. Thus, use it with caution.
        """
        if len(self.end_effectors) == 0:
            end_effectors = {}

            # go through all the joints/links
            for joint in range(self.getNumberOfJoints()):
                # get useful information from current joint/link
                info = self.sim.getJointInfo(self.id, joint)
                parentIdx, linkName = info[-1], info[-5]

                # add this link in the end-effectors dict
                end_effectors[joint] = linkName

                # remove parent index from the end-effectors dict if present
                end_effectors.pop(parentIdx, None)

            self.end_effectors = end_effectors.keys()
            self.end_effector_names = {name: idx for idx, name in end_effectors.items()}

    def getNumberOfEndEffectors(self):
        """
        Return the number of end-effectors.

        Returns:
            int: number of end-effectors
        """
        return len(self.end_effectors)

    def getEndEffectorIds(self, endEffector=None):
        """
        Get the end effector ids from the name(s) or index(ices).

        Note that the end-effector id is unique and goes from 0 to the total number of end-effectors.

        Args:
            endEffector (str, int, list of str/int, None): if str, it will get the end-effector id associated to the
                given name. If int, it will get the end-effector id associated to the given q index. If it is a list
                of str and/or int, it will get the corresponding end-effector ids. If None, it will return all the
                end-effector ids.

        Returns:
            if 1 end-effector:
                int: end-effector id
            if multiple end-effectors:
                int[N]: end-effector ids
        """
        if endEffector is None:
            return self.end_effectors

        def getIndex(link):
            if isinstance(link, str):
                return self.end_effector_names[link]
            elif isinstance(link, int):
                return self.end_effectors[link]
            else:
                raise TypeError("Incorrect type")

        # list of links
        if isinstance(endEffector, collections.Iterable) and not isinstance(endEffector, str):
            return [getIndex(link) for link in endEffector]

        # one link
        return getIndex(endEffector)

    def getEndEffectorNames(self, endEffectorId=None):
        """
        Return the name of the given end-effector(s).

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the name of all end-effectors.

        Returns:
            if 1 end-effector:
                str: end-effector name
            if multiple end-effectors:
                str[N]: name of each end-effector
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkNames(endEffectorId)

    def getEndEffectorWorldPositions(self, endEffectorId=None, flatten=True):
        """
        Return the world position of the given end-effector(s).

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the position of all end-effectors.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 end-effector:
                float[3]: world position of the end-effector
            if multiple end-effectors:
                float[Nx3], float[N,3]: world position of each end-effector
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkWorldPositions(endEffectorId, flatten)

    def getEndEffectorPositions(self, endEffectorId=None, wrtLinkId=None, flatten=True):
        """
        Return the position of the end-effector(s) wrt the position of (an)other link(s). By default, it is the base.

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the position of all end-effectors.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the position wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 end-effector:
                float[3]: position of the end-effector wrt the link
            if multiple end-effectors:
                float[Nx3], float[N,3]: position of each end-effector wrt the link
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkPositions(endEffectorId, wrtLinkId, flatten)

    def getEndEffectorWorldOrientations(self, endEffectorId=None, flatten=True):
        """
        Return the end-effector orientation (in the Cartesian world space).

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the orientation of all end-effectors.
            flatten (bool): if True, it will return a 1D array of float numbers instead of an array of quaternion

        Returns:
            if 1 end-effector:
                quaternion: orientation of the end-effector (w,x,y,z)
            if multiple end-effectors:
                float[Nx4], quaternion[N]: orientation of each end-effector (w,x,y,z)
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkWorldOrientations(endEffectorId, flatten)

    def getEndEffectorOrientations(self, endEffectorId=None, wrtLinkId=None, flatten=True):
        """
        Return the end-effector orientation wrt to specified link(s). By default, it will be wrt to the base
        orientation.

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the orientation of all end-effectors.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the orientation wrt to the base.
            flatten (bool): if True, it will return a 1D array of float numbers instead of an array of quaternion

        Returns:
            if 1 end-effector:
                quaternion: orientation of the end-effector (w,x,y,z)
            if multiple end-effectors:
                float[Nx4], quaternion[N]: orientation of each end-effector (w,x,y,z)
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkOrientations(endEffectorId, wrtLinkId, flatten)

    def getEndEffectorWorldVelocities(self, endEffectorId=None, flatten=True):
        """
        Return the linear and angular velocities (expressed in the Cartesian world space coordinates) for the given
        end-effector(s).

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the velocities of all end-effectors.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 end-effector:
                float[6]: linear and angular velocity of the end-effector in the Cartesian world space
            if multiple end-effectors:
                float[Nx6], float[N,6]: linear and angular velocity of each end-effector
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkWorldVelocities(endEffectorId, flatten)

    def getEndEffectorVelocities(self, endEffectorId=None, wrtLinkId=None, flatten=True):
        """
        Return the linear and angular velocities (expressed in the Cartesian world space coordinates) for the given
        end-effector(s) wrt to specified link(s). By default, it is the base.

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the velocities of all end-effectors.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the velocities wrt to the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 end-effector:
                float[6]: linear and angular velocity of the end-effector
            if multiple end-effectors:
                float[Nx6], float[N,6]: linear and angular velocity of each end-effector
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkVelocities(endEffectorId, wrtLinkId, flatten)

    def getEndEffectorWorldLinearVelocities(self, endEffectorId=None, flatten=True):
        """
        Return the linear velocities (expressed in the Cartesian world space coordinates) for the given
        end-effector(s).

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the linear velocities of all end-effectors.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 end-effector:
                float[3]: linear velocity of the end-effector in the Cartesian world space
            if multiple end-effectors:
                float[Nx3], float[N,3]: linear velocity of each end-effector
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkWorldLinearVelocities(endEffectorId, flatten)

    def getEndEffectorLinearVelocities(self, endEffectorId=None, wrtLinkId=None, flatten=True):
        """
        Return the linear velocities (expressed in the Cartesian world space coordinates) for the given
        end-effector(s) wrt to specified link(s). By default, it is the base.

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the linear velocities of all end-effectors.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the linear velocities wrt to
                the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 end-effector:
                float[3]: linear velocity of the end-effector
            if multiple end-effectors:
                float[Nx3], float[N,3]: linear velocity of each end-effector
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkLinearVelocities(endEffectorId, wrtLinkId, flatten)

    def getEndEffectorWorldAngularVelocities(self, endEffectorId=None, flatten=True):
        """
        Return the angular velocities (expressed in the Cartesian world space coordinates) for the given
        end-effector(s).

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the angular velocities of all end-effectors.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 end-effector:
                float[3]: angular velocity of the end-effector in the Cartesian world space
            if multiple end-effectors:
                float[Nx3], float[N,3]: angular velocity of each end-effector
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkWorldAngularVelocities(endEffectorId, flatten)

    def getEndEffectorAngularVelocities(self, endEffectorId=None, wrtLinkId=None, flatten=True):
        """
        Return the angular velocities (expressed in the Cartesian world space coordinates) for the given
        end-effector(s) wrt to specified link(s). By default, it is the base.

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the angular velocities of all end-effectors.
            wrtLinkId (int, int[N], None): the other link id(s). If None, returns the angular velocities wrt to
                the base.
            flatten (bool): if True, it will return a 1D array instead of a 2D array

        Returns:
            if 1 end-effector:
                float[3]: angular velocity of the end-effector
            if multiple end-effectors:
                float[Nx3], float[N,3]: angular velocity of each end-effector
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkAngularVelocities(endEffectorId, wrtLinkId, flatten)

    def getEndEffectorAccelerations(self, endEffectorId=None, flatten=True):
        raise NotImplementedError

    def getEndEffectorForce(self, endEffectorId=None, flatten=True):
        raise NotImplementedError

    def getEndEffectorContacts(self, endEffectorId=None):
        """
        Check if the given end-effector(s) is/are in contact with something in the environment, and return all the
        contact points involving the given robot end-effector(s).

        Warnings: note that in reality, you can't know if your end-effector(s) is/are in contact with an object unless
        there is a sensor attached to it. However, this can be useful in simulation to optimize, for instance,
        trajectories.

        Args:
            endEffectorId (int, int[N], None): end-effector id, or list of desired end-effector ids.
                If None, get the contacts of all end-effectors.

        Returns:
            if 1 end-effector:
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
            if multiple end-effectors: list of above
        """
        if endEffectorId is None:
            endEffectorId = self.end_effectors
        return self.getLinkContacts(endEffectorId)

    ##############
    # Transforms #
    ##############

    def getHomogeneousTransform(self, position, orientation):
        """
        Return the Homogeneous transform matrix given the position vector and the orientation.

        Args:
            position (float[3]): position vector
            orientation (np.quaternion, float[4], float[3,3], float[3]): orientation

        Returns:
            float[4,4]: homogeneous matrix
        """
        if isinstance(orientation, quaternion.quaternion):
            R = quaternion.as_rotation_matrix(orientation)
        else:
            orientation = np.array(orientation)
            if orientation.shape == (3,): # RPY Euler angles
                R = self.sim.getMatrixFromQuaternion(self.sim.getQuaternionFromEuler(orientation))
                R = np.array(R).reshape(3,3)
            elif orientation.shape == (4,): # quaternion in the form (x,y,z,w)
                R = np.array(self.sim.getMatrixFromQuaternion(orientation)).reshape(3,3)
            elif orientation.shape == (3,3): # Rotation matrix
                R = orientation
            else:
                raise ValueError("Expecting a quaternion, RPY Euler angles, or rotation matrix")

        H = np.vstack((np.hstack((R, position.reshape(-1,1))), np.array([[0,0,0,1]])))
        return H

    ##############
    # Kinematics #
    ##############

    # TODO: allow to slice the Jacobian to only get what interests the user
    def calculateJacobian(self, linkId, q=None, localPosition=None):
        """
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q), J_{ang}(q)]^T`, such that:

        .. math:: v = [\dot{p}, \omega]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            linkId (int): linkId
            q (float[N]): joint positions of size N, where N is the number of DoFs. If None, it will compute q based
                on the current joint positions.
            localPosition: the point on the specified link to compute the Jacobian (in link local coordinates around
                its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            float[6,N], float[6,(6+N)]: full geometric (linear and angular) Jacobian matrix. The number of columns
                depends if the base is fixed or floating.
        """
        if q is None:
            q = self.getJointPositions()
        else:
            if len(q) != len(self.joints):
                raise ValueError("The length of q ({}) is different from the number of DoFs"
                                 " ({}).".format(len(q), len(self.joints)))

        if isinstance(q, np.ndarray):
            q = q.tolist() # Note that q has to be a list; it doesn't work if numpy array in Pybullet
        dq = [0]*len(self.joints)

        # specify point on the link
        if localPosition is None:
            localPosition = self.sim.getLinkState(self.id, linkId)[2] # Link CoM position in the link frame

        # calculate full jacobian
        lin_jac, ang_jac = self.sim.calculateJacobian(self.id, linkId, localPosition=localPosition,
                                                      objPositions=q, objVelocities=dq, objAccelerations=dq)

        return np.vstack((lin_jac, ang_jac))

    def calculateLinearJacobian(self, linkId, q=None, localPosition=None):
        """
        Return the full linear (geometric) Jacobian matrix :math:`J_{lin}(q)`, such that:

        .. math:: \dot{p} = J_{lin}(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            linkId (int): link id
            q (float[N]): joint positions of size N, where N is the number of DoFs. If None, it will compute q based
                on the current joint positions.
            localPosition: the point on the specified link to compute the Jacobian (in link local coordinates around
                its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            float[3,N], float[3,(6+N)]: full linear geometric Jacobian matrix. The number of columns depends if the
                base is fixed or floating.
        """
        return self.calculateJacobian(linkId, q, localPosition)[:3]

    def calculateAngularJacobian(self, linkId, q=None, localPosition=None):
        """
        Return the full angular (geometric) Jacobian matrix :math:`J_{ang}(q)`, such that:

        .. math:: \omega = J_{ang}(q) \dot{q}

        where :math:`\omega` is the link angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            linkId (int): link id
            q (float[N]): joint positions of size N, where N is the number of DoFs. If None, it will compute q based
                on the current joint positions.
            localPosition: the point on the specified link to compute the Jacobian (in link local coordinates around
                its center of mass). If None, it will use the CoM position (in the link frame).

        Returns:
            float[3,N], float[3,(6+N)]: full angular geometric Jacobian matrix. The number of columns depends if the
                base is fixed or floating.
        """
        return self.calculateJacobian(linkId, q, localPosition)[3:]

    # aliases
    getJacobian = calculateJacobian
    getGeometricJacobian = getJacobian
    getLinearJacobian = calculateLinearJacobian
    getAngularJacobian = calculateAngularJacobian

    def getJacobianDerivativeRPYToAngularVelocity(self, rpyAngle):
        """
        Return the Jacobian that maps RPY angle rates to angular velocities, i.e. :math:`\omega = T(\phi) \dot{\phi}`.

        Warnings: :math:`T` is singular when the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}`

        Args:
            rpyAngle (float[3]): RPY Euler angles [rad]

        Returns:
            float[3,3]: Jacobian matrix that maps RPY angle rates to angular velocities.
        """
        r,p,y = rpyAngle
        T = np.array([[1., 0., np.sin(p)],
                      [0., np.cos(r), -np.cos(p) * np.sin(r)],
                      [0., np.sin(r), np.cos(p) * np.cos(r)]])
        return T

    def getJacobianDerivativeZYZToAngularVelocity(self, zyzAngle):
        """
        Return the Jacobian that maps ZYZ angle rates to angular velocities, i.e. :math:`\omega = T(\phi) \dot{\phi}`.

        Warnings: :math:`T` is singular when the angle associated with `Y` is :math:`0` or :math:`\pi`.

        Args:
            rpyAngle (float[3]): ZYZ Euler angles [rad]

        Returns:
            float[3,3]: Jacobian matrix that maps ZYZ angle rates to angular velocities.
        """
        z, y = zyzAngle[:2]
        T = np.array([[0., -np.sin(z), np.cos(z) * np.sin(y)],
                      [0., np.cos(z), np.sin(z) * np.sin(y)],
                      [1., 0., np.cos(y)]])
        return T

    def getAnalyticalJacobian(self, jacobian, rpyAngle):
        """
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
            jacobian (float[6,N], float[6,6+N]): full geometric Jacobian.
            rpyAngle (float[3]): RPY Euler angles

        Returns:
            float[6,N], foat[6,(6+N)]: the full analytical Jacobian. The number of columns depends if the base is fixed
                or floating.
        """
        T = self.getJacobianDerivativeRPYToAngularVelocity(rpyAngle)
        Tinv = np.linalg.inv(T)
        Ja = np.vstack((np.hstack((np.identity(3), np.zeros((3,3)))),
                        np.hstack((np.zeros((3,3)), Tinv)))).dot(jacobian)
        return Ja

    def getAngularVelocitiesFromDerivativeRPY(self, rpyAngle, dRPY):
        """
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
            rpyAngle (float[3]): RPY Euler angles [rad]
            dRPY (float[3]): time derivative of RPY Euler angles [rad/s]

        Returns:
            float[3]: angular velocities [rad/s]
        """
        T = self.getJacobianDerivativeRPYToAngularVelocity(rpyAngle)
        return T.dot(dRPY)

    def getDerivativeRPYFromAngularVelocities(self, rpyAngle, angularVelocity):
        """
        Return the time derivative of RPY Euler angles :math:`\dot{\phi}` given the angular velocities :math:`\omega`.

        . .math:: \dot{\phi} = T^{-1}(\phi) \omega

        Warning: if the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}`, then :math:`T` is singular, and the
        corresponding angular velocities :math:`\omega` are not defined.

        Args:
            rollPitchYaw (float[3]): RPY Euler angles [rad]
            angularVelocity (float[3]): angular velocities [rad/s]

        Returns:
            float[3]: time derivative of RPY Euler angles [rad/s]

        Raises:
            LinAlgError: if singular configuration.
        """
        T = self.getJacobianDerivativeRPYToAngularVelocity(rpyAngle)
        Tinv = np.linalg.inv(T)
        return Tinv.dot(angularVelocity)

    def getJJT(self, jacobian):
        """
        Given the Jacobian, it returns :math:`JJ^T`. This relation is used in many places in robotics.

        Args:
            jacobian (float[D,N]): Jacobian matrix

        Returns:
            float[D,D]: :math:`JJ^T`
        """
        return jacobian.dot(jacobian.T)

    def getDampedLeastSquaresInverse(self, jacobian, dampingFactor=0.01):
        """
        Return the damped least-squares (DLS) inverse, given by:

        .. math:: \hat{J} = J^T (JJ^T + k^2 I)^{-1}

        which can then be used to get joint velocities :math:`\dot{q}` from the cartesian velocities :math:`v`, using
        :math:`\dot{q} = \hat{J} v`.

        Args:
            jacobian (float[D,N]): Jacobian matrix
            dampingFactor (float): damping factor

        Returns:
            float[N,D]: DLS inverse matrix
        """
        J, k = jacobian, dampingFactor
        return (J.T).dot(np.linalg.inv(J.dot(J.T) + k**2 * np.identity(J.shape[0])))

    # alias
    getDLSInverse = getDampedLeastSquaresInverse

    def getPinvJacobian(self, jacobian):
        """
        Return the right pseudo-inverse of the jacobian, i.e. :math:`J^\dagger = J^T(JJ^T)^{-1}`.

        Args:
            jacobian (float[D,N]): Jacobian matrix

        Returns:
            float[N,N]: right pseudo-inverse of the Jacobian
        """
        return np.linalg.pinv(jacobian)

    def getNullSpaceProjector(self, jacobian):
        """
        The null space projector :math:`P` is the matrix that projects any vectors to the null space of :math:`J`.
        This is given by: :math:`P = (I - J^\dagger J)`, where :math:`J^\dagger = J^T(JJ^T)^{-1}` is the right
        pseudo-inverse of the jacobian :math:`J`. This is notably used to perform inverse kinematics, where
        :math:`\dot{q} = J^\dagger v + P \dot{q}_0` with :math:`\dot{q}_0` representing arbitrary joint velocities.

        Args:
            jacobian (float[D,N]): Jacobian matrix

        Returns:
            float[N,N]: null space projector matrix
        """
        J = jacobian
        JJT = self.getJJT(jacobian)
        I = np.identity(J.shape[1])
        return (I - self.getPinvJacobian(J=J).dot(J))

    def computeManipulabilityMeasure(self, jacobian):
        """
        Compute the manipulability measure `w(q) = sqrt( det(J(q)J(q)^T) )`. This is useful to get a general sense
        about the manipulation ability of the manipulator. This term, for instance, vanishes at singular
        configurations (see [1]).

        Args:
            jacobian (float[D,N]): Jacobian matrix

        Returns:
            float: manipulability measure :math:`w(q)`

        References:
            [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010, chap 3.5 and 3.9
        """
        return np.sqrt(np.linalg.det(self.getJJT(jacobian)))

    def isInSingularConfiguration(self, jacobian):
        """
        Return True if we are in a singular configuration.

        Singularities are interesting because (see [1]):
        - they represent configurations where the mobility of the manipulator is reduced
        - infinite solutions to the IK problem may exist
        - around them, small velocities in the task/operational space may cause large velocities in the joint space

        Args:
            jacobian (float[D,N]): Jacobian matrix

        Returns:
            bool: True if in a singular configuration

        References:
            [1] "Robotics: Modelling, Planning, and Control" (book), Siciliano et al., 2010, chap 3.3
        """
        # TODO: define close to singular configuration using SVD
        J = jacobian
        m = np.min(J.shape)
        r = np.linalg.matrix_rank(J) # this uses SVD to compute the rank
        return (r < m)

    def getJointVelocitiesFromCartesianVelocities(self, jacobian, velocity):
        """
        Return the joint velocities :math:`\dot{q}` from the cartesian velocities :math:`v`.

        .. math:: \dot{q} = J^\dagger v

        where :math:`J^\dagger` is the right pseudo-inverse of J, i.e. :math:`J^\dagger = J^T(JJ^T)^{-1}`.

        Args:
            jacobain (float[3,N], float[6,N]): Jacobian matrix
            velocity (float[3], float[6]): linear and/or angular velocities

        Returns:
            float[N]: joint velocities
        """
        Jpinv = self.getPinvJacobian(jacobian)
        return Jpinv.dot(velocity)

    def getCartesianVelocitiesFromJointVelocities(self, jacobian, dq):
        """
        Return the Cartesian velocities :math:`v = [\dot{p}, \omega]^T` where :math:`\dot{p}` and :math:`\omega`
        are the linear and angular velocities, respectively.

        .. math:: v = J(q) \dot{q}

        Returns:
            float[6]: Cartesian linear and angular velocities
        """
        return jacobian.dot(dq)

    # TODO: implement IK for several links (also by exploiting the null space)
    def calculateInverseKinematics(self, linkId, targetPosition, targetOrientation=None,
                  lowerLimits=None, upperLimits=None, jointRanges=None, restPoses=None,
                  jointDamping=None, maxIter=1, threshold=1e-4):
        """
        Compute the FULL Inverse kinematics; it will return a position for all the actuated joints.

        Args:
            linkId (int): link id
            targetPosition (float[3]): target position
            targetOrientation (float[4]): target orientation

        Returns:
            float[M]: joint positions
        """
        # build dictionary
        d = {}

        # orientation for IK
        if targetOrientation is not None:
            d[targetOrientation] = targetOrientation

        # null-space
        if not (lowerLimits is None or upperLimits is None or jointRanges is None or restPoses is None):
            d[lowerLimits], d[upperLimits], d[jointRanges], d[restPoses] = lowerLimits, upperLimits, jointRanges, \
                                                                           restPoses

        # for damped IK
        if jointDamping is not None:
            d[jointDamping] = jointDamping

        # perform IK
        if maxIter < 2 or maxIter is None:
            # calculate joint positions solving IK and return them
            return np.array(self.sim.calculateInverseKinematics(self.id, linkId, targetPosition, **d))
        else:
            # perform IK for a certain number of steps
            closeEnough = False
            iter, dist = 0, np.inf

            for iter in range(maxIter):
                # calculate IK
                q = self.sim.calculateInverseKinematics(self.id, linkId, targetPosition, **d)

                # set positions
                self.setJointPositions(q, self.joints)

                # calculate position error in task/operation space
                pos = self.getLinkPositions(linkId)
                dist = np.linalg.norm(targetPosition - pos)

                # check if over
                if dist < threshold:
                    break

            return q

    def hardPriorities(self, jacobians, taskVelocities, method='backtrack'):
        """
        Return dq.

        Args:
            jacobians:
            taskVelocities:
            methods: 'successive', 'augmented', 'backtrack'.

        Returns:

        """
        pass

    ############
    # Dynamics #
    ############

    def calculateInverseDynamics(self, des_ddq, dq=None, q=None):
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
            q (float[M]): joint positions
            dq (float[M]): joint velocities
            des_ddq (float[M]): desired joint accelerations

        Returns:
            float[M]: joint torques computed using the rigid-body equation of motion

        References:
            [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        # if the joint velocities and positions are not provided, read them
        if dq is None:
            dq = self.getJointVelocities()
        if q is None:
            q = self.getJointPositions()

        # convert numpy arrays to lists
        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(dq, np.ndarray):
            dq = dq.tolist()
        if isinstance(des_ddq, np.ndarray):
            des_ddq = des_ddq.tolist()

        # return the joint torques to be applied for the desired joint accelerations
        return np.array(self.sim.calculateInverseDynamics(self.id, q, dq, des_ddq))

    # alias
    calculateID = calculateInverseDynamics

    def calculateForwardDynamics(self, torques, dq=None, q=None):
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
            q (float[M]): joint positions
            dq (float[M]): joint velocities
            torques (float[M]): desired joint torques

        Returns:
            float[M]: joint accelerations computed using the rigid-body equation of motion

        References:
            [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        # if the joint velocities and positions are not provided, read them
        if dq is None:
            dq = self.getJointVelocities()
        if q is None:
            q = self.getJointPositions()

        # compute and return joint accelerations
        torques = np.array(torques)
        Hinv = np.linalg.inv(self.calculateMassMatrix(q))
        C = self.calculateInverseDynamics(np.zeros(len(q)), dq=dq, q=q)
        acc = Hinv.dot(torques - C)
        return acc

    # alias
    calculateFD = calculateForwardDynamics

    def calculateMassMatrix(self, q=None, qIdx=None):
        """
        Return the mass/inertia matrix :math:`H(q)`.

        Warnings: If the base is floating, it will return a [6+N,6+N] inertia matrix, where N is the number of actuated
            joints. If the base is fixed, it will return a [N,N] inertia matrix

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            qIdx (slice, None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N).

        Returns:
            float[N,N], float[6+N,6+N], float[M,M]: inertia matrix
        """
        if q is None:
            q = self.getJointPositions()
        else:
            if len(q) != self.getNumberOfDoFs():
                raise ValueError("All the joint positions need to be given to this method. You can then slice the"
                                 "inertia matrix afterward.")

        # TODO: we need to get all the joints (even the fixed ones) --> need to test
        # make sure that we have all the joints even the fixed ones
        qAug = np.zeros(self.getNumberOfJoints())
        qAug[self.joints] = q
        qAug = qAug.tolist()  # Note that pybullet doesn't accept numpy arrays here

        if qIdx is None:
            return np.array(self.sim.calculateMassMatrix(self.id, qAug))
        return np.array(self.sim.calculateMassMatrix(self.id, qAug))[qIdx, qIdx]

    # alias
    getInertiaMatrix = calculateMassMatrix

    def getCartesianInertiaMatrix(self, H=None, Ja=None):
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
                provided and the linkId

        Returns:
            float[6,6]: Cartesian inertia matrix
        """
        Ja_inv = np.linalg.inv(Ja)
        return Ja_inv.T.dot(H).dot(Ja_inv)

    def getKineticEnergy(self, q=None, dq=None, qIdx=None):
        """
        Return the kinetic energy due to the movement of the specified joint(s).

        .. math:: T(q,\dot{q}) = \frac{1}{2} \dot{q}^T H(q) \dot{q}

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (float[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).
            qIdx (slice, None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: kinetic energy
        """
        if dq is None:
            dq = self.getJointVelocities()
            if qIdx is not None and len(dq) != len(qIdx):
                dq = dq[qIdx]
        H = self.calculateMassMatrix(q, qIdx)
        return 1./2 * dq.dot(H.dot(dq))

    def getGravityPotentialEnergy(self, q=None, qIdx=None, g=np.array((0.,0.,-9.81))):
        """
        Return the potential energy due to gravity.

        .. math:: V(q) = - \sum_{i=1}^N m_{l_i} g^T p_{l_i}

        where :math:`l_i` represents the link `i`, :math:`m_l` is the mass of the link, :math:`g` is the gravity
        vector, and :math:`p_l` is the position of the link.

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. THIS IS CURRENTLY
                NOT USED, as we can get the link positions from the simulator (instead of using forward kinematics).
            qIdx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: potential energy due to gravity
        """
        linkIds = list(range(self.getNumberOfLinks()))
        p = self.getLinkWorldPositions(linkId=linkIds, flatten=False)
        m = self.getLinkMasses(linkId=linkIds)
        if qIdx is not None:
            p = p[self.joints[qIdx]]
            m = m[self.joints[qIdx]]
        return np.sum((p.T * m).T * g)

    def getPotentialEnergy(self, q=None, dq=None, qIdx=None):
        """
        Return the potential energy of the system.

        WARNING: Note that we currently assume rigid body systems (thus rigid links). With this assumption, the
        potential energy is only due to gravitational forces. So, for now this is just an alias to
        `getGravityPotentialEnergy`.

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (float[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).
            qIdx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: potential energy
        """
        return self.getGravityPotentialEnergy(q, qIdx)

    def getLagrangian(self, q=None, dq=None, qIdx=None):
        """
        Return the Lagrangian evaluate at the given configuration.

        .. math:: L(q, \dot{q}) = T(q, \dot{q}) - V(q)

        where :math:`T` and :math:`V` are the kinetic and potential energy respectively.

        Args:
            q (float[N], None): joint positions of size N, where N is the total number of DoFs. If None, it will
                get the current joint positions (but note that this could lead to a decrease of performance).
            dq (float[M], None): joint velocities of size M (with 0 < M <= N). If None, it will
                get the current joint velocities (but note that this could lead to a decrease of performance).
            qIdx (int[M], None): if provided, it will slice the inertia matrix at the given q indices (0 < M <= N),
                and the joint velocities vector.

        Returns:
            float: value of the Lagrangian
        """
        T = self.getKineticEnergy(q=q, dq=dq, qIdx=qIdx)
        V = self.getPotentialEnergy(q=q, qIdx=qIdx)
        return T - V

    def applyExternalForce(self, force, linkId=-1, position=(0.,0.,0.), flag=1):
        """
        Apply an external force on a body, or a link of the body. Note that after each simulation step, the external
        forces are cleared to 0.

        Warnings: This does not work when using `sim.setRealTimeSimulation(1)`.

        Args:
            force (float[3]): Cartesian forces to be applied on the body
            linkId (int): link id to apply the force, if -1 it will apply the force on the base
            position (float[3]): position on the link where the force is applied.
            frameFlag (int): allows to specify the coordinate system of force/position. sim.LINK_FRAME (=1) for local
                link frame, and sim.WORLD_FRAME (=2) for world frame. By default, it is the link frame.
        """
        self.sim.applyExternalForce(self.id, linkId, force, position, flag)

    def applyExternalTorque(self, torque, linkId=-1, flag=1):
        """
        Apply an external torque on a body, or a link of the body. Note that after each simulation step, the external
        torques are cleared to 0.

        Warnings: This does not work when using `sim.setRealTimeSimulation(1)`.

        Args:
            torque (float[3]): Cartesian torques to be applied on the body
            linkId (int): link id to apply the torque, if -1 it will apply the torque on the base
            frameFlag (int): allows to specify the coordinate system of torque. sim.LINK_FRAME (=1) for local
                link frame, and sim.WORLD_FRAME (=2) for world frame. By default, it is the link frame.
        """
        self.sim.applyExternalTorque(self.id, linkId, force, flag)

    def getJointTorquesFromCartesianWrench(self, jacobian, wrench):
        """
        Return the joint torques from the given Cartesian wrench (=force and torque) using the provided Jacobian.

        .. math:: \tau = J^T(q) f

        where :math:`\tau` are the joint torques, :math:`f` is the wrench vector (i.e. it contains the forces/torques
        applied at the link), and :math:`J` is the geometric Jacobian.

        Returns:
            float[N]: joint torques [Nm]
        """
        return jacobian.T.dot(wrench)

    def getCartesianWrenchFromJointTorques(self, jacobian, torque):
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

    def enableCoriolisAndGravityCompensation(self, enable=True):
        """
        Enable the gravity and Coriolis compensation when applying torques. This will automatically compute these
        terms and add them automatically to the given torques when using torque control.

        Args:
            enable (bool): If True, enable the gravity and Coriolis compensation when applying torques.
        """
        self.coriolisAndGravityCompensation = enable

    def getCoriolisAndGravityCompensationTorques(self, q=None, dq=None, qIdx=None):
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
            q = self.getJointPositions()
        if dq is None:
            dq = self.getJointVelocities()

        ddq = np.zeros(len(self.joints)).tolist()

        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(dq, np.ndarray):
            dq = dq.tolist()

        if qIdx is None:
            return np.array(self.sim.calculateInverseDynamics(self.id, q, dq, ddq))
        return np.array(self.sim.calculateInverseDynamics(self.id, q, dq, ddq))[qIdx]

    def getGravityCompensationTorques(self, q=None, qIdx=None):
        if q is None:
            q = self.getJointPositions()
        dq = np.zeros(len(q))
        return self.getCoriolisAndGravityCompensationTorques(q, dq, qIdx)

    def applyCoriolisAndGravityCompensation(self, q=None, dq=None, qIdx=None, external_torques=0.):
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
        torques = self.getCoriolisAndGravityCompensationTorques(q,dq,qIdx)
        self.setJointTorques(jointId, torques + external_torques)

    # TODO: finish to implement the method + think about multiple links + think about dimensions
    def getActiveCompliantTorques(self, q=None, dq=None, qIdx=None, jacobian=None, linkVelocity=None, linkId=None, kd=60):
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
            q = self.getJointPositions()
        if dq is None:
            dq = self.getJointVelocities()
        if jacobian is None:
            jacobian = self.calculateJacobian(linkId, q)
        if linkVelocity is None:
            linkVelocity = self.getLinkWorldVelocities(linkId)
        if isinstance(kd, int):
            kd = kd * np.identity(6)

        torques = self.getCoriolisAndGravityCompensationTorques(q, dq, qIdx)
        torques += jacobian.T.dot(-kd * linkVelocity)
        return torques

    # TODO: finish to implement the method
    def applyActiveCompliance(self, q=None, dq=None, qIdx=None, external_torques=0.):
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
        torques = self.getActiveCompliantTorques(q, dq, qIdx)
        self.setJointTorques(jointId, torques + external_torques)

    def getImpedanceTorques(self, x=0, dx=0, ddx=0):
        """
        .. math:: F_{a} = H_m (\ddot{x} - \ddot{x}_d) + D_m (\dot{x} - \dot{x}_d) + K_m (x - x_d)
        """
        raise NotImplementedError

    def applyTaskImpedanceControl(self):
        """
        .. math:: F_{a} = H_m (\ddot{x} - \ddot{x}_d) + D_m (\dot{x} - \dot{x}_d) + K_m (x - x_d)
        """
        raise NotImplementedError

    def getAttractorTorques(self):
        """
        The torques to be applied are given by:

        .. math::  \tau = C(q,\dot{q}) \dot{q} + g(q) + J^T F

        where :math:`F = K(x_d - x) - D v` with :math:`x` and :math:`v` are the Cartesian position and velocities,
        and :math:`K` and :math:`D` are the stiffness and damping factor, respectively.
        """
        raise NotImplementedError

    ######################
    # Symbolic Equations #
    ######################

    def getSymbolicEquationsOfMotion(self):
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

    # alias
    getSymbolicEOM = getSymbolicEquationsOfMotion

    def linearizeEquationOfMotion(self, point=None):
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

    # alias
    linearizeEOM = linearizeEquationOfMotion

    ###########
    # Sensors #
    ###########

    def getNumberOfSensors(self):
        """
        Return the total number of sensors.

        Returns:
            int: total number of sensors.
        """
        return len(self.sensors)

    def enableJointForceTorqueSensor(self, jointId=None, enable=True):
        """
        Enable/disable the force/torque sensors of the specified joint(s).

        Warnings: Note that you should normally use a F/T sensor. However, enabling/disabling F/T sensors can be
        useful for debug among other things.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, enable/disable the F/T sensors
                on all the actuated joints.
            enable (bool): If True, it will enable the F/T sensors, otherwise it will disable them.
        """
        if isinstance(jointId, int):
            self.sim.enableJointForceTorqueSensor(self.id, jointId, enableSensor=enable)
        else:
            if jointId is None:
                jointId = self.joints
            for joint in jointId:
                self.sim.enableJointForceTorqueSensor(self.id, joint, enableSensor=enable)

    def disableJointForceTorqueSensor(self, jointId=None):
        """
        Disable the force/torque sensors of the specified joint(s).

        Warnings: Note that you should normally use a F/T sensor. However, enabling/disabling F/T sensors can be
        useful for debug among other things.

        Args:
            jointId (int, int[N], None): joint id, or list of joint ids. If None, disable the F/T sensors on all the
                actuated joints.
        """
        if isinstance(jointId, int):
            self.sim.enableJointForceTorqueSensor(self.id, jointId, enableSensor=0)
        else:
            if jointId is None:
                jointId = self.joints
            for joint in jointId:
                self.sim.enableJointForceTorqueSensor(self.id, joint, enableSensor=0)

    def getSensors(self, idx=None):
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

    def getIMU(self, idx=0):
        raise NotImplementedError

    def getForceTorqueSensor(self, idx=0):
        raise NotImplementedError

    def hasCamera(self):
        return False

    def getCamera(self, idx=0):
        raise NotImplementedError

    def getCameraImage(self, idx=0):
        raise NotImplementedError

    def getMainCamera(self):
        raise NotImplementedError

    def getMainCameraImage(self):
        raise NotImplementedError

    #############
    # Actuators #
    #############

    def getNumberOfActuators(self):
        """
        Return the total number of actuators.

        Returns:
            int: total number of actuators.
        """
        return len(self.actuators)

    def getActuators(self, idx=None):
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

    def getContacts(self):
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
        return self.sim.getContactPoints(bodyA=self.id)

    #########
    # Debug #
    #########

    def printJointInfo(self, jointId):
        """
        Print information about the given joint.

        Args:
            jointId (int): unique joint id.
        """
        jnt = self.sim.getJointInfo(self.id, jointId)
        print('joint index: {}'.format(jnt[0]))
        print('joint name: {}'.format(jnt[1]))
        print('joint type: {}'.format(self.getJointTypeStr(jnt[2])))
        print('q index: {}'.format(jnt[3]))
        print('qd index: {}'.format(jnt[4]))
        print('joint damping: {}'.format(jnt[6]))
        print('joint friction: {}'.format(jnt[7]))
        print('joint lower limit: {}'.format(jnt[8]))
        print('joint upper limit: {}'.format(jnt[9]))
        print('joint max force: {}'.format(jnt[10]))
        print('joint max velocity: {}'.format(jnt[11]))
        print('associated link name: {}'.format(jnt[12]))
        print('joint axis: {}'.format(jnt[13]))
        print('position wrt parent frame: {}'.format(jnt[14]))
        print('orientation wrt parent frame: {}'.format(jnt[15]))
        print('parent link index: {}'.format(jnt[16]))

    def printLinkInfo(self, linkId):
        """
        Print information about the given link. The information printed include the link frame position and
        orientation, its center of mass position and orientation, its dimensions, its mass, its local inertia
        diagonal, etc.

        Args:
            linkId (int): unique link id
        """
        pass

    def printRobotInfo(self):
        """
        Print general information about the robot.
        """
        print("\nRobot: {}".format(self))
        print("Number of DoFs: {}".format(self.getNumberOfDoFs()))
        print("Joint ids: {}".format(list(range(self.getNumberOfJoints()))))
        print("Actuated joint ids: {}".format(self.joints))
        print("Link names (associated with actuated joints): {}".format(self.getLinkNames(self.joints)))
        print("End-effector names: {}".format(self.getEndEffectorNames()))
        print("Floating base? {}".format(self.hasFloatingBase()))
        print("Total mass = {} kg".format(self.getTotalMass()))

    def addJointSlider(self, jointId=None):
        """
        Add a slider for the given joint id.

        Args:
            jointId (int, str, list of str/int, None): if int, the id is between {0, N} where N=number of non-fixed
                joint. If str, the name of the joint. If list/tuple, it contains the id or name of the joints.
                If None, add a slider for each non-fixed joint.
        """
        # show debug visualizer
        self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_GUI, 1)

        def getIndex(jnt):
            if isinstance(jnt, int): # joint id
                return jnt
            elif isinstance(jnt, str): # joint name
                return self.getJointIds(jnt)
            else:
                raise TypeError('Expecting a str or int for the joint: {}'.format(jnt))

        # get the joint indices
        if jointId is None:
            jointId = self.joints
        else:
            if isinstance(jointId, int):  # joint id
                jointId = [jointId]
            elif isinstance(jointId, str):  # joint name
                jointId = [self.getJointIds(jointId)]
            elif isinstance(jointId, collections.Iterable):
                jointId = [getIndex(jnt) for jnt in jointId]
            else:
                raise TypeError("jointId has to be a None, int, str, or a list/tuple of int/str.")

        # get informations about the joints
        names = self.getJointNames(jointId)
        limits = self.getJointLimits(jointId)
        positions = self.getJointPositions(jointId)
        lower_limits, upper_limits = limits[:, 0], limits[:, 1]
        # # apply lower and upper limit
        lower_limits[lower_limits < -2 * np.pi] = -2. * np.pi
        upper_limits[upper_limits > 2 * np.pi] = 2. * np.pi

        # add sliders in pybullet
        for i in range(len(jointId)):
            slider = self.sim.addUserDebugParameter(names[i], lower_limits[i], upper_limits[i], positions[i])
            self.joint_sliders[jointId[i]] = slider

    def updateJointSlider(self):
        """
        Read the specified joint slider value, and set the robot's corresponding joint to this one
        using position control.
        """
        # for each slider
        for jointId, slider in self.joint_sliders.items():
            # read joint value from slider
            pos = self.sim.readUserDebugParameter(slider)

            # set joint position to the read value
            self.sim.setJointMotorControl2(self.id, jointId, self.sim.POSITION_CONTROL, targetPosition=pos)

    # alias
    readJointSlider = updateJointSlider

    def removeJointSlider(self, jointId=None):
        """
        Remove the specified joint slider(s).

        Args:
            jointId (int, str, list of str/int, None): if int, the id is between {0, N} where N=number of non-fixed
                joint. If str, the name of the joint. If list/tuple, it contains the id or name of the joints.
                If None, add a slider for each non-fixed joint.
        """
        def getIndex(jnt):
            if isinstance(jnt, int): # joint id
                return jnt
            elif isinstance(jnt, str): # joint name
                return self.getJointIds(jnt)
            else:
                raise TypeError('Expecting a str or int for the joint: {}'.format(jnt))

        # get the joint indices
        if jointId is None:
            jointId = self.joints
        else:
            if isinstance(jointId, int):  # joint id
                jointId = [jointId]
            elif isinstance(jointId, str):  # joint name
                jointId = [self.getJointIds(jointId)]
            elif isinstance(jointId, collections.Iterable):
                jointId = [getIndex(jnt) for jnt in jointId]
            else:
                raise TypeError("jointId has to be a None, int, str, or a list/tuple of int/str.")

        # remove sliders in pybullet
        for joint in jointId:
            if joint in self.joint_sliders:
                self.sim.removeUserDebugItem(self.joint_sliders[joint])
                self.joint_sliders.pop(joint)

        # if no sliders anymore, remove the debug visualizer
        self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_GUI, 0)

    ####################
    # online plotting  # # WARNING: ALL THE FOLLOWING METHODS NEED A SIMULATOR IN WHICH TO RUN #
    ####################

    def plotJointPositions(self, jointId=None):
        raise NotImplementedError

    def plotJointVelocities(self, jointId=None):
        raise NotImplementedError

    def plotJointAccelerations(self, jointId=None):
        raise NotImplementedError

    def plotCoMPosition(self):
        raise NotImplementedError

    def plotCoMVelocity(self):
        raise NotImplementedError

    def plotCoMAcceleration(self):
        raise NotImplementedError

    def plotCartesianPositions(self, linkId=None):
        raise NotImplementedError

    def plotCartesianVelocities(self, linkId=None):
        raise NotImplementedError

    def plotCartesianAccelerations(self, linkId=None):
        raise NotImplementedError

    ########
    # draw # # WARNING: ALL THE FOLLOWING METHODS NEED A SIMULATOR IN WHICH TO RUN #
    ########

    def _drawSphere(self, position, radius=0.1, color=(1,1,1,1)):
        visual = self.sim.createVisualShape(self.sim.GEOM_SPHERE, radius=radius, rgbaColor=color)
        body = self.sim.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=position)
        return body

    def _drawCylinder(self, position, orientation, radius=1, height=1, color=(1,1,1,1)):
        visual = self.sim.createVisualShape(self.sim.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        orientation = self.angular_converter.convertFrom(orientation)  # convert to list
        body = self.sim.createMultiBody(baseMass=0., baseVisualShapeIndex=visual, basePosition=position,
                                        baseOrientation=orientation)
        return body

    def _drawFrame(self, position, orientation, radius, length):
        # quat = self.angular_converter.convertFrom(orientation)
        # R = np.array(self.sim.getMatrixFromQuaternion(quat)).reshape(3,3)
        R = quaternion.as_rotation_matrix(orientation)
        # H = np.vstack((np.hstack((R, position.reshape(-1,1))), np.array([[0,0,0,1]])))
        x = R.dot(np.array([length/2., 0, 0])) + position  # H.dot(np.array([length/2., 0, 0, 1]))[:3]
        y = R.dot(np.array([0, length/2., 0])) + position  # H.dot(np.array([0, length/2., 0, 1]))[:3]
        z = R.dot(np.array([0, 0, length/2.])) + position  # H.dot(np.array([0, 0, length/2., 1]))[:3]

        qx = quaternion.quaternion(0.707, 0.707, 0, 0)  # 90deg around x
        qy = quaternion.quaternion(0.707, 0, 0.707, 0)  # 90deg around y

        # draw x, y, z cylinders
        # self._drawCylinder(x, qy*orientation, radius, length, color=(1,0,0,1))
        self._drawCylinder(x, orientation*qy, radius, length, color=(1, 0, 0, 1))
        self._drawCylinder(y, orientation*qx, radius, length, color=(0, 1, 0, 1))
        self._drawCylinder(z, orientation, radius, length, color=(0, 0, 1, 1))

    def _drawDebugBox(self, bbMin, bbMax):
        (x0, y0, z0), (xf, yf, zf) = bbMin, bbMax
        self.sim.addUserDebugLine((x0, y0, z0), (x0, yf, z0), (1, 1, 1))
        self.sim.addUserDebugLine((x0, yf, z0), (x0, yf, zf), (1, 1, 1))
        self.sim.addUserDebugLine((x0, yf, zf), (x0, y0, zf), (1, 1, 1))
        self.sim.addUserDebugLine((x0, y0, zf), (x0, y0, z0), (1, 1, 1))

        self.sim.addUserDebugLine((xf, y0, z0), (xf, yf, z0), (1, 1, 1))
        self.sim.addUserDebugLine((xf, yf, z0), (xf, yf, zf), (1, 1, 1))
        self.sim.addUserDebugLine((xf, yf, zf), (xf, y0, zf), (1, 1, 1))
        self.sim.addUserDebugLine((xf, y0, zf), (xf, y0, z0), (1, 1, 1))

        self.sim.addUserDebugLine((x0, y0, z0), (xf, y0, z0), (1, 1, 1))
        self.sim.addUserDebugLine((x0, yf, z0), (xf, yf, z0), (1, 1, 1))
        self.sim.addUserDebugLine((x0, y0, zf), (xf, y0, zf), (1, 1, 1))
        self.sim.addUserDebugLine((x0, yf, zf), (xf, yf, zf), (1, 1, 1))

    def changeTransparency(self, alpha=0.5):
        """
        Change the transparency of a robot.

        WARNING: THIS CAN CHANGE THE COLOR OF SOME LINKS IF THEY WERE NOT DEFINED IN THE URDF!!

        Args:
            alpha (float): alpha channel. 1 is opaque, and 0 is completely transparent.
        """
        for shapeId in self.visualShapes:
            rgba = self.visualShapes[shapeId]['color']
            rgba[-1] = alpha
            self.sim.changeVisualShape(self.id, shapeId, rgbaColor=rgba)
            # print("Link {} - color: {}".format(link, rgba))

    def updateVisual(self):
        """
        Update all visuals.
        """
        pass

    def computeAndDrawCoMPosition(self, radius=0.05, color=(1,0,0,1)):
        """
        Compute the CoM and draw it as a sphere in the simulator.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot.
            color (float[4]): rgba color of the sphere. By default, it is red.

        Returns:
            float[3]: center of mass
        """
        self.getCoMPosition()
        self.drawCoMPosition(radius=radius, color=color)
        return self.com

    def drawCoMPosition(self, radius=0.05, color=(1, 0, 0, 1)):
        """
        Draw the CoM in the simulator.

        WARNING: `getCoMPosition()` must be called before calling this method. Otherwise, check the other method
        `computeAndDrawCoMPosition()`.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot
            color (float[4]): rgba color of the sphere. By default it is red.
        """
        if self.comVisual is None: # create visual shape if not already created
            comVisualShape = self.sim.createVisualShape(self.sim.GEOM_SPHERE, radius=radius, rgbaColor=color)
            self.comVisual = self.sim.createMultiBody(baseMass=0,
                                                      baseVisualShapeIndex=comVisualShape,
                                                      basePosition=self.com)
        else:  # set CoM position
            self.sim.resetBasePositionAndOrientation(self.comVisual, self.com, [0, 0, 0, 1])

    def removeCoM(self):
        """
        Remove the CoM from the simulator.
        """
        if self.comVisual is not None:
            self.sim.removeBody(self.comVisual)
            self.comVisual = None

    def getProjectedCoMPosition(self, max_depth=5):
        """
        Get the projected center of mass position.

        WARNING: This method only works in the simulator!! It requires some knowledge about the environment.

        Args:
            max_depth (float): if there is an object more than max_depth, it is not considered

        Returns:
            float[3], None: position of the projected CoM, or None if it couldn't project the CoM
        """
        com = self.getCoMPosition()
        object_id, _, _, hit_position, _ = self.sim.rayTest(com, com - np.array([0.,0.,max_depth]))[0]
        if object_id >= 0: # if there is a collision
            return hit_position # = projected com
        else:
            return None

    def computeAndDrawProjectedCoMPosition(self, radius=0.05, color=(0,0,1,1)):
        """
        Compute and draw the projected center of mass.

        Args:
            radius (float): radius of the sphere representing the CoM of the robot
            color (float[4]): rgba color of the sphere. By default it is blue.

        Returns:
            float[3], None: position of the projected CoM, or None if it couldn't project the CoM
        """
        projected_com = self.getProjectedCoMPosition()
        if projected_com is not None:
            # if visual shape not already created, create one
            if self.projectedCoMVisual is None:
                visualShape = self.sim.createVisualShape(self.sim.GEOM_SPHERE, radius=radius, rgbaColor=color)
                self.projectedCoMVisual = self.sim.createMultiBody(baseMass=0,
                                                                   baseVisualShapeIndex=visualShape,
                                                                   basePosition=projected_com)

            # otherwise update projected CoM position
            else:
                self.sim.resetBasePositionAndOrientation(self.projectedCoMVisual, projected_com, [0, 0, 0, 1])

        return projected_com

    def removeProjectedCoM(self):
        """
        Remove the projected CoM from the simulator.
        """
        if self.projectedCoMVisual is not None:
            self.sim.removeBody(self.projectedCoMVisual)
            self.projectedCoMVisual = None

    # def drawProjectedCoM(self, radius=0.05, color=(1,0,0,1)):
    #     """
    #     draw the projected CoM on the walking surface
    #     """
    #     pass

    def drawLinkCoMs(self, linkId=None, scaling=1.):
        """
        Draw the CoM of the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get links associated to
                actuated joints.
            scaling (float): scaling factor
        """
        if linkId is None:
            linkId = self.joints
        elif isinstance(linkId, int):
            linkId = [linkId]

        for link in linkId:
            if link in self.visualShapes:
                pos = self.getLinkWorldPositions(link)
                dim = self.visualShapes[link]['dimensions']
                # radius = min(dim) * scaling * 0.2
                radius = 0.01
                self._drawSphere(pos, radius, color=(0,0,0,1))

    def drawLinkFrames(self, linkId=None, scaling=1.):
        """
        Draw frames of the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get links associated to
                actuated joints.
            scaling (float): scaling factor
        """
        if linkId is None:
            linkId = [-1] + self.joints
        elif isinstance(linkId, int):
            linkId = [linkId]

        for link in linkId:
            if link in self.visualShapes:
                if link == -1:
                    pos, orientation = self.getBasePositionAndOrientation()
                else:
                    pos = self.getLinkFrameWorldPositions(link)
                    orientation = self.getLinkFrameWorldOrientations(link)
                dim = self.visualShapes[link]['dimensions']
                # radius = min(dim) * scaling * 0.2
                radius = 0.005 * scaling
                # self._drawSphere(pos, radius, color=(0,0,0,1))
                # length = 4*radius
                length = 0.05 * scaling
                self._drawFrame(pos, orientation, radius, length)

    def drawJointFrames(self, jointId=None):
        """
        Draw (actuated) joint frame
        """
        pass

    def drawBoundingBoxes(self, linkId=None):
        """
        Draw bounding box around the given link(s).

        Args:
            linkId (int, int[N], None): link id, or list of desired link ids. If None, get links associated to
                actuated joints.
        """
        if linkId is None:
            linkId = [-1] + self.joints
        elif isinstance(linkId, int):
            linkId = [linkId]

        for link in linkId:
            if link in self.visualShapes:
                bbMin, bbMax = self.sim.getAABB(self.id, link)
                self._drawDebugBox(bbMin, bbMax)

    def draw3DEllipsoid(self, position, orientation=(0.,0.,0.,1.), scale=(1.,1.,1.), color=(0,1,0,0.7)):
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
        visualShape = self.sim.createVisualShape(self.sim.GEOM_MESH, fileName=filename,
                                                 meshScale=scale, rgbaColor=color)
        ellipsoid = self.sim.createMultiBody(baseMass=0.,
                                             baseVisualShapeIndex=visualShape,
                                             basePosition=position,
                                             baseOrientation=orientation)
        return ellipsoid

    def getEllipsoidOrientationAndScale(self, X):
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
        orientation = self.sim.getQuaternionFromEuler([roll, pitch, yaw])

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

    def drawVelocityManipulabilityEllipsoid(self, linkId, Jlin=None, JJT=None, color=(0,1,0,0.7)):
        """
        evecs of JJ^T = directions
        singular values of JJ^T = dimensions

        Args:
            linkId (int): link id. This will be used to check where to draw the ellipsoid.
            J (float[3,N], None): linear Jacobian matrix. It doesn't need to be provided if `JJT` is given.
            JJT (float[3,3], None): if None, it will compute it using the provided linear Jacobian matrix.

        Returns:
            int: id of the visual ellipsoid
        """
        if JJT is None:
            if Jlin is None:
                raise ValueError("Please provide the linear Jacobian matrix")
            JJT = self.getJJT(Jlin)

        orientation, scale = self.getEllipsoidOrientationAndScale(JJT)

        # load ellipsoid
        position = self.getLinkWorldPositions(linkId)
        self.draw3DEllipsoid(position, orientation, scale=scale, color=color)

    def drawForceManipulabilityEllipsoid(self, linkId, J=None, JJT=None):
        """
        Kineto-statics duality: direction with good velocity manipulability is obtained a direction along which poor
        force manipulability is obtained.

        evecs((JJ^T)^{-1})

        Args:
            linkId:
            J:
            JJT:
        """
        pass

    def updateManipulabilityEllipsoid(self, linkId, ellipsoidId):
        """
        Update the position, orientation, and scaling of the given manipulability ellipsoid.

        Warnings: currently, the bullet simulator do not allow to update the scale, only the position and orientation.
        """
        raise NotImplementedError("Currently, this feature is not available on pybullet")
        #self.sim.resetBasePositionAndOrientation(ellipsoidId, position, orientation)

    def removeManipulabilityEllipsoid(self, ellipsoidId):
        """
        Remove the given ellipsoid manipulability ellipsoid.

        Args:
            ellipsoidId (int): id of the ellipsoid to remove
        """
        self.sim.removeBody(ellipsoidId)
