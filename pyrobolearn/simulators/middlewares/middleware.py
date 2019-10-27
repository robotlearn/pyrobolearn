# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the abstract middleware API.

Dependencies in PRL:
* NONE
"""


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Middleware(object):
    r"""Middleware (abstract) class

    Middleware can be provided to simulators which can then use them to send/receive messages.

    Here are the possible combinations between the different values for subscribe (S), publish (P), teleoperate (T),
    and command (C):

    - S=1, P=0, T=0: subscribes to the topics, and get the messages when calling the corresponding getter methods.
    - S=0, P=1, T=0: publishes the various messages to the topics when calling the corresponding setter methods.
    - S=1, P=0, T=1, C=0/1: get messages by subscribing to the topics that publish some commands/states. The
      received commands/states are then set in the simulator. Depending on the value of `C`, it will subscribe to
      topics that publish some commands (C=1) or states (C=0). Example: should we subscribe to joint trajectory
      commands, or joint states when teleoperating the robot in the simulator? This C value allows to specify
      which one we are interested in.
    - S=0, P=1, T=1: when calling the getters methods such as joint positions, velocities, and others, it also
      publishes the joint states. This is useful if we are moving/teleoperating the robot in the simulator.
    - S=0, P=0, T=1/0: doesn't do anything.
    - S=1, P=1, T=0: subscribes to some topics and publish messages to other topics. The messages are can be
      sent/received by calling the appropriate getter/setter methods.
    - S=1, P=1, T=1: not allowed, because teleoperating the robot is not a two-way communication process.
    """

    def __init__(self, subscribe=False, publish=False, teleoperate=False, command=True):
        """
        Initialize the middleware to communicate.

        Args:
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robots, and will read
              the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robots.
            teleoperate (bool): if True, it will move the robot based on the received or sent values based on the 2
              previous attributes :attr:`subscribe` and :attr:`publish`.
            command (bool): if True, it will subscribe to the joint commands. If False, it will subscribe to the
              joint states.
        """
        # set variables
        # self.is_subscribing = subscribe
        # self.is_publishing = publish
        # self.is_teleoperating = teleoperate
        # self.is_commanding = command
        self._subscribe, self._publish, self._teleoperate, self._command = False, False, False, False
        self._robots = {}  # {body_id: RobotMiddleware}
        self.switch_mode(subscribe=subscribe, publish=publish, teleoperate=teleoperate, command=command)

    ##############
    # Properties #
    ##############

    @property
    def is_subscribing(self):
        return self._subscribe

    # @is_subscribing.setter
    # def is_subscribing(self, subscribe):
    #     self._subscribe = bool(subscribe)

    @property
    def is_publishing(self):
        return self._publish

    # @is_publishing.setter
    # def is_publishing(self, publish):
    #     self._publish = bool(publish)

    @property
    def is_teleoperating(self):
        return self._teleoperate

    # @is_teleoperating.setter
    # def is_teleoperating(self, teleoperate):
    #     self._teleoperate = bool(teleoperate)

    @property
    def is_commanding(self):
        return self._command

    # @is_commanding.setter
    # def is_commanding(self, command):
    #     self._command = bool(command)

    # aliases
    subscribe = is_subscribing
    publish = is_publishing
    teleoperate = is_teleoperating
    command = is_commanding

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a readable string about the class."""
        return self.__class__.__name__

    def __del__(self):
        """Close/Delete the simulator."""
        self.close()

    def __copy__(self):
        """Return a shallow copy of the middleware. This can be overridden in the child class."""
        return self.__class__(subscribe=self.is_subscribing, publish=self.is_publishing,
                              teleoperate=self.is_teleoperating)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the middleware. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass.
        """
        # if the object has already been copied return the reference to the copied object
        if self in memo:
            return memo[self]

        # create a new copy of the simulator
        middleware = self.__class__(subscribe=self.is_subscribing, publish=self.is_publishing,
                                    teleoperate=self.is_teleoperating)

        memo[self] = middleware
        return middleware

    ###########
    # Methods #
    ###########

    def switch_mode(self, body_id=None, subscribe=None, publish=None, teleoperate=None, command=None):
        """
        Switch middleware mode.

        Here are the possible combinations between the different values for subscribe (S), publish (P),
        teleoperate (T), and command (C):

        - S=1, P=0, T=0: subscribes to the topics, and get the messages when calling the corresponding getter methods.
        - S=0, P=1, T=0: publishes the various messages to the topics when calling the corresponding setter methods.
        - S=1, P=0, T=1, C=0/1: get messages by subscribing to the topics that publish some commands/states. The
          received commands/states are then set in the simulator. Depending on the value of `C`, it will subscribe to
          topics that publish some commands (C=1) or states (C=0). Example: should we subscribe to joint trajectory
          commands, or joint states when teleoperating the robot in the simulator? This C value allows to specify
          which one we are interested in.
        - S=0, P=1, T=1: when calling the getters methods such as joint positions, velocities, and others, it also
          publishes the joint states. This is useful if we are moving/teleoperating the robot in the simulator.
        - S=0, P=0, T=1/0: doesn't do anything.
        - S=1, P=1, T=0: subscribes to some topics and publish messages to other topics. The messages are can be
          sent/received by calling the appropriate getter/setter methods.
        - S=1, P=1, T=1: not allowed, because teleoperating the robot is not a two-way communication process.

        Args:
            body_id (int): unique body id to switch the mode.
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robots, and will read
              the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robots.
            teleoperate (bool): if True, it will move the robot based on the received or sent values based on the 2
              previous attributes :attr:`subscribe` and :attr:`publish`.
            command (bool): if True, it will subscribe to the joint commands. If False, it will subscribe to the
              joint states.
        """
        if body_id is None:
            if subscribe is None:
                subscribe = self.subscribe
            if publish is None:
                publish = self.publish
            if teleoperate is None:
                teleoperate = self.teleoperate
            if command is None:
                command = self.command

            if teleoperate and publish and subscribe:
                raise ValueError("The three following arguments 'subscribe', 'publish', and 'teleoperate' can not be "
                                 "all true at the same time. Select maximum two to be set to True (see method "
                                 "documentation).")

            self._subscribe = bool(subscribe)
            self._publish = bool(publish)
            self._teleoperate = bool(teleoperate)
            self._command = bool(command)

            for robot in self._robots.values():
                robot.switch_mode(subscribe=self._subscribe, publish=self._publish, teleoperate=self._teleoperate,
                                  command=self._command)

    def close(self):
        """
        Close the middleware.
        """
        pass

    def reset(self):
        """
        Reset the middleware.
        """
        pass

    def get_robot_middleware(self, robot_id):
        r"""
        Get the robot middleware associated with the given robot id.

        Args:
            robot_id (int): robot unique id.

        Returns:
            RobotMiddleware, None: robot middleware. None if it could not find the associated robot midddleware.
        """
        pass

    def load_urdf(self, urdf):
        """Load the given URDF file.

        The load_urdf will send a command to the physics server to load a physics model from a Universal Robot
        Description File (URDF). The URDF file is used by the ROS project (Robot Operating System) to describe robots
        and other objects, it was created by the WillowGarage and the Open Source Robotics Foundation (OSRF).
        Many robots have public URDF files, you can find a description and tutorial here:
        http://wiki.ros.org/urdf/Tutorials

        Args:
            urdf (str): a relative or absolute path to the URDF file on the file system of the physics server.

        Returns:
            int (non-negative): unique id associated to the load model.
        """
        pass

    def has_sensor(self, body_id, name):
        """
        Check if the specified robot has the given sensor.

        Args:
            body_id (int): body unique id.
            name (str): name of the sensor.

        Returns:
            bool: True if the specified robot has the given sensor.
        """
        pass

    def get_sensor_values(self, body_id, name):
        """
        Return the sensor.

        Args:
            body_id (int): body unique id.
            name (str): name of the sensor.

        Returns:
            np.array, dict, list, None: sensor values. None if it didn't have anything.
        """
        pass

    def reset_joint_states(self, body_id, joint_ids, positions, velocities=None):
        """
        Reset the joint states. It is best only to do this at the start, while not running the simulation:
        `reset_joint_state` overrides all physics simulation.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint indices where each joint index is between [0..num_joints(body_id)]
            positions (float, list[float], np.array[float]): the joint position(s) (angle in radians [rad] or
              position [m])
            velocities (float, list[float], np.array[float]): the joint velocity(ies) (angular [rad/s] or linear
              velocity [m/s])
        """
        pass

    def get_joint_positions(self, body_id, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        pass

    def set_joint_positions(self, body_id, joint_ids, positions, velocities=None, kps=None, kds=None, forces=None,
                            check_teleoperate=False):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            positions (float, np.array[float[N]]): desired position, or list of desired positions [rad]
            velocities (None, float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            kps (None, float, np.array[float[N]]): position gain(s)
            kds (None, float, np.array[float[N]]): velocity gain(s)
            forces (None, float, np.array[float[N]]): maximum motor force(s)/torque(s) used to reach the target values.
            check_teleoperate (bool): if True, it will check if the given `teleoperate` argument has been set to True,
              and if so, it will set the joint positions. If the `teleoperate` argument has been set to False, it
              won't set the joint positions.
        """
        pass

    def get_joint_velocities(self, body_id, joint_ids):
        """
        Get the velocity of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.array[float[N]]: joint velocities [rad/s]
        """
        pass

    def set_joint_velocities(self, body_id, joint_ids, velocities, max_force=None, check_teleoperate=False):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            velocities (float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            max_force (None, float, np.array[float[N]]): maximum motor forces/torques.
            check_teleoperate (bool): if True, it will check if the given `teleoperate` argument has been set to True,
              and if so, it will set the joint velocities. If the `teleoperate` argument has been set to False, it
              won't set the joint velocities.
        """
        pass

    def get_joint_torques(self, body_id, joint_ids):
        """
        Get the applied torque(s) on the given joint(s). "This is the motor torque applied during the last `step`.
        Note that this only applies in VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the
        applied joint motor torque is exactly what you provide, so there is no need to report it separately." [1]

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.array[float[N]]: torques associated to the given joints [Nm]
        """
        pass

    def set_joint_torques(self, body_id, joint_ids, torques, check_teleoperate=False):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            torques (float, list[float]): desired torque(s) to apply to the joint(s) [N].
            check_teleoperate (bool): if True, it will check if the given `teleoperate` argument has been set to True,
              and if so, it will set the joint torques. If the `teleoperate` argument has been set to False, it won't
              set the joint torques.
        """
        pass

    def get_jacobian(self, body_id, link_id, local_position=None, q=None):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q), J_{ang}(q)]^T`, such that:

        .. math:: v = [\dot{p}, \omega]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            body_id (int): unique body id.
            link_id (int): link id.
            local_position (np.array[float[3]]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
            q (np.array[float[N]]): joint positions of size N, where N is the number of DoFs.

        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: full geometric (linear and angular) Jacobian matrix. The
                number of columns depends if the base is fixed or floating.
        """
        pass

    def get_inertia_matrix(self, body_id, q):
        r"""
        Return the mass/inertia matrix :math:`H(q)`, which is used in the rigid-body equation of motion (EoM) in joint
        space given by (see [1]):

        .. math:: \tau = H(q)\ddot{q} + C(q,\dot{q})

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Warnings: If the base is floating, it will return a [6+N,6+N] inertia matrix, where N is the number of actuated
            joints. If the base is fixed, it will return a [N,N] inertia matrix

        Args:
            body_id (int): body unique id.
            q (np.array[float[N]]): joint positions of size N, where N is the total number of DoFs.

        Returns:
            np.array[float[N,N]], np.array[float[6+N,6+N]]: inertia matrix
        """
        pass
