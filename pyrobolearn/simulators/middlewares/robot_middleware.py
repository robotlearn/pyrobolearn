#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Robot middleware API.

The robot middleware interface is an interface between a particular robot and the middleware. The middleware
possesses a list of Robot middleware interfaces (one for each robot). If you have a specific robot, you have to
implement this class otherwise it will use the provided default one.

For instance, when using ROS, the `RobotMiddleWare` is inherited by the `ROSRobotMiddleware` class from which all
ROS robot middlewares have to inherit from. A `DefaultROSRobotMiddleware` that inherits from the `ROSRobotMiddleware`
is also provided.

Dependencies in PRL:
* `pyrobolearn.simulators.middlewares.robot_middleware.RobotMiddleWare`
"""

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotMiddleware(object):
    r"""Robot middleware interface.

    The robot middleware interface is an interface between a particular robot and the middleware. The middleware
    possesses a list of Robot middleware interfaces (one for each robot).

    Notably, the robot middleware has a unique id, has a list of publishers and subscribers associated with the given
    robot.

    Here are the possible combinations between the different values for subscribe (S), publish (P), teleoperate (T),
    and command (C):

    - S=1, P=0, T=0: subscribes to the topics, and get the messages when calling the corresponding getter methods.
    - S=0, P=1, T=0: publishes the various messages to the topics when calling the corresponding setter methods.
    - S=1, P=0, T=1, C=0/1: get messages by subscribing to the topics that publish some commands/states. The received
      commands/states are then set in the simulator. Depending on the value of `C`, it will subscribe to topics that
      publish some commands (C=1) or states (C=0). Example: should we subscribe to joint trajectory commands, or joint
      states when teleoperating the robot in the simulator? This C value allows to specify which one we are interested
      in.
    - S=0, P=1, T=1: when calling the getters methods such as joint positions, velocities, and others, it also
      publishes the joint states. This is useful if we are moving/teleoperating the robot in the simulator.
    - S=0, P=0, T=1/0: doesn't do anything.
    - S=1, P=1, T=0: subscribes to some topics and publish messages to other topics. The messages are can be
      sent/received by calling the appropriate getter/setter methods.
    - S=1, P=1, T=1: not allowed, because teleoperating the robot is not a two-way communication process.
    """

    def __init__(self, robot_id, urdf=None, subscribe=False, publish=False, teleoperate=False, command=True,
                 control_file=None):
        """
        Initialize the robot middleware interface.

        Args:
            robot_id (int): robot unique id.
            urdf (str): path to the URDF file.
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robot, and will read
              the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robot.
            teleoperate (bool): if True, it will move the robot based on the received or sent values based on the 2
              previous attributes :attr:`subscribe` and :attr:`publish`.
            command (bool): if True, it will subscribe to the joint commands. If False, it will subscribe to the
              joint states.
            control_file (str, None): path to the YAML control file.
        """
        # set variables
        self.id = robot_id
        self.urdf = urdf
        self.control_file = control_file

        self._subscribe, self._publish, self._teleoperate, self._command = False, False, False, False
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

    def __del__(self):
        """
        Close all topics.
        """
        self.close()

    ###########
    # Methods #
    ###########

    def unregister(self):
        """
        Unsubscribe from a topic. Topic instance is no longer valid after this call. Additional calls to `unregister()`
        have no effect.
        """
        pass

    def close(self):
        """
        Close all topics. Topic instances are no longer valid after this call.
        """
        self.unregister()

    def switch_mode(self, subscribe=None, publish=None, teleoperate=None, command=None):
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
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robots, and will read
              the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robots.
            teleoperate (bool): if True, it will move the robot based on the received or sent values based on the 2
              previous attributes :attr:`subscribe` and :attr:`publish`.
            command (bool): if True, it will subscribe to the joint commands. If False, it will subscribe to the
              joint states.
        """
        if subscribe is None:
            subscribe = self.subscribe
        if publish is None:
            publish = self.publish
        if teleoperate is None:
            teleoperate = self.teleoperate
        if command is None:
            command = self.command

        if teleoperate and publish and subscribe:
            raise ValueError("The three following arguments 'subscribe', 'publish', and 'teleoperate' can not be all "
                             "true at the same time. Select maximum two to be set to True (see method documentation).")

        self._subscribe = bool(subscribe)
        self._publish = bool(publish)
        self._teleoperate = bool(teleoperate)
        self._command = bool(command)

    def reset_joint_states(self, positions, joint_ids=None, velocities=None):
        """
        Reset the joint states. It is best only to do this at the start, while not running the simulation:
        `reset_joint_state` overrides all physics simulation.

        Args:
            positions (float, list[float], np.array[float]): the joint position(s) (angle in radians [rad] or
              position [m])
            joint_ids (int, list[int]): joint indices where each joint index is between [0..num_joints(body_id)]
            velocities (float, list[float], np.array[float]): the joint velocity(ies) (angular [rad/s] or linear
              velocity [m/s])
        """
        pass

    def get_joint_positions(self, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        pass

    def set_joint_positions(self, positions, joint_ids=None, velocities=None, kps=None, kds=None, forces=None):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            positions (float, np.array[float[N]]): desired position, or list of desired positions [rad]
            joint_ids (int, list[int], None): joint id, or list of joint ids.
            velocities (None, float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            kps (None, float, np.array[float[N]]): position gain(s)
            kds (None, float, np.array[float[N]]): velocity gain(s)
            forces (None, float, np.array[float[N]]): maximum motor force(s)/torque(s) used to reach the target values.
        """
        pass

    def get_joint_velocities(self, joint_ids):
        """
        Get the velocity of the given joint(s).

        Args:
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.array[float[N]]: joint velocities [rad/s]
        """
        pass

    def set_joint_velocities(self, velocities, joint_ids=None, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            velocities (float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            joint_ids (int, list[int], None): joint id, or list of joint ids.
            max_force (None, float, np.array[float[N]]): maximum motor forces/torques.
        """
        pass

    def get_joint_torques(self, joint_ids):
        """
        Get the applied torque(s) on the given joint(s). "This is the motor torque applied during the last `step`.
        Note that this only applies in VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the
        applied joint motor torque is exactly what you provide, so there is no need to report it separately." [1]

        Args:
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.array[float[N]]: torques associated to the given joints [Nm]
        """
        pass

    def set_joint_torques(self, torques, joint_ids=None):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            torques (float, list[float]): desired torque(s) to apply to the joint(s) [N].
            joint_ids (int, list[int], None): joint id, or list of joint ids.
        """
        pass

    def has_sensor(self, name):
        """
        Check if the given robot middleware has the specified sensor.

        Args:
            name (str): name of the sensor.

        Returns:
            bool: True if the robot middleware has the sensor.
        """
        pass

    def get_sensor_values(self, name):
        """
        Get the sensor values associated with the given sensor name.

        Args:
            name (str): unique name of the sensor.

        Returns:
            object, np.array, float, int: sensor values.
        """
        pass

    def get_pid(self, joint_ids):
        """
        Get the PID coefficients associated to the given joint ids.

        Args:
            joint_ids (list[int]): list of unique joint ids.

        Returns:
            list[np.array[float[3]]]: list of PID coefficients for each joint.
        """
        pass

    def set_pid(self, joint_ids, pid):
        """
        Set the given PID coefficients to the given joint ids.

        Args:
            joint_ids (list[int]): list of unique joint ids.
            pid (list[np.array[float[3]]]): list of PID coefficients for each joint. If one of the value is -1, it
              will left untouched the associated PID value to the previous one.
        """
        pass

    def get_jacobian(self, link_id, local_position=None, q=None):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q), J_{ang}(q)]^T`, such that:

        .. math:: v = [\dot{p}, \omega]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            link_id (int): link id.
            local_position (None, np.array[float[3]]): the point on the specified link to compute the Jacobian (in link
              local coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
            q (np.array[float[N]], None): joint positions of size N, where N is the number of DoFs. If None, it will
              compute q based on the current joint positions.

        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: full geometric (linear and angular) Jacobian matrix. The
              number of columns depends if the base is fixed or floating.
        """
        pass

    def get_inertia_matrix(self, q=None):
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
            q (np.array[float[N]], None): joint positions of size N, where N is the total number of DoFs. If None, it
              will get the current joint positions.

        Returns:
            np.array[float[N,N]], np.array[float[6+N,6+N]]: inertia matrix
        """
        pass
