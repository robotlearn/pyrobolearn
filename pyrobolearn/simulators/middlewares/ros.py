#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the ROS middleware API.

ROS (Robot Operating System) [1] is a robotics middleware which "provides libraries and tools to help software
developers create robot applications. It provides hardware abstraction, device drivers, libraries, visualizers,
message-passing, package management, and more."

- rospy [2]: "rospy is a pure Python client library for ROS. The rospy client API enables Python programmers to
  quickly interface with ROS Topics, Services, and Parameters."
- ros_control [3]: "A set of packages that include controller interfaces, controller managers, transmissions and
  hardware_interfaces."
- robot_state_publisher [4]: "This package allows you to publish the state of a robot to tf. Once the state gets
  published, it is available to all components in the system that also use tf. The package takes the joint angles of
  the robot as input and publishes the 3D poses of the robot links, using a kinematic tree model of the robot. The
  package can both be used as a library and as a ROS node."
- joint_state_publisher [5]: "This package contains a tool for setting and publishing joint state values for a given
  URDF."

Note to compile ROS packages using `catkin_make`, you might have to specify the Python version used using:
- catkin_make -DPYTHON_EXECUTABLE=path/to/bin/python3

Dependencies in PRL:
* `pyrobolearn.simulators.middlewares.middleware.Middleware`

References:
    - [1] ROS: http://www.ros.org/  and  http://wiki.ros.org
    - [2] rospy: http://wiki.ros.org/rospy
    - [3] ros_control: http://wiki.ros.org/ros_control
        - ROS control an overview: https://roscon.ros.org/2014/wp-content/uploads/2014/07/ros_control_an_overview.pdf
    - [4] robot_state_publisher: http://wiki.ros.org/robot_state_publisher
    - [5] joint_state_publisher: http://wiki.ros.org/joint_state_publisher
    - [6] roslaunch: http://wiki.ros.org/roslaunch/API%20Usage
"""

# TODO
import os
import subprocess
import psutil
import signal
import importlib
import inspect
import yaml
import numpy as np

import rospy
import roslaunch
import rosparam
import rosmsg, rosservice
import rostopic
import controller_manager.controller_manager_interface as cm_interface

import std_msgs.msg as std_msg
import sensor_msgs.msg as sensor_msg
import gazebo_msgs.msg as gazebo_msg
import geometry_msgs.msg as geometry_msg
import trajectory_msgs.msg as trajectory_msg

from pyrobolearn.simulators.middlewares.middleware import Middleware
from pyrobolearn.simulators.middlewares.robot_middleware import RobotMiddleware
from pyrobolearn.simulators.middlewares.ros_publisher import PublisherData, Publisher, RobotPublisher
from pyrobolearn.simulators.middlewares.ros_subscriber import SubscriberData, Subscriber, RobotSubscriber

from pyrobolearn.utils.parsers.robots.urdf_parser import URDFParser


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["ROS (Willow Garage)", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Remapper(object):
    """Remapper from old topic to a new topic.

    Note that this doesn't replace the old topic by the new one; the old topic will still be present.
    """

    def __init__(self, old_topic, new_topic, msg_class, queue_size=10, new_msg_class=None, function=None):
        """
        Initialize the Remapper that subscribes to the given topic.

        Args:
            old_topic (str, list[str]): old topic name(s). If multiple topics are provided, it will group them. Note
              that you can only group topics that use the same message class.
            new_topic (str, list[str]): new topic name(s). If multiple topics are provided, it will group them and send
              to each of the new topic the (converted) data.
            msg_class (class): message class for serialization. If `new_msg_class` is provided, then `msg_class`
              represents the message class used in the old topic(s).
            queue_size (int): The queue size used for asynchronously publishing messages from different threads. A
              size of zero means an infinite queue, which can be dangerous. When None is passed all publishing will
              happen synchronously and a warning message will be printed.
            new_msg_class (class, None): optional new message class for serialization. This is the message associated
              with the new topic(s).
            function (callable, None): optional function that should accept the old message object and process it
              and/or convert it into the new one.
        """
        rospy.init_node("remapper", anonymous=True)

        # create subscriber and publisher
        self.old_topic = old_topic
        self.new_topic = new_topic
        self.subscriber = rospy.Subscriber(old_topic, msg_class, callback=self.callback)
        if new_msg_class is None:
            new_msg_class = msg_class
        self.publisher = rospy.Publisher(new_topic, new_msg_class, queue_size=queue_size)
        # self.msg = msg_class()

        # check processing function
        if function is not None and not callable(function):
            raise TypeError("Expecting the given 'function' to be callable...")
        self.function = function
        # TODO: should I test the function to make sure it returns the correct object type?

    def callback(self, data):
        """
        Callback function that publishes the received the data from the old topic to the new topic.

        Args:
            data (object, list[object]): message class instance.
        """
        if self.function is not None:
            data = self.function(data)
        self.publisher.publish(data)

    def unregister(self):
        """
        Unsubscribe from a topic. Topic instance is no longer valid after this call. Additional calls to `unregister()`
        have no effect.
        """
        self.subscriber.unregister()
        self.publisher.unregister()

    def close(self):
        """
        Close all topics.
        """
        self.unregister()

    def __del__(self):
        """
        Close all topics.
        """
        self.unregister()


class ROSRobotMiddleware(RobotMiddleware):
    r"""ROS robot middleware interface.

    The robot middleware interface is an interface between a particular robot and the middleware. The middleware
    possesses a list of Robot middleware interfaces (one for each robot).

    Notably, the ROS robot middleware has a unique id, has a list of publishers and subscribers associated with the
    given robot.

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
                 control_file=None, launch_file=None, joint_state_topics=None, joint_state_msg_class=None):
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
            control_file (str, None): path to the YAML control file. If provided, it will be parsed.
            launch_file (str, None): path to the ROS launch file. If provided, it will be parsed.
            joint_state_topics (str, list[str]): joint state topic(s). If not provided the joint state topic will be
              set to '/<robot_name>/joint_states'.
            joint_state_msg_class (class): message serialization class used for the provided joint state topic. By
              default, it will be set to 'sensor_msg.JointState'.
        """
        super(ROSRobotMiddleware, self).__init__(robot_id, urdf, subscribe, publish, teleoperate, command, control_file)
        self.launch_file = launch_file

        print("\n Creating ROSRobotMiddleware")

        # get path to the URDF folder
        path = os.path.abspath(urdf)  # /path/to/pyrobolearn/robots/urdfs/<robot>/robot.urdf
        # dirname = str(os.path.dirname(path))  # /path/to/pyrobolearn/robots/urdfs/<robot>/
        basename = str(os.path.basename(path).split('.')[-2])  # robot name without extension

        # parse URDF to get joint names, q indices, etc.
        self.urdf_parser = URDFParser()
        tree = self.urdf_parser.parse(urdf)
        print("ROSRobotMiddleware - publisher - name: ", tree.name)
        print("Num joints: ", tree.num_joints)
        print("Num actuated joints: ", tree.num_actuated_joints)
        self.tree = tree
        self.q_indices = np.zeros(tree.num_joints, dtype=int)
        self.joint_names = []
        count = 0
        for i, joint in enumerate(tree.joints.values()):
            if joint.dtype != 'fixed':
                print("Adding joint {} with type={}, q_idx={}".format(joint.name, joint.dtype, i))
                self.q_indices[i] = count
                self.joint_names.append(joint.name)
                count += 1

        # subscriber and publisher associated with the given robot
        # if self.is_subscribing:
        print("Creating Robot Subscriber")
        self.subscriber = RobotSubscriber(name=basename, joint_state_topics=joint_state_topics,
                                          joint_state_msg_class=joint_state_msg_class)
        # if self.is_publishing:
        print("Creating Robot Publisher")
        self.publisher = RobotPublisher(name=basename)

    def unregister(self):
        """
        Unsubscribe from a topic. Topic instance is no longer valid after this call. Additional calls to `unregister()`
        have no effect.
        """
        if self.is_subscribing:
            self.subscriber.unregister()
        if self.is_publishing:
            self.publisher.unregister()

    def create_publisher(self, name, topic, msg_class, queue_size=10):
        """
        Create a publisher to the specific topic. If the publisher already exists, it unregister the previous one
        and replace it by the new one.

        Args:
            name (str): unique name of the publisher. The name must be unique. You will be able to access to this
              publisher using its name.
            topic (str, list[str]): name of the topic(s).
            msg_class (object): message class serialization.
            queue_size (int): The queue size used for asynchronously publishing messages from different threads. A
              size of zero means an infinite queue, which can be dangerous. When None is passed all publishing will
              happen synchronously and a warning message will be printed.

        Returns:
            PublisherData: the publisher data holder.
        """
        return self.publisher.create_publisher(name=name, topic=topic, msg_class=msg_class, queue_size=queue_size)

    def create_subscriber(self, name, topic, msg_class):
        """
        Create a subscriber to the specific topic. If the subscriber already exists, it unregister the previous one
        and replace it by the new one.

        Args:
            name (str): unique name of the subscriber. The name must be unique. You will be able to access to this
              subscriber using its name.
            topic (str, list[str]): name of the topic(s).
            msg_class (object): message class serialization.

        Returns:
            SubscriberData: the subscriber data holder.
        """
        return self.subscriber.create_subscriber(name=name, topic=topic, msg_class=msg_class)

    def change_topic(self, old_topic, new_topic, new_msg=None, queue_size=None):
        """
        Change a publisher's or subscriber's topic name to a new one with possibly a new message class and queue size.

        Args:
            old_topic (str): old topic name.
            new_topic (str): new topic name.
            new_msg (object): message class serialization. If None, it will use the same message class than the old
              topic.
            queue_size (int): The queue size used for asynchronously publishing messages from different threads. A
              size of zero means an infinite queue, which can be dangerous. If None, it will use the same queue size
              than the old topic.

        Returns:
            PublisherData, SubscriberData: the publisher data holder.
        """
        if self.publisher.has_topic(old_topic):
            self.publisher.change_topic(old_topic=old_topic, new_topic=new_topic, new_msg=new_msg,
                                        queue_size=queue_size)
        if self.subscriber.has_subscriber(old_topic):
            self.subscriber.change_topic(old_topic=old_topic, new_topic=new_topic, new_msg=new_msg)


class DefaultROSRobotMiddleware(ROSRobotMiddleware):
    r"""Default ROS robot middleware interface.

    This is the default ROS robot middleware interface which can be created when no specific interfaces are provided
    by the user. Specific interfaces can be found in the `robots` folder.

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

    def __init__(self, robot_id, urdf=None, subscribe=False, publish=False, teleoperate=False, command=False,
                 control_file=None, launch_file=None):
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
            launch_file (str, None): path to the ROS launch file. If provided, it will be parsed.
        """
        # set variables
        super(DefaultROSRobotMiddleware, self).__init__(robot_id, urdf=urdf, subscribe=subscribe, publish=publish,
                                                        teleoperate=teleoperate, command=command,
                                                        control_file=control_file, launch_file=launch_file)

        if self.is_publishing:
            topics = []
            count = 0
            for i, joint in enumerate(self.tree.joints.values()):
                if joint.dtype != 'fixed':
                    topic = '/' + self.tree.name + '/joint' + str(count+1) + '_position_controller/command'
                    topics.append(topic)
            print("Publishing Topics: ", topics)
            publisher = self.publisher.create_publisher(name='qpos', topic=topics, msg_class=std_msg.Float64)
            self.publisher.init_set_joint_positions(publisher=publisher, msg_attribute_name='data')

        # sensors

    def get_joint_positions(self, joint_ids=None):
        """
        Get the position of the given joint(s).

        Args:
            joint_ids (int, list[int], None): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        if self.is_subscribing:
            q_indices = None if joint_ids is None else self.q_indices[joint_ids]
            return self.subscriber.get_joint_positions(q_indices)

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
        if self.is_publishing:
            q_indices = None if joint_ids is None else self.q_indices[joint_ids]
            self.publisher.set_joint_positions(positions, q_indices=q_indices)
            self.publisher.publish('qpos')

    def get_joint_velocities(self, joint_ids=None):
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
        if self.is_subscribing:
            q_indices = None if joint_ids is None else self.q_indices[joint_ids]
            return self.subscriber.get_joint_velocities(q_indices)

    def set_joint_velocities(self, velocities, joint_ids=None, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            velocities (float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            joint_ids (int, list[int], None): joint id, or list of joint ids.
            max_force (None, float, np.array[float[N]]): maximum motor forces/torques.
        """
        if self.is_publishing:
            q_indices = None if joint_ids is None else self.q_indices[joint_ids]
            self.publisher.set_joint_velocities(velocities, q_indices=q_indices)
            self.publisher.publish('qvel')

    def get_joint_torques(self, joint_ids=None):
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
        if self.is_subscribing:
            q_indices = None if joint_ids is None else self.q_indices[joint_ids]
            return self.subscriber.get_joint_torques(q_indices)

    def set_joint_torques(self, torques, joint_ids=None):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            torques (float, list[float]): desired torque(s) to apply to the joint(s) [N].
            joint_ids (int, list[int], None): joint id, or list of joint ids.
        """
        if self.is_publishing:
            q_indices = None if joint_ids is None else self.q_indices[joint_ids]
            self.publisher.set_joint_velocities(torques, q_indices=q_indices)
            self.publisher.publish('torques')

    def has_sensor(self, name):
        """
        Check if the given robot middleware has the specified sensor.

        Args:
            name (str): name of the sensor.

        Returns:
            bool: True if the robot middleware has the sensor.
        """
        if self.is_subscribing:
            return self.subscriber.has_subscriber(name)
        return False

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


class ROS(Middleware):
    r"""ROS Interface middleware

    This middleware class can be given to the simulator which can then interact with the various robots, sensors, and
    actuators through the ROS middleware. Not only that, it can also be used to create ROS nodes, publishers,
    subscribers, topics, and services. Users can then use this middleware interface to interact with these last ones.

    Basically, it is the main ROS API that is used in PyRoboLearn (PRL) to access to the various ROS components
    including ROS software/nodes implemented by the robotics community.
    For instance, if given to a `Simulator` instance (defined in PRL), every time a robot is loaded in simulation
    using the `load_urdf` method, this middleware can load the corresponding publishers, subscribers and plugins like
    controllers (defined in `ros_control`), and interact with these ones when calling the relevant methods defined
    in the `Simulator` instance.

    Examples:

        import pyrobolearn as prl

        # create middleware instance
        ros = prl.middlewares.ROS(...)

        # create simulator instance and gives the middleware as argument
        sim = prl.simulators.Bullet(render=True, middleware=ros)

        # load a robot in the simulator.
        robot = prl.robots.<RobotName>(sim)

        # the following command might, depending on the arguments provided to the middleware, return the joint
        # positions published on a specific ROS topic.
        print(robot.get_joint_positions())

        # under the hood, `robot.get_joint_position`, will ask the simulator to return the joint positions related to
        # the specified robot, which will in turn ask the middleware to return the joint positions if possible.

        # Note that can also use the middleware alone to access to the various ROS nodes, publishers, subscribers.
    """

    def __init__(self, subscribe=False, publish=False, teleoperate=False, command=True, master_uri=11311,
                 init_core=True, **kwargs):
        """
        Initialize the ROS middleware.

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
            command (bool): if True, it will subscribe/publish to some (joint) commands. If False, it will
              subscribe/publish to some (joint) states.
            master_uri (int): ROS master URI.
            init_core (bool): initialize the ROS core if specified.
        """
        super(ROS, self).__init__(subscribe=subscribe, publish=publish, teleoperate=teleoperate, command=command)

        # Environment variable
        self.env = os.environ.copy()

        # run ROS core if not already running
        self.core = None  # = roscore
        self.gui = None   # = RQT
        self.init_core(port=master_uri)

        # remember each publisher/subscriber
        self.subscribers = {}
        self.publishers = {}
        self.remappers = {}
        self.models = []

        # self._robots = {}  # {body_id: ROSRobotMiddleware}
        self.count_id = -1

        # init roslaunch
        # From doc: "the ROSLaunchParent represents the main 'parent' roslaunch process. It is responsible for
        # loading the launch files, assigning machines, and then starting up any remote processes. The __main__ method
        # delegates most of runtime to ROSLaunchParent."
        self.launch = self.init_launch_process(start_process=True)

        self.processes = {}  # {Node: process}

        rospy.init_node(self.__class__.__name__, anonymous=True)

    ###########
    # Methods #
    ###########

    def close(self):
        """
        Close everything.
        """
        # delete each robot middleware
        for robot in self._robots.values():
            robot.close()

        # delete each subscriber
        for subscriber in self.subscribers.values():
            subscriber.close()

        # delete each publisher
        for publisher in self.publishers.values():
            publisher.close()

        # delete each process
        for process in self.processes.values():
            process.stop()

        # stop launch
        if self.launch:
            self.launch.stop()

        # terminate ROS core process
        if self.core is not None:
            os.killpg(os.getpgid(self.core.pid), signal.SIGTERM)

        # terminate RQT process
        if self.gui is not None:
            os.killpg(os.getpgid(self.core.pid), signal.SIGTERM)

    def reset(self):
        """
        Reset the middleware.
        """
        pass

    ############
    # ROS core #
    ############

    @staticmethod
    def is_core_running():
        """Return True if the ROS core is running."""
        try:
            rostopic.get_topic_class('/rosout')
        except rostopic.ROSTopicIOException as e:
            return False
        return True

    def init_core(self, uri='localhost', port=11311):
        """Initialize the core if it is not running.

        From [1], "the ROS core will start up:
        - a ROS master
        - a ROS parameter server
        - a rosout logging node"

        Args:
            uri (str): ROS master URI. The ROS_MASTER_URI will be set to `http://<uri>:<port>/`.
            port (int): Port to run the master on.

        References:
            - [1] roscore: http://wiki.ros.org/roscore
        """
        # if the core is not already running, run it
        if not self.is_core_running():
            # Environment variable
            self.env["ROS_MASTER_URI"] = "http://" + uri + ":" + str(port)

            # this is for the rospy methods such as: wait_for_service(), init_node(), ...
            os.environ['ROS_MASTER_URI'] = self.env['ROS_MASTER_URI']

            # run ROS core if not already running
            # if "roscore" not in [p.name() for p in psutil.process_iter()]:
            # subprocess.Popen("roscore", env=self.env)
            self.core = subprocess.Popen(["roscore", "-p", str(port)], env=self.env,
                                         preexec_fn=os.setsid)  # , shell=True)

    ###################
    # ROS launch/node #
    ###################

    def init_launch_process(self, start_process=True):
        """
        Initialize the roslaunch process.

        From the official documentation, the "ROSLaunchParent represents the main 'parent' roslaunch process. It is
        responsible for loading the launch files, assigning machines, and then starting up any remote processes.
        The __main__ method delegates most of runtime to ROSLaunchParent."

        Args:
            start_process (bool): if we should start the process or not. If False, this is let to the user to start it.

        Raises:
            RLException: if it fails to initialize.

        Returns:
            roslaunch.scriptapi.ROSLaunch: ROS launch process.
        """
        if not self.is_core_running():
            return roslaunch.core.RLException("ROS core has to be run before calling this method. See `init_core` "
                                              "method.")
        launch = roslaunch.scriptapi.ROSLaunch()
        if start_process:
            launch.start()
        return launch

    def load_launch_file(self, filename):
        """
        Load the given roslaunch file.

        Args:
            filename (str): roslaunch filename.
        """
        self.launch.load(filename)

    def load_launch_str(self, string):
        """
        Load the given roslaunch string.

        Args:
            string (str): string representation of roslaunch config.
        """
        self.launch.load_str(string)

    @staticmethod
    def init_node(name, argv=None, anonymous=False):
        rospy.init_node(name, argv=argv, anonymous=anonymous)

    @staticmethod
    def create_node(package, node_type, name=None, namespace='/', machine_name=None, args='', respawn=False,
                    respawn_delay=0.0, remap_args=None, env_args=None, output=None, cwd=None, launch_prefix=None,
                    required=False, filename='<unknown>'):
        """
        Create a ROS node; data structure for storing information about a desired node in the ROS system Corresponds
        to the 'node' tag in the launch specification.

        This basically wraps the `roslaunch.core.Node(...)`.

        Args:
            package (str): node package name.
            node_type (str): node type.
            name (str): node name.
            namespace (str): namespace for node.
            machine_name (str): name of machine to run node on.
            args (str): argument string to pass to node executable.
            respawn (bool): if True, respawn node if it dies.
            respawn_delay (float): if respawn is True, respawn node after delay.
            remap_args (list[tuple[str,str]]): list of [(from, to)] remapping arguments.
            env_args (list[tuple[str,str]]): list of [(key, value)] of additional environment vars to set for node.
            output (str): where to log output to, either Node, 'screen' or 'log'.
            cwd (str): current working directory of node, either 'node', 'ROS_HOME'. Default: ROS_HOME.
            launch_prefix (str): launch command/arguments to prepend to node executable arguments.
            required (bool): node is required to stay running (launch fails if node dies).
            filename (str): name of file Node was parsed from.

        Raises:
            ValueError: if parameters do not validate.

        Returns:
            roslaunch.core.Node: ROS node data structure.
        """
        return roslaunch.core.Node(package=package, node_type=node_type, name=name, namespace=namespace,
                                   machine_name=machine_name, args=args, respawn=respawn, respawn_delay=respawn_delay,
                                   remap_args=remap_args, env_args=env_args, output=output, cwd=cwd,
                                   launch_prefix=launch_prefix, required=required, filename=filename)

    def get_node(self, node_id):
        """
        Get the ROS node.

        Args:
            node_id (int): ROS node id.

        Returns:
            roslaunch.core.Node: ROS node data structure associated with the given id.
        """
        pass

    def launch(self, ros_node):
        """
        Launch a ROS node.

        Args:
            ros_node (roslaunch.core.Node): ROS node.

        Returns:
            roslaunch.nodeprocess.LocalProcess: the process.
        """
        process = self.launch.launch(ros_node)
        self.processes[ros_node] = process
        return process

    @staticmethod
    def get_namespace():
        """
        Get the namespace.

        Returns:
            str: namespace.
        """
        return rospy.get_namespace()

    ###########################
    # ROS parameters (+ YAML) #
    ###########################

    @staticmethod
    def load_parameter_file(filename, namespace=None):
        """
        Load a parameter YAML file.

        Args:
            filename (str): path to the YAML file.
            namespace (str): default namespace.

        Returns:
            list[dict[str,dict[str:dict[str:dict]]], str]: [{robot_name: {param_name: value}}, namespace]
        """
        return rosparam.load_file(filename, default_namespace=namespace)

    @staticmethod
    def load_parameter_string(parameters, namespace=None):
        """
        Load a control configuration YAML string.

        Args:
            parameters (str): string in the YAML format specifying the control parameters.
            namespace (str): default namespace.

        Returns:
            list[dict[str,dict[str:dict[str:dict]]], str]: [{robot_name: {param_name: value}}, namespace]
        """
        return rosparam.load_str(parameters, default_namespace=namespace)

    @staticmethod
    def set_parameter(name, value):
        """
        Set a parameter on the param server.

        Args:
            name (str): name of the parameter.
            value (str, dict): parameter value. "If param_value is a dictionary it will be treated as a parameter
              tree, where param_name is the namespace. For example::: {'x':1,'y':2,'sub':{'z':3}} will set
              `name/x=1`, `name/y=2`, and `name/sub/z=3`. Furthermore, it will replace all existing parameters in
              the `name` namespace with the parameters in `value`. You must set parameters individually if you wish
              to perform a union update.
        """
        rospy.set_param(name, value)

    @staticmethod
    def get_parameter(name, default=None):
        r"""
        Get the parameter associated with the given name from the param server.

        Args:
            name (str): name of the parameter.
            default (object): default value to return if the parameter is not found.

        Returns:
            str: parameter value
        """
        return rospy.get_param(name, default=default)

    @staticmethod
    def get_parameter_names():
        """
        Return the parameter names that have been loaded.

        Returns:
            list[str]: list of parameter names.
        """
        return rospy.get_param_names()

    @staticmethod
    def upload_parameters(values, namespace='/'):
        """
        Upload parameters to the Parameter Server.

        Args:
            values (dict): dictionary where keys are parameter names and values are parameter values.
            namespace (str): namespace to load parameters.
        """
        rosparam.upload_params(ns=namespace, values=values)

    @staticmethod
    def load_config_file(filename, namespace=None):
        """
        Load a control configuration YAML file.

        Args:
            filename (str): path to the YAML file.
            namespace (str): default namespace.

        Returns:
            list[dict[str,dict[str:dict[str:dict]]], str]: [{robot_name: {param_name: value}}, namespace]
        """
        params = rosparam.load_file(filename, default_namespace=namespace)
        params = params[0]
        namespace = params[1] + 'rrbot/'  # TODO: replace rrbot
        params = params[0]['rrbot']
        for key, value in params.items():
            rosparam.upload_params(ns=namespace + key, values=value)

    @staticmethod
    def load_config_string(parameters, namespace=None):
        """
        Load a control configuration YAML string.

        Args:
            parameters (str): string in the YAML format specifying the control parameters.
            namespace (str): default namespace.

        Returns:
            list[list[str,str]]:
        """
        params = rosparam.load_str(parameters, default_namespace=namespace)

    #########################
    # ROS Messages/Services #
    #########################

    #######################
    # ROS Topics/Services #
    #######################

    def remap_topic(self, old_topic, new_topic, msg_class, queue_size=10):
        """
        Remap a old topic to a new topic. Note that this doesn't replace the old topic by the new one; the old topic
        will still be present.

        Args:
            old_topic (str): name of the old topic.
            new_topic (str): name of the new topic.
            msg_class (class): message class for serialization.
            queue_size (int): The queue size used for asynchronously publishing messages from different threads. A
              size of zero means an infinite queue, which can be dangerous. When None is passed all publishing will
              happen synchronously and a warning message will be printed.

        Returns:
            Remapper: remapper subscribe and publish node.
        """
        remapper = Remapper(old_topic, new_topic, msg_class, queue_size=queue_size)
        self.remappers[len(self.remappers)] = remapper
        return remapper

    @staticmethod
    def get_topics(namespace='/'):
        """
        Get the published topics.

        Returns:
            list[list[str, str]]: list of tuples where the first is the topic and the second element is the message
              type that topic accepts.
        """
        return rospy.get_published_topics(namespace=namespace)

    @staticmethod
    def get_services(namespace=None):
        """
        Get the list of services.

        Returns:
            list[str]: list of services.
        """
        return rosservice.get_service_list(namespace=namespace)

    @staticmethod
    def has_topic(name, namespace='/'):
        """
        Check if the specified topic has been advertised.

        Warnings: this has a O(N) time complexity.

        Args:
            name (str): name of the topic.
            namespace (str): namespace.

        Returns:
            bool: True if the topic has been advertised.
        """
        topics = rospy.get_published_topics(namespace=namespace)
        for topic_name, dtype in topics:
            if topic_name == name:  # or topic_name.split('/')[-1] == name:
                return True
        return False

    @staticmethod
    def get_topic_type(name):
        """
        Get the topic type name of the given topic name.

        This is the same as typing the following in the terminal:
        - `rostopic type /topic_name`

        Args:
            name (str): name of the topic.

        Returns:
            str, None: type of the topic.
        """
        return rostopic.get_topic_type(name)[0]

    @staticmethod
    def get_topic_class(name):
        """
        Get the topic class type of the given topic name.

        Args:
            name (str): name of the topic.

        Returns:
            type: topic class type. This can later be instantiated.
        """
        return rostopic.get_topic_class(name)

    @staticmethod
    def get_service_type(name):
        """
        Get the service class type of the given service name.

        Args:
            name (str): name of hte service.

        Returns:
            str, None: type of the service.
        """
        return rosservice.get_service_type(name)

    @staticmethod
    def get_service_class(name):
        """
        Get the service class type of the given service name.

        Args:
            name (str): name of the service.

        Returns:
            type: service class type. This can later be instantiated.
        """
        return rosservice.get_service_class_by_name(name)

    @staticmethod
    def get_service_args(name):
        """
        Get the service arguments of the given service name.

        Args:
            name (str): name of the service.

        Returns:
            list: service arguments.
        """
        return rosservice.get_service_args(name)

    ##############################
    # ROS Publishers/Subscribers #
    ##############################

    def create_publisher(self, body_id, topic_name, freq, fct, attribute):
        """
        Create a publisher for the given body id.

        Args:
            body_id (int): unique body id.
            topic_name (str): name of the topic to publish to.
            freq (float): update frequency rate.
            fct (callable): function to call each time we set the data.
            attribute (list[object]): attributes to the callable fct.

        Returns:
            PublisherData: publisher.
        """
        pass

    def create_subscriber(self, body_id, topic_name, freq, fct, attribute):
        """
        Create a subscriber for the given body id.

        Args:
            body_id (int): unique body id.
            topic_name (str): name of the topic to subscribe to.
            freq (float): update frequency rate.
            fct (callable): callback function to be called each time we receive a message.
            attribute (list[object]): attributes to the callable fct.

        Returns:
            SubscriberData: subscriber.
        """
        pass

    def add_publisher(self, publisher):
        """
        Add a publisher.

        Args:
            publisher (rospy.topics.Publisher, PublisherData): publisher.

        Returns:
            int: unique publisher id.
        """
        pass

    def add_subscriber(self, subscriber):
        """
        Add a subscriber.

        Args:
            subscriber (rospy.topics.Subscriber, SubscriberData): subscriber.

        Returns:
            int: unique subscriber id.
        """
        pass

    def get_publisher(self, publisher_id):
        """
        Get the publisher corresponding to the given publisher id.

        Args:
            publisher_id (int): unique publisher id.

        Returns:
            rospy.topics.Publisher, PublisherData: corresponding publisher to the given id.
        """
        pass

    def get_subscriber(self, subscriber_id):
        """
        Get the subscriber corresponding to the given subscriber id.

        Args:
            subscriber_id (int): unique subscriber id.

        Returns:
            rospy.topics.Subscriber, SubscriberData: corresponding subscriber to the given id.
        """
        pass

    def remove_publisher(self, publisher_id):
        """
        Remove the publisher corresponding to the given publisher id.

        Args:
            publisher_id (int): unique publisher id.
        """
        pass

    def remove_subscriber(self, subscriber_id):
        """
        Remove the subscriber corresponding to the given subscriber id.

        Args:
            subscriber_id (int): unique subscriber id.
        """
        pass

    ###############
    # ROS CONTROL #
    ###############

    def get_pid(self, body_id, joint_ids):
        """
        Get the PID coefficients associated to the given body id and joint ids.

        Args:
            body_id (int): unique body id.
            joint_ids (list[int]): list of unique joint ids.

        Returns:
            list[np.array[float[3]]]: list of PID coefficients for each joint.
        """
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.get_pid(joint_ids)

    def set_pid(self, body_id, joint_ids, pid):
        """
        Set the given PID coefficients to the given body id and joint ids.

        Args:
            body_id (int): unique body id.
            joint_ids (list[int]): list of unique joint ids.
            pid (list[np.array[float[3]]]): list of PID coefficients for each joint. If one of the value is -1, it
              will left untouched the associated PID value to the previous one.
        """
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.set_pid(joint_ids, pid)

    @staticmethod
    def load_controller(name):
        """
        Load controller based on the given name.

        Args:
            name (str): name of the controller to load.

        Returns:
            bool: True if we could load the controller.
        """
        cm_interface.load_controller(name)

    @staticmethod
    def unload_controller(name):
        """
        Unload the controller based on the given name.

        Args:
            name (str): name of the controller to unload.

        Returns:
            bool: True if we could unload the controller.
        """
        cm_interface.unload_controller(name)

    @staticmethod
    def start_controller(name):
        """
        Start the controller based on the given name.

        Args:
            name (str): name of the controller to start.

        Returns:
            bool: True if we could start the controller.
        """
        cm_interface.start_controller(name)

    @staticmethod
    def stop_controller(name):
        """
        Stop the controller based on the given name.

        Args:
            name (str): name of the controller to stop.

        Returns:
            bool: True if we could stop the controller.
        """
        return cm_interface.stop_controller(name)

    @staticmethod
    def get_controllers():
        """
        Return the list of controllers.
        """
        rospy.wait_for_service('controller_manager/list_controllers')
        s = rospy.ServiceProxy('controller_manager/list_controllers', cm_interface.ListControllers)
        resp = s.call(cm_interface.ListControllersRequest())
        return [(c.name, list(set(r.hardware_interface for r in c.claimed_resources)), c.state)
                for c in resp.controller]

    @staticmethod
    def get_controller_types():
        """
        Return the list of controller types.
        """
        rospy.wait_for_service('controller_manager/list_controller_types')
        s = rospy.ServiceProxy('controller_manager/list_controller_types', cm_interface.ListControllerTypes)
        resp = s.call(cm_interface.ListControllerTypesRequest())
        return [t for t in resp.types]

    @staticmethod
    def is_controller_manager_loaded():
        """
        Check if the controller manager is loaded.
        """
        try:
            rospy.wait_for_service('controller_manager/list_controllers', timeout=1)  # wait for 1sec max
        except rospy.exceptions.ROSException:
            return False
        return True

    #############
    # RQT (GUI) #
    #############

    def launch_gui(self):
        """
        Launch the GUI; in this case, it will launch `rqt`.
        """
        if self.gui is None:
            self.gui = subprocess.Popen(["rqt"], env=self.env, preexec_fn=os.setsid)  # , shell=True)

    ##########
    # Robots #
    ##########

    def get_robot_middleware(self, robot_id):
        """
        Return the robot middleware associated with the given robot id.

        Args:
            robot_id (int): unique robot id.

        Returns:
            ROSRobotMiddleware, None: robot middleware corresponding to the given body id. None if it could not find it.
        """
        return self._robots.get(robot_id)

    def load_urdf(self, urdf):
        """Load the given URDF file.

        The `load_urdf` will send a command to the physics server to load a physics model from a Universal Robot
        Description File (URDF). The URDF file is used by the ROS project (Robot Operating System) to describe robots
        and other objects, it was created by the WillowGarage and the Open Source Robotics Foundation (OSRF).
        Many robots have public URDF files, you can find a description and tutorial here:
        http://wiki.ros.org/urdf/Tutorials

        Args:
            urdf (str): a relative or absolute path to the URDF file on the file system of the physics server.

        Returns:
            int (non-negative): unique id associated to the load model.
        """
        # check URDF path; check if it is a valid robot directory. If not a robot, just skip
        path = os.path.dirname(os.path.abspath(urdf))  # /path/to/pyrobolearn/robots/urdfs/<robot>/
        # robot_path = '/'.join(path.split('/')[-4:-1])
        if 'pyrobolearn/robots/urdfs' in path:
            id_ = self.count_id
            self.count_id += 1

            # check if specific robot middleware exists in `robots` folder, and if so load it
            dirname = os.path.dirname(os.path.abspath(__file__))
            robot_name = str(os.path.basename(os.path.abspath(urdf)).split('.')[-2])  # name without extension (.urdf)
            if os.path.exists(dirname + '/robots/' + robot_name + '.py'):
                # import module
                module = importlib.import_module('pyrobolearn.simulators.middlewares.robots.' + robot_name)

                def predicate(cls):
                    return inspect.isclass(cls) and issubclass(cls, ROSRobotMiddleware) and cls != ROSRobotMiddleware

                # get classes inside modules that are a subclass of ROSRobotMiddleware
                classes = dict(inspect.getmembers(module, predicate))
                cls = DefaultROSRobotMiddleware

                # get first specific ROS robot middleware class if present
                for key in classes:
                    cls = classes[key]
                    if cls != ROSRobotMiddleware:
                        break

                print("Creating specific robot middleware: ", cls.__name__)

                robot = cls(id_, urdf=urdf, subscribe=self.is_subscribing, publish=self.is_publishing,
                            teleoperate=self.is_teleoperating, command=self.is_commanding, control_file=None,
                            launch_file=None)

            # otherwise, create default robot middleware
            else:
                print("Creating default robot middleware")
                robot = DefaultROSRobotMiddleware(id_, urdf=urdf, subscribe=self.is_subscribing,
                                                  publish=self.is_publishing, teleoperate=self.is_teleoperating,
                                                  command=self.is_commanding, control_file=None)
            self._robots[id_] = robot

            return id_

        return -1

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

        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.reset_joint_states(positions, joint_ids=joint_ids, velocities=velocities)

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
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.get_joint_positions(joint_ids)

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
        robot = self._robots.get(body_id)
        if robot is not None:
            if not check_teleoperate or self.is_teleoperating:
                robot.set_joint_positions(positions, joint_ids, velocities=velocities, kps=kps, kds=kds, forces=forces)

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
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.get_joint_velocities(joint_ids)

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
        robot = self._robots.get(body_id)
        if robot is not None:
            if not check_teleoperate or self.is_teleoperating:
                robot.set_joint_velocities(velocities, joint_ids, max_force=max_force)

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
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.get_joint_torques(joint_ids)

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
        robot = self._robots.get(body_id)
        if robot is not None:
            if not check_teleoperate or self.is_teleoperating:
                robot.set_joint_torques(torques, joint_ids)

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
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.get_jacobian(link_id, local_position=local_position, q=q)

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
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.get_inertia_matrix(q)

    def has_sensor(self, body_id, name):
        """
        Check if the specified robot has the given sensor.

        Args:
            body_id (int): body unique id.
            name (str): name of the sensor.

        Returns:
            bool: True if the specified robot has the given sensor.
        """
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.has_sensor(name)
        return False

    def get_sensor_values(self, body_id, name):
        """
        Return the sensor.

        Args:
            body_id (int): body unique id.
            name (str): name of the sensor.

        Returns:
            np.array, dict, list, None: sensor values. None if it didn't have anything.
        """
        robot = self._robots.get(body_id)
        if robot is not None:
            return robot.get_sensor_values(name)

    def can_teleoperate(self, body_id):
        """
        Check if we can teleoperate the given body.

        Args:
            body_id (int): unique body id.

        Returns:
            bool: True if we can teleoperate it.
        """
        robot = self._robots.get(body_id, None)
        if robot is None:
            return False
        return robot.teleoperate

    def can_subscribe(self, body_id):
        """
        Check if we can subscribe to topics with the given body.

        Args:
            body_id (int): unique body id.

        Returns:
            bool: True if we can subscribe.
        """
        robot = self._robots.get(body_id, None)
        if robot is None:
            return False
        return robot.subscribe

    def can_publish(self, body_id):
        """
        Check if we can publish to topics with the given body.

        Args:
            body_id (int): unique body id.

        Returns:
            bool: True if we can publish.
        """
        robot = self._robots.get(body_id, None)
        if robot is None:
            return False
        return robot.publish

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
        super(ROS, self).switch_mode(body_id=body_id, subscribe=subscribe, publish=publish, teleoperate=teleoperate,
                                     command=command)
        if body_id is not None:
            robot = self._robots.get(body_id, None)
            robot.switch_mode(subscribe=subscribe, publish=publish, teleoperate=teleoperate, command=command)


# Tests
if __name__ == '__main__':

    # check ROS
    pass
