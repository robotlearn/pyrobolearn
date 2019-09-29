# -*- coding: utf-8 -*-
#!/usr/bin/env python
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
* `pyrobolearn.simulators.middlewares.middleware.MiddleWare`

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

import rospy
import rostopic
import controller_manager.controller_manager_interface as cm_interface
import roslaunch
import rosparam
import std_msgs.msg as std_msg
import sensor_msgs.msg as sensor_msg
import gazebo_msgs.msg as gazebo_msg
import geometry_msgs.msg as geometry_msg

from pyrobolearn.simulators.middlewares.middleware import MiddleWare
from pyrobolearn.simulators.middlewares.ros_publisher import Publisher, PublisherData, RobotPublisher
from pyrobolearn.simulators.middlewares.ros_subscriber import Subscriber, SubscriberData, RobotSubscriber


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["ROS (Willow Garage)", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotMiddleWare(object):
    r"""Robot middleware interface.

    """

    def __init__(self, robot_id, urdf=None, subscribe=False, publish=False, teleoperate=False, control_file=None):
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
            control_file (str): path to the YAML control file.
        """
        self.id = robot_id
        self.urdf = urdf
        self.control_file = control_file
        self.subscribers = {}
        self.publishers = {}

        # set variables
        self.subscribe = subscribe
        self.publish = publish
        self.teleoperate = teleoperate

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
        q = self.subscribers[body_id].get_joint_positions(joint_ids)
        if self.teleoperate and body_id in self.publishers:
            self.publishers[body_id].set_joint_positions(joint_ids, q)
            self.publishers[body_id].publish('joint_states')
        return q

    def set_joint_positions(self, joint_ids, positions, velocities=None, kps=None, kds=None, forces=None):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            joint_ids (int, list[int]): joint id, or list of joint ids.
            positions (float, np.array[float[N]]): desired position, or list of desired positions [rad]
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

    def set_joint_velocities(self, joint_ids, velocities, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            joint_ids (int, list[int]): joint id, or list of joint ids.
            velocities (float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
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

    def set_joint_torques(self, joint_ids, torques):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            joint_ids (int, list[int]): joint id, or list of joint ids.
            torques (float, list[float]): desired torque(s) to apply to the joint(s) [N].
        """
        pass


class ROS(MiddleWare):
    r"""ROS Interface middleware

    This middleware can be given to the simulator which can then interact with robots.
    """

    def __init__(self, subscribe=False, publish=False, teleoperate=False, master_uri=11311, **kwargs):
        """
        Initialize the ROS middleware.

        Args:
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robots, and will read
              the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robots.
            teleoperate (bool): if True, it will move the robot based on the received or sent values based on the 2
              previous attributes :attr:`subscribe` and :attr:`publish`.
            master_uri (int): ROS master URI.
        """
        super(ROS, self).__init__(subscribe=subscribe, publish=publish, teleoperate=teleoperate)

        # Environment variable
        self.env = os.environ.copy()
        self.env["ROS_MASTER_URI"] = "http://localhost:" + str(master_uri)

        # this is for the rospy methods such as: wait_for_service(), init_node(), ...
        os.environ['ROS_MASTER_URI'] = self.env['ROS_MASTER_URI']

        # run ROS core if not already running
        self.roscore = None
        if "roscore" not in [p.name() for p in psutil.process_iter()]:
            # subprocess.Popen("roscore", env=self.env)
            self.roscore = subprocess.Popen(["roscore", "-p", str(master_uri)], env=self.env,
                                            preexec_fn=os.setsid)  # , shell=True)

        # remember each publisher/subscriber
        self.subscribers = {}
        self.publishers = {}
        self.models = []

        self._robots = {}  # {body_id: RobotMiddleware}

        self.count_id = -1

        # roslaunch
        self.launch = roslaunch.scriptapi.ROSLaunch()
        self.launch.start()

        self.processes = {}  # {Node: process}

    ##############
    # Properties #
    ##############

    @property
    def is_subscribing(self):
        """Return True if we are subscribing to topics."""
        return self.subscribe

    @property
    def is_publishing(self):
        """Return True if we are publishing to topics."""
        return self.publish

    ###########
    # Methods #
    ###########

    ##############
    # ROS launch #
    ##############

    @staticmethod
    def create_ros_node(package, node_type, name=None, namespace='/', machine_name=None, args='', respawn=False,
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

    ################################################
    # ROS Topics/Services + Publishers/Subscribers #
    ################################################

    def add_subscriber(self, body_id, topic_name, freq, fct, attribute):
        pass

    def add_publisher(self, body_id, topic_name, freq, fct, attribute):
        pass

    def remap(self, topic1, topic2, freq):
        pass

    def remove_publisher(self, publisher_id):
        pass

    def remove_subscriber(self, subscriber_id):
        pass

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
            str: type of the topic
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
        namespace = params[1] + 'rrbot/'
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

    ###############
    # ROS CONTROL #
    ###############

    def get_pid(self, body_id):
        pass

    def set_pid(self, body_id, p=None, i=None, d=None):
        pass

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

    ##############
    # Middleware #
    ##############

    def close(self):
        """
        Close everything
        """
        # delete each subscribers

        # delete each publishers

        for process in self.processes.values():
            process.stop()

        # stop launch
        if self.launch:
            self.launch.stop()

        # delete ROS
        if self.roscore is not None:
            os.killpg(os.getpgid(self.roscore.pid), signal.SIGTERM)

    def load_urdf(self, filename, position=None, orientation=None, use_maximal_coordinates=None,
                  use_fixed_base=None, flags=None, scale=None):
        """Load the given URDF file.

        The load_urdf will send a command to the physics server to load a physics model from a Universal Robot
        Description File (URDF). The URDF file is used by the ROS project (Robot Operating System) to describe robots
        and other objects, it was created by the WillowGarage and the Open Source Robotics Foundation (OSRF).
        Many robots have public URDF files, you can find a description and tutorial here:
        http://wiki.ros.org/urdf/Tutorials

        Args:
            filename (str): a relative or absolute path to the URDF file on the file system of the physics server.
            position (np.array[float[3]]): create the base of the object at the specified position in world space
              coordinates [x,y,z]
            orientation (np.array[float[4]]): create the base of the object at the specified orientation as world
              space quaternion [x,y,z,w]
            use_maximal_coordinates (int): Experimental. By default, the joints in the URDF file are created using the
                reduced coordinate method: the joints are simulated using the Featherstone Articulated Body algorithm
                (btMultiBody in Bullet 2.x). The useMaximalCoordinates option will create a 6 degree of freedom rigid
                body for each link, and constraints between those rigid bodies are used to model joints.
            use_fixed_base (bool): force the base of the loaded object to be static
            flags (int): URDF_USE_INERTIA_FROM_FILE (val=2): by default, Bullet recomputed the inertia tensor based on
                mass and volume of the collision shape. If you can provide more accurate inertia tensor, use this flag.
                URDF_USE_SELF_COLLISION (val=8): by default, Bullet disables self-collision. This flag let's you
                enable it.
                You can customize the self-collision behavior using the following flags:
                    * URDF_USE_SELF_COLLISION_EXCLUDE_PARENT (val=16) will discard self-collision between links that
                        are directly connected (parent and child).
                    * URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS (val=32) will discard self-collisions between a
                        child link and any of its ancestors (parents, parents of parents, up to the base).
                    * URDF_USE_IMPLICIT_CYLINDER (val=128), will use a smooth implicit cylinder. By default, Bullet
                        will tessellate the cylinder into a convex hull.
            scale (float): scale factor to the URDF model.

        Returns:
            int (non-negative): unique id associated to the load model.
        """
        id_ = self.count_id
        self.count_id += 1

        # get path to directory of urdf
        path = os.path.dirname(os.path.abspath(filename))           # /path/to/pyrobolearn/robots/urdfs/<robot>/
        robot_directory_name = path.split('/')[-1]                  # <robot>
        path = path + '/../../ros/' + robot_directory_name + '/'    # /path/to/pyrobolearn/robots/ros/<robot>/

        # check if valid robot directory
        # if os.path.isdir(path):
        robot_path = '/'.join(path.split('/')[-5:-2])
        if robot_path == 'pyrobolearn/robots/urdfs':

            # get corresponding subscriber/publisher
            def check_ros(name, dictionary, id_):
                # TODO: do I really need a class for each robot? Can I not just use RobotPublisher?
                # if os.path.isfile(path + name + '.py'):
                #     module = importlib.import_module('pyrobolearn.robots.ros.' + robot_directory_name + '.' + name)
                #     classes = inspect.getmembers(module, inspect.isclass)  # list of (name, class)
                #     length = len(name)
                #     robot_name = ''.join(robot_directory_name.split('_'))
                #
                #     # go through each class and get the :attr:`name` corresponding to the robot and add it to the
                #     # given :attr:`dictionary`
                #     for name, cls in classes:
                #         if name[:-length].lower() == robot_name:
                #             dictionary[id_] = cls(id_=id_)
                #             break
                module = importlib.import_module('pyrobolearn.robots.ros.' + name)
                classes = dict(inspect.getmembers(module, inspect.isclass))
                cls = classes['Robot' + name.capitalize()]
                dictionary[id_] = cls(name=robot_directory_name, id_=id_)

            # load subscriber in simulator
            if self.subscribe:
                check_ros('subscriber', self.subscribers, id_)

            # load publisher in simulator
            if self.publish:
                check_ros('publisher', self.publishers, id_)


        # get path to the URDF folder
        path = os.path.abspath(filename)  # /path/to/pyrobolearn/robots/urdfs/<robot>/robot.urdf
        dirname = str(os.path.dirname(path))   # /path/to/pyrobolearn/robots/urdfs/<robot>/
        basename = str(os.path.basename(path).split('.')[-2])  # robot name without extension
        config_file = dirname + basename + ".yaml"

        # check for YAML control configuration file
        if os.path.isfile(config_file):
            # if it exists, import it
            data = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
        else:
            data = {basename: {'joint_state_controller': {'publish_rate': 50,
                                                          'type': 'joint_state_controller/JointStateController'}}}

        node = roslaunch.core.Node(package="controller_manager", name="controller_spawner", node_type="spawner",
                                   respawn="false", output="screen", namespace="/rrbot", args=' '.join(
                ["joint_state_controller", "joint1_position_controller", "joint2_position_controller"]))

        # create subscribers

        return id_

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
            if check_teleoperate and not self.teleoperate:
                return None
            return robot.set_joint_positions(joint_ids, positions, velocities=velocities, kps=kps, kds=kds,
                                             forces=forces)

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
            if check_teleoperate and not self.teleoperate:
                return None
            return robot.set_joint_velocities(joint_ids, velocities, max_force=max_force)

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
            if check_teleoperate and not self.teleoperate:
                return None
            return robot.set_joint_torques(joint_ids, torques)

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
        robot = self._robots.get(body_id, None)
        if robot is None:
            return False
        return robot.teleoperate

    def can_subscribe(self, body_id):
        robot = self._robots.get(body_id, None)
        if robot is None:
            return False
        return robot.subscribe

    def can_publish(self, body_id):
        robot = self._robots.get(body_id, None)
        if robot is None:
            return False
        return robot.publish
