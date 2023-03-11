#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gazebo ROS simulator

This simulator uses Gazebo as the simulator, ROS to communicate with this simulator (to send and receive any
information related to the simulator and the objects inside of it like the robots), and RBDL to compute the kinematics
and dynamics of the robots.

Dependencies in PRL:
* `pyrobolearn.simulators.ros_rbdl.ROS_RBDL`

References:
    [1] ROS: http://www.ros.org/
    [2] Gazebo: http://gazebosim.org/
    [3] RBDL: https://rbdl.bitbucket.io/
"""

# TODO: this is not finished

import numpy as np
import subprocess, os, signal, sys, time

# import ROS and RBDL
import rospy
import rbdl

# messages and services
import std_msgs.msg as stdmsg
import std_srvs.srv as stdsrv
import gazebo_msgs.msg as gazmsg
import gazebo_msgs.srv as gazsrv
import geometry_msgs.msg import geomsg

# import Gazebo-ROS related libraries
from gazebo_ros import gazebo_interface

# from gazebo_msgs.msg import *
# from gazebo_msgs.srv import *
# from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench, Vector3
import tf.transformations as tft

# import PRL
from pyrobolearn.simulators.ros_rbdl import ROS_RBDL

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GazeboROS(ROS_RBDL):
    r"""Gazebo ROS RBDL Interface

    This simulator uses Gazebo as the simulator, ROS to communicate with this simulator (to send and receive any
    information related to the simulator and the objects inside of it like the robots), and RBDL to compute the
    kinematics and dynamics of the robots. This class acts as the main bridge that connects what happens between
    the simulator Gazebo and the PyRoboLearn framework.

    Examples::
        from pyrobolearn.simulators import GazeboROS

        sim = GazeboROS(render=True)

    References:
        [1] ROS: http://www.ros.org/
        [2] Gazebo: http://gazebosim.org/
        [3] RBDL: https://rbdl.bitbucket.io/

    Repositories:
    * Xacro package: https://github.com/ros/xacro
    * Gazebo ROS packages: https://github.com/ros-simulation/gazebo_ros_pkgs
    * ROS control packages: https://github.com/ros-controls/ros_control
    """

    def __init__(self, render=True, ros_master_uri=11316, gazebo_master_uri=11345):
        super(GazeboROS, self).__init__()

        # Environment variable
        self.env = os.environ.copy()

        self.env["ROS_MASTER_URI"] = "http://localhost:" + str(ros_master_uri)
        self.env["GAZEBO_MASTER_URI"] = "http://localhost:" + str(gazebo_master_uri)

        # this is for the rospy methods such as: wait_for_service(), init_node(), ...
        os.environ['ROS_MASTER_URI'] = self.env['ROS_MASTER_URI']

        # create ROS core
        # subprocess.Popen("roscore", env=self.env)
        self.ros_proc = subprocess.Popen(["roscore", "-p", str(ros_master_uri)], env=self.env,
                                         preexec_fn=os.setsid)  # , shell=True)

        # create Gazebo ROS
        self.gzserver_proc = None
        self.gzclient_proc = None

        # Gazebo Services
        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_simulation', stdsrv.Empty)
        self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', stdsrv.Empty)
        self.unpause_srv = rospy.ServiceProxy('/gazebo/unpause_physics', stdsrv.Empty)
        self.get_physics_properties_srv = rospy.ServiceProxy('/gazebo/get_physics_properties', stdsrv.Empty)
        self.set_physics_properties_srv = rospy.ServiceProxy('/gazebo/set_physics_properties',
                                                             gazsrv.SetPhysicsProperties)

        # keep a list of bodies
        self.bodies = []

    # Simulators

    def reset(self):
        """
        Reset the Gazebo simulation.
        """
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_srv()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

    def close(self):
        """
        Close everything
        """
        # delete Gazebo
        if self.gzclient_proc is not None:
            os.killpg(os.getpgid(self.gzclient_proc.pid), signal.SIGTERM)
        if self.gzserver_proc is not None:
            os.killpg(os.getpgid(self.gzserver_proc.pid), signal.SIGTERM)

        # delete ROS
        os.killpg(os.getpgid(self.ros_proc.pid), signal.SIGTERM)

    def seed(self, seed=None):
        """Set the given seed in the simulator."""
        if seed is None:
            return []
        rospy.wait_for_service('/gazebo/set_seed')
        try:
            rospy.ServiceProxy('/gazebo/set_seed', SetSeedSrv)(seed)
        except rospy.ServiceException as e:
            print("/GazeboRosGym/set_seed service call failed")
        return [seed]

    def step(self, sleep_dt=0):
        """Perform a step in the simulator, and sleep the specified time."""
        self.unpause()
        time.sleep(sleep_dt)
        # TODO apply stuffs in simulator
        self.pause()

    def render(self, enable=True):
        """Render the simulation."""
        if enable:
            if self.gzclient_proc is None:
                pass
        else:
            if self.gzclient_proc is not None:
                pass

    def set_time_step(self, time_step):
        """Set the time step in the simulator."""
        set_physics_request = self.get_physics_properties()

        # set time step
        set_physics_request.time_step = time_step

        # set the physics properties
        rospy.wait_for_service('/gazebo/set_physics_properties')
        try:
            self.set_physics_properties_srv(set_physics_request)
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

    def set_real_time(self):
        """Enable real time in the simulator."""
        self.unpause()

    def pause(self):
        """Pause the simulator if in real-time."""
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_srv()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

    def unpause(self):
        """Unpause the simulator if in real-time."""
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_srv()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

    def get_physics_properties(self):
        """Get the physics engine parameters."""
        rospy.wait_for_service('/gazebo/get_physics_properties')
        try:
            srv = self.get_physics_properties_srv()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        return srv

    def set_physics_properties(self, *args, **kwargs):
        """Set the physics engine parameters."""
        rospy.wait_for_service('/gazebo/set_physics_properties')
        try:
            self.set_physics_properties_srv()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

    def start_logging(self, *args, **kwargs):
        """Start the logging."""
        pass

    def stop_logging(self, logger_id):
        """Stop the logging."""
        pass

    def set_gravity(self, gravity=(0, 0, -9.81)):
        """Set the gravity in the simulator."""
        set_physics_request = self.get_physics_properties()

        # set attributes
        set_physics_request.gravity.x = gravity[0]
        set_physics_request.gravity.y = gravity[1]
        set_physics_request.gravity.z = gravity[2]

        # set the physics properties
        rospy.wait_for_service('/gazebo/set_physics_properties')
        try:
            self.set_physics_properties_srv(set_physics_request)
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

    def save(self, on_disk=False):
        """Save the state of the simulator."""
        pass

    def load(self, state):
        """Load the simulator to a previous state."""
        pass

    def load_plugin(self, plugin):
        """Load a certain plugin in the simulator."""
        pass

    def execute_plugin_commands(self, plugin_id, commands):
        """Execute the commands on the specified plugin."""
        pass

    def unload_plugin(self, plugin_id):
        """Unload the specified plugin from the simulator."""
        pass

    # loading URDFs, SDFs, MJCFs

    def load_urdf(self, filename, position, orientation):
        """Load a URDF file in the simulator."""
        robot_namespace = rospy.get_namespace()
        gazebo_namespace = "/gazebo"
        reference_frame = ""
        model_name = filename.split('/')[-1].split('.')[0]  # assume filename='path/to/file(.xacro).urdf'

        # if xacro file, use xacro.py with the list of arguments

        # load file
        f = open(filename, 'r')
        model_xml = f.read()
        if model_xml == "":
            rospy.logerr("Error: file is empty %s", filename)
            sys.exit(0)

        # create initial pose
        initial_pose = geomsg.Pose()
        initial_pose.position.x = position[0]
        initial_pose.position.y = position[1]
        initial_pose.position.z = position[2]
        q = geomsg.Quaternion()
        q.x = orientation[0]
        q.y = orientation[1]
        q.z = orientation[2]
        q.w = orientation[3]
        initial_pose.orientation = q

        success = gazebo_interface.spawn_urdf_model_client(model_name, model_xml, robot_namespace, initial_pose,
                                                           reference_frame, gazebo_namespace)
        if not success:
            raise ValueError("Could not load the given URDF in Gazebo.")

        body_id = len(self.bodies)
        self.bodies.append(model_name)
        return body_id

    def load_sdf(self, filename):
        """Load a SDF file in the simulator."""
        robot_namespace = rospy.get_namespace()
        gazebo_namespace = "/gazebo"
        reference_frame = ""
        position = (0., 0., 0.)
        orientation = (0., 0., 0., 1.)
        model_name = filename.split('/')[-1].split('.')[-2]  # assume filename='path/to/file.sdf'

        # load file
        f = open(filename, 'r')
        model_xml = f.read()
        if model_xml == "":
            rospy.logerr("Error: file is empty %s", filename)
            sys.exit(0)

        # create initial pose
        initial_pose = geomsg.Pose()
        initial_pose.position.x = position[0]
        initial_pose.position.y = position[1]
        initial_pose.position.z = position[2]
        q = geomsg.Quaternion()
        q.x = orientation[0]
        q.y = orientation[1]
        q.z = orientation[2]
        q.w = orientation[3]
        initial_pose.orientation = q

        success = gazebo_interface.spawn_sdf_model_client(model_name, model_xml, robot_namespace, initial_pose,
                                                          reference_frame, gazebo_namespace)
        if not success:
            raise ValueError("Could not load the given SDF in Gazebo.")

        body_id = len(self.bodies)
        self.bodies.append(model_name)
        return body_id

    def load_mjcf(self, filename):
        """Load MJCF file."""
        raise NotImplementedError("Loading a MJCF xml file in Gazebo is currently not possible.")




class GazeboROSEnv(gazebo_env.GazeboEnv):
    """
    This class defines the Gazebo - OpenAI Gym interface.
    The communication between the 2 systems is done using ROS.
    """
    
    def __init__(self, roslaunch_filename, package_name, ros_master_uri=11316, gazebo_master_uri=11345):

        if roslaunch_filename is None:
            raise ValueError("Expecting the roslaunch filename to be different from None")
        if package_name is None:
            raise ValueError("Expecting the package name to be different from None")

        # Environment variable
        self.env = os.environ.copy()

        self.env["ROS_MASTER_URI"] = "http://localhost:" + str(ros_master_uri)
        self.env["GAZEBO_MASTER_URI"] = "http://localhost:" + str(gazebo_master_uri)

        # this is for the rospy methods such as: wait_for_service(), init_node(), ...
        os.environ['ROS_MASTER_URI'] = self.env['ROS_MASTER_URI']

        # Roscore and init node
        #subprocess.Popen("roscore", env=self.env)
        print('ROSCORE...')
        self.ros_proc = subprocess.Popen(["roscore", "-p", str(ros_master_uri)], env=self.env, preexec_fn=os.setsid) #, shell=True)

        rospy.wait_for_service('/rosout/get_loggers')
        print('REGISTERING NODE...')
        rospy.init_node('gym', anonymous=True)

        # Roslaunch
        print('ROSLAUNCH...')
        print(package_name)
        print(roslaunch_filename)
        self.roslaunch_proc = subprocess.Popen(["roslaunch", package_name, roslaunch_filename, 'gui:=false', 'paused:=true'],
                                                env=self.env,
                                                preexec_fn=os.setsid)
                                                #shell=True)
        self.gzclient_pid = 0

        rospy.wait_for_service('/gazebo/reset_simulation')
        print('ROSLAUNCH DONE')

        # Gazebo Services
        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_simulation', stdSrv.Empty)
        self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', stdSrv.Empty)
        self.unpause_srv = rospy.ServiceProxy('/gazebo/unpause_physics', stdSrv.Empty)

    def _seed(self, seed):
        """
        Set the seed in Gazebo using the new defined service.
        """
        if seed is None: return []
        rospy.wait_for_service('/GazeboRosGym/set_seed')
        try:
            rospy.ServiceProxy('/GazeboRosGym/set_seed', SetSeed)(seed)
        except rospy.ServiceException as e:
            print("/GazeboRosGym/set_seed service call failed")
        return [seed]
    
    def reset_simulation(self):
        """
        Reset the Gazebo simulation.
        """
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_srv()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
            
    def pause_physics(self):
        """
        Pause the Gazebo physics engine.
        """
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_srv()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")
            
    def unpause_physics(self):
        """
        Unpause the Gazebo physics engine.
        """
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_srv()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
            
    def configure(self, *args, **kwargs):
        """
        Configure Gazebo with the given parameters.
        
        Example (using the various rosservice for Gazebo):
        - set the PID parameters
        - set link/joint properties
        - set link state
        - set model configuration/state
        - set physics properties (time step & update rate)
        """
        raise NotImplementedError("This function needs to be overwritten...")
            
    def act(self, action):
        """
        Apply the action in the environment.
        """
        raise NotImplementedError("This function needs to be overwritten...")
            
    def get_state(self):
        """
        Return the state.

        Example:
        The observation could be an image, while the state could be the position 
        (and velocity) of a target on the picture. The state is used to compute
        the reward function.
        """
        raise NotImplementedError("This function needs to be overwritten...")

    def get_obs(self):
        """
        Return the observation.

        Example:
        The observation could be an image, while the state could be the position 
        (and velocity) of a target on the picture. The state is used to compute
        the reward function.
        """
        raise NotImplementedError("This function needs to be overwritten...")

    def get_state_and_obs(self):
        """
        Return the state and observation.

        Example:
        The observation could be an image, while the state could be the position
        (and velocity) of a target on the picture. The state is used to compute
        the reward function.
        """
        return get_state(), get_obs()
        
    def compute_reward(self, state, obs):
        """
        Compute and return the reward based on the state and on the observation.
        It also returns a boolean value indicating if the task is over or not.
        """
        raise NotImplementedError("This function needs to be overwritten...")
        
    def _step(self, action):
        """
        Run one timestep in the simulator.
        """
        self.unpause_physics()
        
        self.act(action) # should this be before unpause_physics?
        state, obs = self.get_state_and_obs()

        self.pause_physics()
        
        reward, done = self.compute_reward(state, obs)
        return obs, reward, done, state

    def _reset(self):
        """
        Reset the simulator.
        """
        self.reset_simulation()
        self.unpause_physics()
        obs = self.get_state()[1]
        self.pause_physics()
        return obs
