#!/usr/bin/env python
"""Define the Bullet-ROS Simulator API.

This is the main interface that communicates with the PyBullet simulator [1] and use ROS [3] to query the state of the
robot and send instructions to it. By defining this interface, it allows to decouple the PyRoboLearn framework from
the simulator. It also converts some data types to the ones required by PyBullet. For instance, some methods in
PyBullet do not accepts numpy arrays but only lists. The interface provided here makes the necessary conversions.
Using ROS to query the state of the robot, it changes the state of the robot in the simulator, and moving the robot
in the simulator results in the real robot to move. Virtual sensors and actuators can also be defined.

The signature of each method defined here are inspired by [1] but in accordance with the PEP8 style guide [2].

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`
* `pyrobolearn.simulators.bullet.Bullet`
* `pyrobolearn.simulators.ros.ROS`

References:
    - [1] PyBullet: https://pybullet.org
    - [2] PyBullet Quickstart Guide: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
    - [3] ROS: http://www.ros.org/
    - [4] PEP8: https://www.python.org/dev/peps/pep-0008/
"""

# TODO: finish this interface and move ROS stuffs to ros.py

import os
import subprocess
import psutil
import signal
import importlib
import inspect

# from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.simulators.bullet import Bullet
# from pyrobolearn.simulators.ros import ROS


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BulletROS(Bullet):  # , ROS):
    r"""Bullet ROS

    Update the Bullet simulator based on the real robot(s): it updates the robot kinematic and dynamic state based on
    the values returned from the real robot(s).

    This can be useful for debug (check the differences between the real world and the simulated world), for virtual
    sensors, actuators, and forces, to map the real world to the simulated one, etc.
    """

    def __init__(self, render=True, subscribe=False, publish=False, teleoperate=False, ros_master_uri=11311, **kwargs):
        """
        Initialize the Bullet-ROS simulator.

        Args:
            render (bool): if True, it will open the GUI, otherwise, it will just run the server.
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robots, and will read
                the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robots.
            **kwargs (dict): optional arguments (this is not used here).
        """
        super(BulletROS, self).__init__(render=render, **kwargs)

        # Environment variable
        self.env = os.environ.copy()
        self.env["ROS_MASTER_URI"] = "http://localhost:" + str(ros_master_uri)

        # this is for the rospy methods such as: wait_for_service(), init_node(), ...
        os.environ['ROS_MASTER_URI'] = self.env['ROS_MASTER_URI']

        # run ROS core if not already running
        self.roscore = None
        if "roscore" not in [p.name() for p in psutil.process_iter()]:
            # subprocess.Popen("roscore", env=self.env)
            self.roscore = subprocess.Popen(["roscore", "-p", str(ros_master_uri)], env=self.env,
                                            preexec_fn=os.setsid)  # , shell=True)
        else:
            print('ROS core has already been initialized.')

        # set variables
        self.subscribe = subscribe
        self.publish = publish
        self.teleoperate = teleoperate

        # remember each publisher/subscriber
        self.subscribers = {}
        self.publishers = {}

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

    ##################
    # Static methods #
    ##################

    @staticmethod
    def has_middleware_communication_layer():
        """Return True if the simulator has a middleware communication layer (like ROS, YARP, etc)."""
        return True

    ###########
    # Methods #
    ###########

    def close(self):
        """
        Close everything
        """
        # delete each subscribers

        # delete each publishers

        # delete ROS
        if self.roscore is not None:
            os.killpg(os.getpgid(self.roscore.pid), signal.SIGTERM)

        # call parent destructor
        super(BulletROS, self).close()

    def load_urdf(self, filename, position=None, orientation=None, use_maximal_coordinates=None,
                  use_fixed_base=None, flags=None, scale=None):
        """Load the given URDF file.

        The load_urdf will send a command to the physics server to load a physics model from a Universal Robot
        Description File (URDF). The URDF file is used by the ROS project (Robot Operating System) to describe robots
        and other objects, it was created by the WillowGarage and the Open Source Robotics Foundation (OSRF).
        Many robots have public URDF files, you can find a description and tutorial here:
        http://wiki.ros.org/urdf/Tutorials

        Important note:
            most joints (slider, revolute, continuous) have motors enabled by default that prevent free
            motion. This is similar to a robot joint with a very high-friction harmonic drive. You should set the joint
            motor control mode and target settings using `pybullet.setJointMotorControl2`. See the
            `setJointMotorControl2` API for more information.

        Warning:
            by default, PyBullet will cache some files to speed up loading. You can disable file caching using
            `setPhysicsEngineParameter(enableFileCaching=0)`.

        Args:
            filename (str): a relative or absolute path to the URDF file on the file system of the physics server.
            position (vec3): create the base of the object at the specified position in world space coordinates [x,y,z]
            orientation (quat): create the base of the object at the specified orientation as world space quaternion
                [x,y,z,w]
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
        id_ = super(BulletROS, self).load_urdf(filename=filename, position=position, orientation=orientation,
                                               use_maximal_coordinates=use_maximal_coordinates,
                                               use_fixed_base=use_fixed_base, flags=flags, scale=scale)
        # get path to directory of urdf
        path = os.path.dirname(os.path.abspath(filename))               # /path/to/pyrobolearn/robots/urdfs/<robot>/
        robot_directory_name = path.split('/')[-1]                      # <robot>
        # path = path + '/../../ros/' + robot_directory_name + '/'        # /path/to/pyrobolearn/robots/ros/<robot>/

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

        return id_

    def get_joint_positions(self, body_id, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.float[N]: joint positions [rad]
        """
        if body_id in self.subscribers:
            q = self.subscribers[body_id].get_joint_positions(joint_ids)
            if len(q) == len(joint_ids):
                super(BulletROS, self).set_joint_positions(body_id=body_id, joint_ids=joint_ids, positions=q)  # reset?
            else:  # if failed to get the joint positions from the subscriber
                q = super(BulletROS, self).get_joint_positions(body_id=body_id, joint_ids=joint_ids)
        else:
            q = super(BulletROS, self).get_joint_positions(body_id=body_id, joint_ids=joint_ids)
            if self.teleoperate and body_id in self.publishers:
                self.publishers[body_id].set_joint_positions(joint_ids, q)
                self.publishers[body_id].publish('joint_states')
        return q

    def set_joint_positions(self, body_id, joint_ids, positions, velocities=None, kps=None, kds=None, forces=None):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            positions (float, np.float[N]): desired position, or list of desired positions [rad]
            velocities (None, float, np.float[N]): desired velocity, or list of desired velocities [rad/s]
            kps (None, float, np.float[N]): position gain(s)
            kds (None, float, np.float[N]): velocity gain(s)
            forces (None, float, np.float[N]): maximum motor force(s)/torque(s) used to reach the target values.
        """
        super(BulletROS, self).set_joint_positions(body_id, joint_ids, positions, velocities, kps, kds, forces)
        if body_id in self.publishers:
            self.publishers[body_id].set_joint_positions(joint_ids, positions)
            self.publishers[body_id].publish('joint_states')

    def get_joint_velocities(self, body_id, joint_ids):
        """
        Get the velocity of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.float[N]: joint velocities [rad/s]
        """
        if body_id in self.subscribers:
            dq = self.subscribers[body_id].get_joint_velocities(joint_ids)
            if len(dq) == len(joint_ids):
                super(BulletROS, self).set_joint_velocities(body_id=body_id, joint_ids=joint_ids, velocities=dq)
            else:  # if failed to get the joint velocities from the subscriber
                dq = super(BulletROS, self).get_joint_velocities(body_id=body_id, joint_ids=joint_ids)
        else:
            dq = super(BulletROS, self).get_joint_velocities(body_id=body_id, joint_ids=joint_ids)
            if self.teleoperate and body_id in self.publishers:
                self.publishers[body_id].set_joint_velocities(joint_ids, dq)
                self.publishers[body_id].publish('joint_states')
        return dq

    def set_joint_velocities(self, body_id, joint_ids, velocities, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            velocities (float, np.float[N]): desired velocity, or list of desired velocities [rad/s]
            max_force (None, float, np.float[N]): maximum motor forces/torques
        """
        super(BulletROS, self).set_joint_velocities(body_id, joint_ids, velocities, max_force)
        if body_id in self.publishers:
            self.publishers[body_id].set_joint_velocities(joint_ids, velocities)
            self.publishers[body_id].publish('joint_states')

    def get_joint_torques(self, body_id, joint_ids):
        """
        Get the applied torque(s) on the given joint(s). "This is the motor torque applied during the last `step`.
        Note that this only applies in VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the
        applied joint motor torque is exactly what you provide, so there is no need to report it separately." [1]

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.float[N]: torques associated to the given joints [Nm]
        """
        if body_id in self.subscribers:
            torques = self.subscribers[body_id].get_joint_torques(joint_ids)
            if len(torques) == len(joint_ids):
                super(BulletROS, self).set_joint_torques(body_id=body_id, joint_ids=joint_ids, torques=torques)
            else:  # if failed to get the joint torques from the subscriber
                torques = super(BulletROS, self).get_joint_torques(body_id=body_id, joint_ids=joint_ids)
        else:
            torques = super(BulletROS, self).get_joint_torques(body_id=body_id, joint_ids=joint_ids)
            if self.teleoperate and body_id in self.publishers:
                self.publishers[body_id].set_joint_torques(joint_ids, torques)
                self.publishers[body_id].publish('joint_states')
        return torques

    def set_joint_torques(self, body_id, joint_ids, torques):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            torques (float, list of float): desired torque(s) to apply to the joint(s) [N].
        """
        super(BulletROS, self).set_joint_torques(body_id, joint_ids, torques)
        if body_id in self.publishers:
            self.publishers[body_id].set_joint_torques(joint_ids, torques)
            self.publishers[body_id].publish('joint_states')


# Test
if __name__ == '__main__':
    pass
