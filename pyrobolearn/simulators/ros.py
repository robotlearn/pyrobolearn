#!/usr/bin/env python
"""Define the Bullet Simulator API.

This is the main interface that communicates with the PyBullet simulator [1]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
PyBullet. For instance, some methods in PyBullet do not accepts numpy arrays but only lists. The interface provided
here makes the necessary conversions.

The signature of each method defined here are inspired by [1,2] but in accordance with the PEP8 style guide [3].

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    [1] PyBullet: https://pybullet.org
    [2] PyBullet Quickstart Guide: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
    [3] PEP8: https://www.python.org/dev/peps/pep-0008/
"""

# TODO

import rospy

from pyrobolearn.simulators.simulator import Simulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ROSModel(object):
    r"""ROS Model

    """

    def __init__(self, filename):
        self.urdf = filename
        # get ros services and ros topics from URDF

        # create
        pass


class ROS(Simulator):
    r"""ROS Interface
    """

    def __init__(self, subscribe=False, publish=False, master_uri=11311, **kwargs):
        super(ROS, self).__init__(render=False)

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

        # set variables
        self.subscribe = subscribe
        self.publish = publish

        # remember each publisher/subscriber
        self.subscribers = {}
        self.publishers = {}
        self.models = []

        self.count_id = -1

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
        # get path to directory of urdf
        path = os.path.dirname(filename)
        robot_directory_name = path.split('/')[-1]
        path = path + '/../../ros/' + robot_directory_name + '/'

        # check if valid robot directory

        # load subscriber in simulator
        if self.subscribe:
            pass

        # load publisher in simulator
        if self.publish:
            pass

        self.count_id += 1

        return self.count_id

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
            return self.subscribers[body_id].get_joint_positions[joint_ids]

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
        if body_id in self.publishers:
            self.publishers[body_id].set_joint_positions(joint_ids, positions)

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
        pass

    def set_joint_velocities(self, body_id, joint_ids, velocities, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            velocities (float, np.float[N]): desired velocity, or list of desired velocities [rad/s]
            max_force (None, float, np.float[N]): maximum motor forces/torques
        """
        pass

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
        pass

    def set_joint_torques(self, body_id, joint_ids, torques):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            torques (float, list of float): desired torque(s) to apply to the joint(s) [N].
        """
        pass
