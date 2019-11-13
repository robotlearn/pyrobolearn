#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the UR5 ROS Robot middleware API.

This is robot middleware interface between the UR5 robot and ROS. This file should be modified by the user!!
Currently, we use the following setup:
- https://github.com/ros-industrial/universal_robot
by launching `ur_gazebo/launch/ur5.launch` or `ur_e_gazebo/launch/ur5e.launch`.

The topics for the joint states and joint commands (=joint trajectories) are:
- /joint_states
- /arm_controller/command
"""

import time
import numpy as np
import rospy

# import ROS messages / services
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from sensor_msgs.msg import JointState

# MoveJoints = None
# try:
#     from reset_services.srv import MoveJoints
# except ImportError as e:
#     print("The service `MoveJoints` is not available... Please compile it using `catkin_make` in order to use it "
#           "when resetting the joint states.\n" + str(e))

from pyrobolearn.simulators.middlewares.ros import ROSRobotMiddleware
from pyrobolearn.utils.filters import MovingAverageFilter


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class UR5ROSMiddleware(ROSRobotMiddleware):
    r"""Robot middleware interface.

    The robot middleware interface is an interface between a particular robot and the middleware. The middleware
    possesses a list of Robot middleware interfaces (one for each robot).

    Notably, the robot middleware has a unique id, has a list of publishers and subscribers associated with the given
    robot.
    """

    def __init__(self, robot_id, urdf=None, subscribe=False, publish=False, teleoperate=False, command=True,
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
            command (bool): if True, it will subscribe/publish to some (joint) commands. If False, it will
              subscribe/publish to some (joint) states.
            control_file (str, None): path to the YAML control file. If provided, it will be parsed.
            launch_file (str, None): path to the ROS launch file. If provided, it will be parsed.
        """
        joint_state_topic = '/joint_states'
        super(UR5ROSMiddleware, self).__init__(robot_id, urdf, subscribe, publish, teleoperate, command,
                                                  control_file, launch_file, joint_state_topics=joint_state_topic)

        # joint names in the messages
        self.msg_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                                'wrist_2_joint', 'wrist_3_joint']

        # joint trajectory point instance
        self.arm_point = JointTrajectoryPoint()
        self.arm_point.positions = np.zeros(len(self.msg_joint_names))
        self.arm_point.velocities = 0.1 * np.ones(len(self.msg_joint_names))
        # self.arm_point.effort = 0.1 * np.ones(len(self.msg_joint_names)

        # update publisher
        arm_topic = '/arm_controller/command'
        self.arm_publisher = self.publisher.create_publisher(name='arm_trajectory', topic=arm_topic,
                                                             msg_class=JointTrajectory)

        # set joint names and trajectory point in message
        self.arm_publisher.msg.joint_names = self.msg_joint_names
        self.arm_publisher.msg.points = [self.arm_point]

        # create reset joint state service
        self.reset_joint_service = None
        self.use_real_robot = True
        # if ResetSrv is not None and self.use_real_robot:
        #     self.reset_joint_service_name = '/arm/move_joints_absolute'
        #     self.reset_joint_service = rospy.ServiceProxy(self.reset_joint_service_name, ResetSrv)

        self.filter = MovingAverageFilter(alpha=0.3)

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
        if self.reset_joint_service is not None:
            input("Press Enter to call ROS service to reset the joint states...")

            # call rosservice to reset the joints
            rospy.wait_for_service(self.reset_joint_service_name)
            try:
                # keep only joint arm indices/positions
                q_indices = None if joint_ids is None else self.q_indices[joint_ids]
                if q_indices is None:
                    q_indices = np.array(range(len(self.msg_joint_names)))
                positions = positions[:len(self.msg_joint_names)]
                q_indices = q_indices[q_indices < len(self.msg_joint_names)]
                positions = positions[q_indices]
                args = np.array(['Joint' + str(i+1) for i in range(len(self.msg_joint_names))])
                kwargs = dict(zip(args[q_indices], positions))
                velocity_scale = 0.1  # 1 = max velocity, 0 = don't move
                duration_time = 10  # 10 secs
                response = self.reset_joint_service(**kwargs, scale=velocity_scale, time=duration_time)

                # blocking call
                request = 'test'
                while request != '':
                    request = input("Once the robot has been reset to the desired joint configuration, please press "
                                    "Enter to move on with the code.")

                    # double check that the robot is at the specified joint configuration and if not, ask the user
                    # to confirm to proceed with the code

            except rospy.ServiceException as e:
                print(self.reset_joint_service_name + " service call failed: " + str(e))

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
            positions = self.subscriber.get_joint_positions(q_indices)
            return positions

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
            q = self.subscriber.get_joint_positions()
            dq = self.subscriber.get_joint_velocities()
            # dq = self.filter(dq)
            # tau = self.subscriber.get_joint_torques()

            if q is not None and len(q) > 0:
                q_indices = None if joint_ids is None else self.q_indices[joint_ids]
                if q_indices is not None:
                    q[q_indices] = positions[:len(q_indices)]
                    if velocities is not None:
                        dq[q_indices] = velocities[:len(q_indices)]

                self.arm_point.positions = q[:len(q_indices)]
                self.arm_point.velocities = dq[:len(q_indices)]
                # self.arm_point.effort = tau[:len(q_indices)]

                # set time duration
                self.arm_point.time_from_start.secs = 0
                self.arm_point.time_from_start.nsecs = 200000000

                # set message and publish it
                self.arm_publisher.msg.points = [self.arm_point]
                self.arm_publisher.publish()

    def get_joint_velocities(self, joint_ids=None):
        """
        Get the velocity of the given joint(s).

        Args:
            joint_ids (int, list[int], None): joint id, or list of joint ids.

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

    def get_joint_torques(self, joint_ids=None):
        """
        Get the applied torque(s) on the given joint(s). "This is the motor torque applied during the last `step`.
        Note that this only applies in VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the
        applied joint motor torque is exactly what you provide, so there is no need to report it separately." [1]

        Args:
            joint_ids (int, list[int], None): a joint id, or list of joint ids.

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
