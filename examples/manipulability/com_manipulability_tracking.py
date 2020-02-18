#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Center of mass manipulability tracking

Track the velocity manipulability ellipsoid of the center of mass of a particular robot. In this example, the robot
base is fixed.

See Also:
    - `com_manipulability_tracking_with_balance.py`: in this example, we track the velocity manipulability ellipsoid
        while keeping the robot balanced.
    - `com_dynamic_manipulability_tracking_with_balance.py`: in this example, the dynamic manipulability ellipsoid is
        tracked instead of the velocity one.

References:
    [1] "Robotics: Modelling, Planning and Control" (section 3.9), Siciliano et al., 2010
    [2] "Geometry-aware Tracking of Manipulability Ellipsoids", Jaquier et al., R:SS, 2018
"""

from itertools import count
import numpy as np
import argparse

from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Centauro, Cogimon, Nao, KukaIIWA


# create parser to select the robot to use
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', help='the robot to track the velocity manipulability ellipsoid', type=str,
                    choices=['nao', 'kuka_iiwa', 'cogimon', 'centauro'], default='nao')
args = parser.parse_args()

# get the robot to use for the example
robot_name = args.robot
dt = 0.01


# Create simulator and world
sim = Bullet()
world = BasicWorld(sim)

# Define robot, velocity manipulability for CoM, and proportional gain
if robot_name == 'kuka_iiwa':
    # load robot
    robot = KukaIIWA(sim)

    # desired Velocity Manipulability for CoM
    desired_velocity_manip = np.array([[0.02193921, -0.01192746, -0.0155832],
                             [-0.01192746, 0.04524443, 0.01892251],
                             [-0.0155832, 0.01892251, 0.02259072]])

    # proportional gain
    Km = 5 * np.eye(6)

elif robot_name == 'nao':
    # load robot
    robot = Nao(sim, fixed_base=True)

    # # desired Velocity Manipulability for CoM
    # desired_velocity_manip = np.array([[2.53907684e-03, 2.65373209e-04, 1.83354058e-04],
    #                          [2.65373209e-04, 2.08082139e-03, -5.84361397e-05],
    #                          [1.83354058e-04, -5.84361397e-05, 8.60170044e-04]])
    desired_velocity_manip = np.array([[1.0e-03, 0.0, 0.0],
                             [0.0, 3.0e-03, 0.0],
                             [0.0, 0.0, 1.0e-04]])

    # proportional gain
    Km = 500 * np.eye(6)

elif robot_name == 'cogimon':
    # load robot
    robot = Cogimon(sim, fixed_base=True)

    # # desired Velocity Manipulability for CoM
    # desired_velocity_manip = np.array([[0.02045482, 0.00356193, 0.00081623],
    #                          [0.00356193, 0.04693186, 0.01798659],
    #                          [0.00081623, 0.01798659, 0.01636723]])
    desired_velocity_manip = np.array([[0.06, 0.0, 0.0],
                             [0.0, 0.01, 0.0],
                             [0.0, 0.0, 0.005]])

    # proportional gain
    Km = 500 * np.eye(6)

elif robot_name == 'centauro':
    # load robot
    robot = Centauro(sim, fixed_base=True)

    # desired Velocity Manipulability for CoM
    desired_velocity_manip = np.array([[0.01, 0.0, 0.0],
                            [0.0, 0.04, 0.0],
                            [0.0, 0.0, 0.005]])

    # proportional gain
    Km = 200 * np.eye(6)

else:
    raise NotImplementedError("The given robot has not been implemented")

# Initial conditions for visualization
# Display initial and desired manipulability ellipsoid
robot = world.load_robot(robot)
world.step()
q0 = robot.get_joint_positions()
Jcom = robot.get_center_of_mass_jacobian(q0)
velocity_manip = robot.compute_velocity_manipulability_ellipsoid(Jcom)
robot.draw_velocity_manipulability_ellipsoid(link_id=-1, JJT=10*desired_velocity_manip, color=(0.1, 0.75, 0.1, 0.6))
ellipsoid_id = robot.draw_velocity_manipulability_ellipsoid(link_id=-1, JJT=10 * velocity_manip[0:3, 0:3],
                                                            color=(0.75, 0.1, 0.1, 0.6))

# Run simulator
for i in count():
    robot.compute_and_draw_com_position(radius=0.03)
    print("CoM: {}".format(robot.get_center_of_mass_position()))

    # get current joint position
    qt = robot.get_joint_positions()

    # get center of mass jacobian
    Jcom = robot.get_center_of_mass_jacobian(qt)
    print("CoM velocity: {}".format(robot.get_center_of_mass_velocity()))
    print("Jcom.dot(qt) = {}".format(Jcom.dot(robot.get_joint_velocities())))

    # Get Inertia Matrix
    # M = robot.get_mass_matrix(q=qt)
    # print("M: {}".format(M))

    # Tracking of CoM velocity manipulability
    velocity_manip = robot.compute_velocity_manipulability_ellipsoid(Jcom)
    print("Mv: {}".format(velocity_manip[0:3, 0:3]))

    # Plot current manipulability ellipsoid
    if i % 10 == 0:
        ellipsoid_id = robot.update_manipulability_ellipsoid(link_id=-1, ellipsoid_id=ellipsoid_id,
                                                             ellipsoid=10 * velocity_manip[:3, :3],
                                                             color=(0.75, 0.1, 0.1, 0.6))

    # Obtaining joint velocity command
    dq = robot.calculate_inverse_differential_kinematics_velocity_manipulability(Jcom, desired_velocity_manip, Km)[0]

    # Set joint position
    q = qt + dq * dt
    robot.set_joint_positions(q)

    world.step(sleep_dt=dt)
