#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manipulability tracking for single-arm task.

In this example, the Centauro robot tracks a desired manipulability for its right arm, while reaching a desired
end-effector position and maintaining balance.

See Also:
    - `com_manipulability_tracking_with_balance.py`: in this example, we track the velocity manipulability ellipsoid
        while keeping the robot balanced.
    - `com_dynamic_manipulability_tracking_with_balance.py`: in this example, the dynamic manipulability ellipsoid is
        tracked instead of the velocity one.
    - `com_dynamic_manipulability_tracking_with_balance.py`: in this example, the dynamic manipulability ellipsoid is
        tracked instead of the velocity one.

References:
    [1] "Robotics: Modelling, Planning and Control" (section 3.9), Siciliano et al., 2010
    [2] "Geometry-aware Tracking of Manipulability Ellipsoids", Jaquier et al., R:SS, 2018
"""

import time
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Centauro

# program variable
robot_name = 'centauro'
dt = 0.01  # Sampling time

# Create simulator and world
sim = Bullet()
world = BasicWorld(sim)

# load robot
robot = Centauro(sim, fixed_base=False)

# Define desired velocity manipulability, proportional gain, and initial configurations
# Desired right-arm manipulability
des_vel_manip = np.array([[0.0977, -0.0778, 0.1339], [-0.0778, 0.5102, 0.0300], [0.1339, 0.0300, 0.3130]])

# Get ids for "feet" (used for kinematics function)
left_foot1_id = robot.get_link_ids('wheel_1')
right_foot1_id = robot.get_link_ids('wheel_2')
left_foot2_id = robot.get_link_ids('wheel_3')
right_foot2_id = robot.get_link_ids('wheel_4')

# Get ids for arms end-effectors (used for kinematics function)
left_arm_ef_id = robot.get_link_ids('arm1_8')
right_arm_ef_id = robot.get_link_ids('arm2_8')

left_arm_joints_ids = [robot.joints.index(i) for i in (robot.left_arm)]
right_arm_joints_ids = [robot.joints.index(i) for i in (robot.right_arm)]

# Gain matrices balancing
Kcom = np.diag((250.0, 250.0, 0.0))  # Proportional gain for CoM position control
Kl1f = np.diag((180, 180, 180))  # Proportional gain for foot position control
Kr1f = np.diag((180, 180, 180))  # Proportional gain for foot position control
Kl2f = np.diag((180, 180, 180))  # Proportional gain for foot position control
Kr2f = np.diag((180, 180, 180))  # Proportional gain for foot position control

# Gain matrix arm position
# Kra = 50. * np.eye(3)  # Proportional gain for arm position control
Kra = 200. * np.eye(3)  # Proportional gain for arm position control

# Gain matrix manipulability
Km = 5. * np.eye(6)  # Proportional gain for Centauro

# Setting initial configuration of the robot
q0 = [0., 0., 0.0003, -0.0003,
      -0.5561, -0.5164, 0.4859, -0.4005, -0.0003, 0.3723, 0.0009, 0.0,
      .3, 1.3, -.68, .7,  # -0.0002, -0.4849, 0.0003, 0.0,
      -0.0001, -0.0370, -0.0008, 0.0003, 0., 0.0004,
      -0.0005, 0.0388, -0.0016, 0.0010, 0.0001, -0.0053,
      -0.0020, -0.0358, -0.0116, -0.0032, 0.0036, -0.0052,
      -0.0012, 0.0343, 0.0001, -0.0073, 0.0013, -0.0016]
q0id = ['torso_yaw', 'neck_yaw', 'neck_pitch', 'neck_velodyne',
        'j_arm1_1', 'j_arm1_2', 'j_arm1_3', 'j_arm1_4', 'j_arm1_5', 'j_arm1_6', 'j_arm1_7', 'j_ft_1',
        'j_arm2_1', 'j_arm2_2', 'j_arm2_3', 'j_arm2_4', #'j_arm2_5', 'j_arm2_6', 'j_arm2_7', 'j_ft_2',
        'hip_yaw_1', 'hip_pitch_1', 'knee_pitch_1', 'ankle_pitch_1', 'ankle_yaw_1', 'j_wheel_1',
        'hip_yaw_2', 'hip_pitch_2', 'knee_pitch_2', 'ankle_pitch_2', 'ankle_yaw_2', 'j_wheel_2',
        'hip_yaw_3', 'hip_pitch_3', 'knee_pitch_3', 'ankle_pitch_3', 'ankle_yaw_3', 'j_wheel_3',
        'hip_yaw_4', 'hip_pitch_4', 'knee_pitch_4', 'ankle_pitch_4', 'ankle_yaw_4', 'j_wheel_4']

# Load robot in world
robot = world.load_robot(robot)
world.step()

# Loop need to set the robot initial posture
if not (isinstance(q0, int) and q0 == 0 and isinstance(q0id, int) and q0id == 0):
    for n in range(45):
        # robot.set_joint_positions(np.asarray(q0))
        robot.set_joint_positions(np.asarray(q0), robot.get_joint_ids(np.asarray(q0id)))
        world.step()

# Loop for setting stable initial conditions
for i in range(50):
    world.step(sleep_dt=0.1)

# Augmented gain matrix for balancing controller
Kbal = block_diag(Kl1f, Kr1f, Kl2f, Kr2f, Kcom)
print("Kbal: {}".format(Kbal))


# Initial conditions
time.sleep(4.0)
num_dofs = robot.num_dofs - 6
CoMr = robot.get_center_of_mass_position()  # Desired CoM
print("CoMr: {}".format(CoMr))

# Desired feet positions
xref_l1f = robot.get_link_world_frame_positions(left_foot1_id)  # Desired position for left foot
xref_r1f = robot.get_link_world_frame_positions(right_foot1_id)  # Desired position for right foot
xref_l2f = robot.get_link_world_frame_positions(left_foot2_id)  # Desired position for left foot
xref_r2f = robot.get_link_world_frame_positions(right_foot2_id)  # Desired position for right foot

# Desired arms positions
# xref_la = robot.get_link_world_frame_positions(left_arm_ef_id)  # Desired position for left arm
# xref_ra = robot.get_link_world_frame_positions(right_arm_ef_id)  # Desired position for right arm
x_torso = robot.get_link_frames(robot.get_link_ids('torso_2'))[0][0]
xref_ra = np.array([1.0, -0.05, 1.15])

# Display initial and desired manipulability ellipsoid
q0 = robot.get_joint_positions()
print("q0: {}".format(q0))
Jra = robot.get_jacobian(right_arm_ef_id, q0)
if not robot.has_fixed_base():
    vel_manip = robot.compute_velocity_manipulability_ellipsoid(Jra[:, 6:])
else:
    vel_manip = robot.compute_velocity_manipulability_ellipsoid(Jra)
print("Md: {}".format(des_vel_manip[0:3, 0:3]))
print("Mv0: {}".format(vel_manip[0:3, 0:3]))

base_pos = robot.get_base_position()
# robot.draw_velocity_manipulability_ellipsoid(link_id=right_arm_ef_id, JJT=0.1 *des_vel_manip, color=(0.1, 0.75, 0.1, 0.6))
# ellipsoid_id = robot.draw_velocity_manipulability_ellipsoid(link_id=right_arm_ef_id, JJT=0.1 * vel_manip[0:3, 0:3],
#                                                             color=(0.75, 0.1, 0.1, 0.6))

# Logging variables
# Format: [q minEigvalue(Jbal) minEigvalue(Jman) balanceError rightArmError CurrentManip(1x9) DesManip(1x9) SPDdistance]
log_array = np.zeros((300, num_dofs + 2 + Kbal.shape[0] + Kra.shape[0] + vel_manip[0:3, 0:3].size + des_vel_manip.size + 1))

# Plot x_ref_ra
xref_visual_shape = robot.sim.create_visual_shape(robot.sim.GEOM_SPHERE, radius=0.03, rgba_color=(1, 0, 0, 0.8))
xref_visual = robot.sim.create_body(mass=0, visual_shape_id=xref_visual_shape, position=xref_ra)

# Run simulator
for i in range(300):
    # Update current robot state
    qt = robot.get_joint_positions()
    CoMt = robot.get_center_of_mass_position()  # Current CoM
    robot.draw_com_position(0.03, color=(0, 0.5, 0, 0.8))

    # Current feet positions
    xt_l1f = robot.get_link_world_frame_positions(left_foot1_id)  # Current position for left foot
    xt_r1f = robot.get_link_world_frame_positions(right_foot1_id)  # Current position for right foot
    xt_l2f = robot.get_link_world_frame_positions(left_foot2_id)  # Current position for left foot
    xt_r2f = robot.get_link_world_frame_positions(right_foot2_id)  # Current position for right foot

    # Current arm positions
    xt_la = robot.get_link_world_frame_positions(left_arm_ef_id)  # Current left arm pos
    xt_ra = robot.get_link_world_frame_positions(right_arm_ef_id)  # Current right arm pos

    # Simple balance control with IK kinematics for CoM and feet
    # Get Jacobians: Jcom, Jlf, and Jrf
    Jcom = robot.get_center_of_mass_jacobian(qt)

    # Weight joint influences in Jcom
    com_weights = np.eye(Jcom.shape[1])
    left_arm_weight_idx = [i + 6 for i in left_arm_joints_ids]
    right_arm_weight_idx = [i + 6 for i in right_arm_joints_ids]
    com_weights[left_arm_weight_idx, left_arm_weight_idx] = 0.01
    com_weights[right_arm_weight_idx, right_arm_weight_idx] = 0.01
    Jcom = np.dot(Jcom, com_weights)

    # Feet Jacobians
    Jl1f = robot.get_jacobian(left_foot1_id, qt)
    Jr1f = robot.get_jacobian(right_foot1_id, qt)
    Jl2f = robot.get_jacobian(left_foot2_id, qt)
    Jr2f = robot.get_jacobian(right_foot2_id, qt)

    # Compose Jacobian and nullspace for balancing task
    Jbal = np.vstack((Jl1f[0:3, ], Jr1f[0:3, ], Jl2f[0:3, ], Jr2f[0:3, ], Jcom[0:3, ]))
  
    Ubal, Sbal, VhBal = np.linalg.svd(Jbal)
    if np.min(Sbal) < 4.5E-2:
        pJbal = robot.get_damped_least_squares_inverse(Jbal, 4.5E-2)
    else:
        pJbal = robot.get_damped_least_squares_inverse(Jbal, 1E-8)
    Nbal = np.eye(Jbal.shape[1]) - np.dot(pJbal, Jbal)

    # Compute balancing task errors
    dx_com = CoMr - CoMt  # CoM error

    dx_l1f = xref_l1f - xt_l1f  # Left foot position error
    dx_r1f = xref_r1f - xt_r1f  # Right foot position error
    dx_l2f = xref_l2f - xt_l2f  # Left foot position error
    dx_r2f = xref_r2f - xt_r2f  # Right foot position error
    dx_bal = np.vstack((dx_l1f.reshape(3, 1), dx_r1f.reshape(3, 1),
                        dx_l2f.reshape(3, 1), dx_r2f.reshape(3, 1), dx_com.reshape(3, 1)))  # Augmented error vector

    # print(dx_bal.T)

    # Proportional controller for position
    dxref_bal = np.dot(Kbal, dx_bal)

    # Compute desired joint velocities for balancing
    dq_bal = np.dot(pJbal, dxref_bal)
    dq_bal = dq_bal.reshape((Jbal.shape[1],))

    # Compute right arm position task error
    dx_ra = xref_ra - xt_ra
    # print(xt_ra)

    # Compose Jacobian and nullspace for position task
    Jra = robot.get_jacobian(right_arm_ef_id, qt)
    Jra_pos = Jra[0:3, ]
    Ura, Sra, Vhra = np.linalg.svd(Jra_pos)
    if np.min(Sra) < 4.5E-2:
        pJra = robot.get_damped_least_squares_inverse(Jra_pos, 4.5E-2)
    else:
        pJra = robot.get_damped_least_squares_inverse(Jra_pos, 1E-8)
    Nra = np.eye(Jra_pos.shape[1]) - np.dot(pJra, Jra_pos)

    # Proportional controller for position
    dxref_ra = np.dot(Kra, dx_ra)

    # Compute desired joint velocities
    dq_ra = np.dot(pJra, dxref_ra)

    # Tracking of right arm velocity manipulability in nullspace
    # Compute velocity manipulability (right arm)
    if not robot.has_fixed_base():
        vel_manip = robot.compute_velocity_manipulability_ellipsoid(Jra[:, 6:])
    else:
        vel_manip = robot.compute_velocity_manipulability_ellipsoid(Jra)

    # print(vel_manip[0:3, 0:3])

    # Plot current manipulability ellipsoid
    # if i % 40 == 0:
    #     ellipsoid_id = robot.update_manipulability_ellipsoid(link_id=right_arm_ef_id, ellipsoid_id=ellipsoid_id,
    #                                                          ellipsoid=0.1 * vel_manip[:3, :3],
    #                                                          color=(0.75, 0.1, 0.1, 0.6))

    # Compute (position) manipulability Jacobian and nullspace
    Jman_red = robot.compute_velocity_manipulability_jacobian(Jra, 3)
    Uman, Sman, Vhman = np.linalg.svd(Jman_red)
    if np.min(Sra) < 4.5E-2:
        pJman_red = robot.get_damped_least_squares_inverse(Jman_red, 4.5E-2)
    else:
        pJman_red = robot.get_damped_least_squares_inverse(Jman_red, 1E-8)
    Nman = np.eye(Jman_red.shape[1]) - np.dot(pJman_red, Jman_red)

    # Obtaining joint velocity command
    if not robot.has_fixed_base():
        dq_man, minSman, SPDdist = robot.calculate_inverse_differential_kinematics_velocity_manipulability(Jra[:, 6:],
                                                                                                           des_vel_manip,
                                                                                                           Km)
        # dq_man = np.vstack((np.zeros((6, 1)), dq_man.reshape(num_dofs, 1)))
    else:
        dq_man, minSman, SPDdist = robot.calculate_inverse_differential_kinematics_velocity_manipulability(Jra, des_vel_manip, Km)

    # Logging
    # Format: [q minEigvalue(Jbal) minEigvalue(Jman) balanceError rightArmError CurrentManip(1x9) DesManip(1x9)
    # SPDdistance]
    log_array[i,] = np.hstack((qt.reshape(1, num_dofs), np.min(Sbal).reshape(1, 1),
                               minSman.reshape(1, 1), dx_bal.T, dx_ra[None],
                               vel_manip[0:3, 0:3].reshape(1, vel_manip[0:3, 0:3].size),
                               des_vel_manip.reshape(1, des_vel_manip.size), SPDdist.reshape(1, 1)))

    # Set joint position
    if not robot.has_fixed_base():
        # dq_ra = np.concatenate((np.zeros((6,)), dq_ra))
        dq_man = np.concatenate((np.zeros((6,)), dq_man))
        # Task priority: balance > right arm position > right arm manipulability
        dq = dq_bal + np.dot(Nbal, dq_ra) + np.dot(Nbal, np.dot(Nra, dq_man))
        # Task priority: balance > right arm manipulability > right arm position
        # dq = dq_bal + np.dot(Nbal, dq_man) + np.dot(Nbal, np.dot(Nman, dq_ra))
        # dq = dq_bal + np.dot(Nbal, dq_man)
        dq = dq[6:, ]
    else:
        dq = dq_bal + np.dot(Nbal, dq_ra) + np.dot(Nbal, np.dot(Nra, dq_man))

    q = qt + (dq * dt)
    robot.set_joint_positions(q)

    world.step(sleep_dt=dt)

# Print final manipulability
print("Mvfinal: {}".format(vel_manip[0:3, 0:3]))

# Saving log data
np.savetxt(robot.name + 'log_Man.csv', log_array, delimiter=',')

# Plotting logged data
plt.rcParams.update({'font.size': 14})
fig1 = plt.figure(1, figsize=(14, 10))
# wspace: width reserved for blank space between subplots, hspace: height reserved for white space between subplots
fig1.subplots_adjust(left=0.09, bottom=0.05, right=0.99, wspace=0.2)
plt.suptitle('Robot joints')
for i in range(num_dofs):
    plt.subplot(7, 7, i + 1)
    plt.ylabel(robot.get_joint_names(robot.get_joint_ids(i)))
    plt.plot(log_array[:, i])
    plt.ylim((-1.5, 1.5))

fig2 = plt.figure(2, figsize=(14, 10))
fig2.subplots_adjust(left=0.09, bottom=0.05, right=0.99, wspace=0.2)
plt.suptitle('MinEigenvalues, balance and manipulability errors.')
plt.subplot(6, 3, 1)
plt.ylim((0, 0.02))

plt.ylabel('minEigvalue(Jbal)')
plt.plot(log_array[:, num_dofs])

plt.subplot(6, 3, 2)
plt.ylim((0., 0.001))

plt.ylabel('minEigvalue(Jman)')
plt.plot(log_array[:, num_dofs + 1])

plt.subplot(6, 3, 3)
plt.ylim((0., 5.0))

plt.ylabel('SPDdist')
plt.plot(log_array[:, -1])

for i in range(dx_bal.shape[0]):
    plt.subplot(6, 3, i + 4)

    plt.ylabel('dx_bal'+str(i+1), fontsize=16)
    plt.plot(log_array[:, num_dofs + 2 + i])
    plt.ylim((-.05, .05))


fig3 = plt.figure(3, figsize=(14, 10))
for i in range(dx_ra.shape[0]):
    plt.subplot(4, 1, i+1)
    plt.ylabel('dx_ra'+str(i+1), fontsize=16)
    plt.plot(log_array[:, num_dofs + 2 + 15 + i])
plt.subplot(4, 1, 4)
plt.ylabel('Manip. error')
plt.plot(log_array[:, -1])

plt.show()
