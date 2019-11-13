#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Center of mass manipulability tracking

Track the velocity manipulability ellipsoid of the center of mass of a particular robot while keeping its balance.

See Also:
    - `com_manipulability_tracking.py`: in this example, the base is fixed.
    - `com_dynamic_manipulability_tracking_with_balance.py`: in this example, the dynamic manipulability ellipsoid is
        tracked instead of the velocity one.

References:
    [1] "Robotics: Modelling, Planning and Control" (section 3.9), Siciliano et al., 2010
    [2] "Geometry-aware Tracking of Manipulability Ellipsoids", Jaquier et al., R:SS, 2018
"""

import time
# from itertools import count
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import argparse

from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Nao, Centauro


# create parser
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', help='the robot to track the velocity manipulability ellipsoid', type=str,
                    choices=['nao', 'centauro'], default='nao')
parser.add_argument('-n', '--init_manipulability', help='which desired velocity manipulability we want to track',
                    type=int, default=4)
parser.add_argument('-i', '--init_config', help='the initial configuration for the robot', type=int,
                    choices=range(1, 5), default=1)
args = parser.parse_args()

# program variable
robot_name = args.robot
dt = 0.01  # Sampling time
num_des_manip = args.init_manipulability   # Number of desired manipulability (Useful for tests)
init_qs = args.init_config                 # Number of initial configuration for the robots (Useful for tests)


# Create simulator and world
sim = Bullet()
world = BasicWorld(sim)

# load robot
if robot_name == 'nao':
    robot = Nao(sim, fixed_base=False)
elif robot_name == 'centauro':
    robot = Centauro(sim, fixed_base=False)
else:
    raise NotImplementedError("The given robot has not been implemented")

# Loop for setting stable initial conditions
for i in range(50):
    world.step(sleep_dt=0.1)

# Define velocity manipulability for CoM, proportional gain, and initial configurations
if robot.name == 'nao':
    Km = 100 * np.eye(6)  # Proportional gain for Nao for manip. tracking
  
    # desired Velocity Manipulability for CoM (Nao)
    des_manips = np.array([[[1.539e-03, 6.653e-04, 0.833e-04],
                            [6.653e-04, 2.080e-03, -5.843e-05],
                            [0.833e-04, -5.843e-05, 8.601e-04]],
                           [[2.580e-03, 1.653e-03, 4.833e-04],
                            [1.653e-03, 1.539e-03, -5.843e-05],
                            [4.833e-04, -5.843e-05, 9.601e-04]],
                           [[1.580e-03, .653e-03, 4.833e-04],
                            [.653e-03, 1.539e-03, -5.843e-05],
                            [4.833e-04, -5.843e-05, 9.601e-04]],
                           [[1e-04, 0.0, 0.0],
                            [0.0, 5e-04, 0.0],
                            [0.0, 0.0, 1e-04]]])
    des_vel_manip = des_manips[num_des_manip - 1, :, :]
  
    # Get ids for feet (used for kinematics function)
    left_foot_id = robot.get_link_ids('l_ankle')
    right_foot_id = robot.get_link_ids('r_ankle')
  
    # Gain matrices
    Kcom = np.diag((50, 40, 0))  # Proportional gain for CoM position control
    Klf = np.diag((20, 20, 20))   # Proportional gain for foot position control
    Krf = np.diag((20, 20, 20))   # Proportional gain for foot position control
  
    # Setting initial configuration of the robot
    if init_qs == 1:
        q0 = [1.55, 0.135, -1.05, -0.36, 1.55, -0.135, 1.05, 0.36]
        q0id = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll',
                'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
    elif init_qs == 2:
        q0 = [1.55, 0.135, -1.05, -0.36, 1.55, -0.135, 1.05, 0.36, 0.1, -0.1, 0.12, 0.12, -0.12, -0.12]
        q0id = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'RShoulderPitch', 'RShoulderRoll',
                'RElbowYaw', 'RElbowRoll', 'LHipRoll', 'RHipRoll', 'LKneePitch', 'RKneePitch', 'LAnklePitch', 
                'RAnklePitch']
    elif init_qs == 3:
        q0 = [2.0, 0.4, -1.3, -1.2, 2.0, -0.4, 1.3, 1.2, 0.05, -0.05, 0.2, 0.2, -0.2, -0.2]
        q0id = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'RShoulderPitch', 'RShoulderRoll',
                'RElbowYaw', 'RElbowRoll', 'LHipRoll', 'RHipRoll', 'LKneePitch', 'RKneePitch', 'LAnklePitch', 
                'RAnklePitch']
    else:
        q0 = 0
        q0id = 0

elif robot.name == 'centauro':
    Km = 50 * np.eye(6)  # Proportional gain for Centauro
    # desired Velocity Manipulability for CoM (Centauro)
    des_manips = np.array([[[0.0207, 0.008, 0.0],
                            [0.008, 0.01, -0.005],
                            [0.0, -0.005, 0.006]],
                           [[5.173e-03, -2.733e-03, 1.920e-03],
                            [-2.733e-03, 2.038e-02, 7.185e-04],
                            [1.920e-03, 7.185e-04, 2.107e-03]],
                           [[2.2e-02, 0.0, -0.01],
                            [0.0, 2.1e-02, 0.0],
                            [-0.01, 0.0, 5.e-03]],
                           [[5e-04, 0.0, 0.0],
                            [0.0, .1, 0.0],
                            [0.0, 0.0, 5e-04]]])
    des_vel_manip = des_manips[num_des_manip - 1, :, :]
    print("desired velocity manipulability: {}".format(des_vel_manip))
  
    # Get ids for "feet" (used for kinematics function)
    left_foot1_id = robot.get_link_ids('wheel_1')
    right_foot1_id = robot.get_link_ids('wheel_2')
    left_foot2_id = robot.get_link_ids('wheel_3')
    right_foot2_id = robot.get_link_ids('wheel_4')
  
    # Gain matrices
    Kcom = np.diag((250.0, 250.0, 0.0))  # Proportional gain for CoM position control
    Kl1f = np.diag((180, 180, 180))  # Proportional gain for foot position control
    Kr1f = np.diag((180, 180, 180))  # Proportional gain for foot position control
    Kl2f = np.diag((180, 180, 180))  # Proportional gain for foot position control
    Kr2f = np.diag((180, 180, 180))  # Proportional gain for foot position control
  
    # Setting initial configuration of the robot
    if init_qs == 1:
        q0 = [-.7, -.65, .61, -.5, .71, .64, -.68, .7]
        q0id = ['j_arm1_1', 'j_arm1_2', 'j_arm1_3', 'j_arm1_4', 'j_arm2_1', 'j_arm2_2', 'j_arm2_3', 'j_arm2_4']
    elif init_qs == 2:
        q0 = [-.7, -.65, .61, -.5, .71, .64, -.68, .7, -0.42, -0.96, -0.59,
              0.42, 0.96, 0.59, 0.42, 0.96, 0.59, -0.42, -0.96, -0.59]
        q0id = ['j_arm1_1', 'j_arm1_2', 'j_arm1_3', 'j_arm1_4', 'j_arm2_1', 'j_arm2_2', 'j_arm2_3', 'j_arm2_4',
                'hip_pitch_1', 'knee_pitch_1', 'ankle_pitch_1', 'hip_pitch_2', 'knee_pitch_2', 'ankle_pitch_2',
                'hip_pitch_3', 'knee_pitch_3', 'ankle_pitch_3', 'hip_pitch_4', 'knee_pitch_4', 'ankle_pitch_4']
    elif init_qs == 3:
        q0 = [-.3, -1.3, .61, -.5, .3, 1.3, -.68, .7, -0.42, -0.96, -0.59,
              0.42, 0.96, 0.59, 0.42, 0.96, 0.59, -0.42, -0.96, -0.59]
        q0id = ['j_arm1_1', 'j_arm1_2', 'j_arm1_3', 'j_arm1_4', 'j_arm2_1', 'j_arm2_2', 'j_arm2_3', 'j_arm2_4',
                'hip_pitch_1', 'knee_pitch_1', 'ankle_pitch_1', 'hip_pitch_2', 'knee_pitch_2', 'ankle_pitch_2',
                'hip_pitch_3', 'knee_pitch_3', 'ankle_pitch_3', 'hip_pitch_4', 'knee_pitch_4', 'ankle_pitch_4']
    else:
        q0 = 0
        q0id = 0

else:
    left_foot_id, right_foot_id = 0, 0

# Loop need to set the robot initial posture
if not (isinstance(q0, int) and q0 == 0 and isinstance(q0id, int) and q0id == 0):
    for n in range(15):
        robot.set_joint_positions(np.asarray(q0), robot.get_joint_ids(np.asarray(q0id)))
        world.step()

# Augmented gain matrix for balancing controller
if robot.name == 'centauro':
    Kbal = block_diag(Kl1f, Kr1f, Kl2f, Kr2f, Kcom)
else:
    Kbal = block_diag(Klf, Krf, Kcom)
print("Kbal: {}".format(Kbal))


# Initial conditions
time.sleep(2.0)

num_dofs = robot.num_dofs - 6
CoMr = robot.get_center_of_mass_position()  # Desired CoM
print("CoMr: {}".format(CoMr))

if robot.name == 'centauro':
    xref_l1f = robot.get_link_world_frame_positions(left_foot1_id)  # Desired position for left foot
    xref_r1f = robot.get_link_world_frame_positions(right_foot1_id)  # Desired position for right foot
    xref_l2f = robot.get_link_world_frame_positions(left_foot2_id)  # Desired position for left foot
    xref_r2f = robot.get_link_world_frame_positions(right_foot2_id)  # Desired position for right foot
else:
    xref_lf = robot.get_link_world_frame_positions(left_foot_id)  # Desired position for left foot
    # Qref_lf = robot.get_link_world_frame_orientations(leftFootId)
    xref_rf = robot.get_link_world_frame_positions(right_foot_id)  # Desired position for right foot
    # Qref_rf = robot.get_link_world_frame_orientations(rightFootId)


# Display initial and desired manipulability ellipsoid
q0 = robot.get_joint_positions()
print("q0: {}".format(q0))
Jcom0 = robot.get_center_of_mass_jacobian(q0)
if not robot.has_fixed_base():
    vel_manip = robot.compute_velocity_manipulability_ellipsoid(Jcom0[:, 6:])
else:
    vel_manip = robot.compute_velocity_manipulability_ellipsoid(Jcom0)
print("Mv0: {}".format(vel_manip[0:3, 0:3]))

base_pos = robot.get_base_position()
robot.draw_velocity_manipulability_ellipsoid(link_id=-1, JJT=10 * des_vel_manip, color=(0.1, 0.75, 0.1, 0.6))
ellipsoid_id = robot.draw_velocity_manipulability_ellipsoid(link_id=-1, JJT=10 * vel_manip[0:3, 0:3],
                                                            color=(0.75, 0.1, 0.1, 0.6))

# Logging variables
# Format: [q minEigvalue(Jbal) minEigvalue(Jman) balanceError CurrentManip(1x9) SPDdistance]
log_array = np.zeros((400, num_dofs + 2 + Kbal.shape[0] + vel_manip[0:3, 0:3].size + 1))


# Run simulator
# for i in count():
for i in range(400):
    # Update current robot state
    qt = robot.get_joint_positions()
    CoMt = robot.get_center_of_mass_position()  # Current CoM
    robot.draw_com_position(0.03)
  
    if robot.name == 'centauro':
        xt_l1f = robot.get_link_world_frame_positions(left_foot1_id)  # Current position for left foot
        xt_r1f = robot.get_link_world_frame_positions(right_foot1_id)  # Current position for right foot
        xt_l2f = robot.get_link_world_frame_positions(left_foot2_id)  # Current position for left foot
        xt_r2f = robot.get_link_world_frame_positions(right_foot2_id)  # Current position for right foot
    else:
        xt_lf = robot.get_link_world_frame_positions(left_foot_id)  # Current left foot pos
        # Qt_lf = robot.get_link_world_frame_orientations(leftFootId)
        xt_rf = robot.get_link_world_frame_positions(right_foot_id)  # Current right foot pos

    # Simple balance control with IK kinematics for CoM and feet
    # Get Jacobians: Jcom, Jlf, and Jrf
    Jcom = robot.get_center_of_mass_jacobian(qt)

    if robot.name == 'centauro':
        Jl1f = robot.get_jacobian(left_foot1_id, qt)
        Jr1f = robot.get_jacobian(right_foot1_id, qt)
        Jl2f = robot.get_jacobian(left_foot2_id, qt)
        Jr2f = robot.get_jacobian(right_foot2_id, qt)
    else:
        Jlf = robot.get_jacobian(left_foot_id, qt)
        Jrf = robot.get_jacobian(right_foot_id, qt)

    # Compose Jacobian and nullspace for balancing task
    if robot.name == 'centauro':
        Jbal = np.vstack((Jl1f[0:3, ], Jr1f[0:3, ], Jl2f[0:3, ], Jr2f[0:3, ], Jcom[0:3, ]))
    else:
        Jbal = np.vstack((Jlf[0:3, ], Jrf[0:3, ], Jcom[0:3, ]))
  
    Ubal, Sbal, VhBal = np.linalg.svd(Jbal)
    if np.min(Sbal) < 4.5E-2:
        pJbal = robot.get_damped_least_squares_inverse(Jbal, 4.5E-2)
    else:
        pJbal = robot.get_damped_least_squares_inverse(Jbal, 1E-8)
    Nbal = np.eye(Jbal.shape[1]) - np.dot(pJbal, Jbal)

    # Compute balancing task errors
    dx_com = CoMr - CoMt  # CoM error

    if robot.name == 'centauro':
        dx_l1f = xref_l1f - xt_l1f  # Left foot position error
        dx_r1f = xref_r1f - xt_r1f  # Right foot position error
        dx_l2f = xref_l2f - xt_l2f  # Left foot position error
        dx_r2f = xref_r2f - xt_r2f  # Right foot position error
        dx_bal = np.vstack((dx_l1f.reshape(3, 1), dx_r1f.reshape(3, 1),
                            dx_l2f.reshape(3, 1), dx_r2f.reshape(3, 1), dx_com.reshape(3, 1)))  # Augmented error vector
    else:
        dx_lf = xref_lf - xt_lf  # Left foot position error
        dx_rf = xref_rf - xt_rf  # Right foot position error
        dx_bal = np.vstack((dx_lf.reshape(3, 1), dx_rf.reshape(3, 1), dx_com.reshape(3, 1)))  # Augmented error vector
        # dx_bal = np.vstack((np.zeros((6, 1)), dx_com.reshape(3, 1)))  # Augmented error vector

    # Proportional controller for position
    dxref_bal = np.dot(Kbal, dx_bal)

    # Compute desired joint velocities for balancing
    dq_bal = np.dot(pJbal, dxref_bal)
    dq_bal = dq_bal.reshape((Jbal.shape[1],))

    # Tracking of CoM velocity manipulability in nullspace
    if not robot.has_fixed_base():
        vel_manip = robot.compute_velocity_manipulability_ellipsoid(Jcom[:, 6:])
    else:
        vel_manip = robot.compute_velocity_manipulability_ellipsoid(Jcom)

    # Plot current manipulability ellipsoid
    if i % 40 == 0:
        ellipsoid_id = robot.update_manipulability_ellipsoid(link_id=-1, ellipsoid_id=ellipsoid_id,
                                                             ellipsoid=10 * vel_manip[:3, :3],
                                                             color=(0.75, 0.1, 0.1, 0.6))

    # Obtaining joint velocity command
    if not robot.has_fixed_base():
        dq_man, minSman, SPDdist = robot.calculate_inverse_differential_kinematics_velocity_manipulability(Jcom[:, 6:],
                                                                                                           des_vel_manip,
                                                                                                           Km)
        # dq_man = np.vstack((np.zeros((6, 1)), dq_man.reshape(num_dofs, 1)))
    else:
        dq_man, minSman, SPDdist = robot.calculate_inverse_differential_kinematics_velocity_manipulability(Jcom, des_vel_manip, Km)

    # Logging
    # Format: [q minEigvalue(Jbal) minEigvalue(Jman) balanceError CurrentManip(1x9) SPDdistance]
    log_array[i,] = np.hstack((qt.reshape(1, num_dofs), np.min(Sbal).reshape(1, 1),
                               minSman.reshape(1, 1), dx_bal.T, vel_manip[0:3, 0:3].reshape(1, vel_manip[0:3, 0:3].size),
                               SPDdist.reshape(1, 1)))

    # Set joint position
    if not robot.has_fixed_base():
        dq_man = np.concatenate((np.zeros((6,)), dq_man))
        dq = dq_bal + np.dot(Nbal, dq_man)
        dq = dq[6:, ]
    else:
        dq = dq_bal + np.dot(Nbal, dq_man)

    q = qt + (dq * dt)
    robot.set_joint_positions(q)

    world.step(sleep_dt=dt)


# Saving log data
# np.savetxt(robot.name + 'log_Man' + str(num_des_manip) + 'Pos' + str(init_qs) + '.csv', log_array, delimiter=',')

# Plotting logged data
fig1 = plt.figure(1, figsize=(14, 10))
# wspace: width reserved for blank space between subplots, hspace: height reserved for white space between subplots
fig1.subplots_adjust(left=0.09, bottom=0.05, right=0.99, wspace=0.2)
plt.suptitle('Robot joints')
plt.rcParams.update({'font.size': 8})
for i in range(num_dofs):
    if robot.name == 'nao':
        plt.subplot(6, 7, i+1)  # NAO
    elif robot.name == 'centauro':
        plt.subplot(7, 7, i + 1)  # Centauro

    plt.ylabel(robot.get_joint_names(robot.get_joint_ids(i)))
    plt.plot(log_array[:, i])
    plt.ylim((-1.5, 1.5))
# fig1.savefig(robot.name + '_joints_Man' + str(num_des_manip) + 'Pos' + str(init_qs) + '.png',
#              bbox_inches='tight', dpi=200)

fig2 = plt.figure(2, figsize=(14, 10))
fig2.subplots_adjust(left=0.09, bottom=0.05, right=0.99, wspace=0.2)
plt.rcParams.update({'font.size': 12})
plt.suptitle('MinEigenvalues, balance and manipulatility errors.')
plt.rcParams.update({'font.size': 8})
if robot.name == 'nao':
    plt.subplot(4, 3, 1)  # NAO
    plt.ylim((0, 0.05))  # NAO
elif robot.name == 'centauro':
    plt.subplot(6, 3, 1)  # Centauro
    plt.ylim((0, 0.02))  # Centauro

plt.ylabel('minEigvalue(Jbal)')
plt.plot(log_array[:, num_dofs])

if robot.name == 'nao':
    plt.subplot(4, 3, 2)  # NAO
    plt.ylim((0., 0.0001))  # NAO
elif robot.name == 'centauro':
    plt.subplot(6, 3, 2)  # Centauro
    plt.ylim((0., 0.001))  # Centauro

plt.ylabel('minEigvalue(Jman)')
plt.plot(log_array[:, num_dofs + 1])

if robot.name == 'nao':
    plt.subplot(4, 3, 3)  # NAO
    plt.ylim((0., 4.0))  # NAO
elif robot.name == 'centauro':
    plt.subplot(6, 3, 3)  # Centauro
    plt.ylim((0., 5.0))  # Centauro

plt.ylabel('SPDdist')
plt.plot(log_array[:, -1])

for i in range(dx_bal.shape[0]):
    if robot.name == 'nao':
        plt.subplot(4, 3, i+4)  # NAO
    elif robot.name == 'centauro':
        plt.subplot(6, 3, i + 4)  # Centauro

    plt.ylabel('dx_bal'+str(i+1))
    plt.plot(log_array[:, num_dofs + 2 + i])
    plt.ylim((-.05, .05))
# fig2.savefig(robot.name + '_eigValsAndErrors_Man' + str(num_des_manip) + 'Pos' + str(init_qs) + '.png',
#              bbox_inches='tight')

# plt.tight_layout()
plt.show()
