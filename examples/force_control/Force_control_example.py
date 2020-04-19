#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inverse kinematics with the Kuka robot where the goal is to follow a moving sphere.
"""
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
from itertools import count

from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA, Body, sensors
from pyrobolearn.utils.transformation import *

from simulate_test_ur.plotting_ee_FT import EeFtRealTimePlot

from threading import Thread

import matplotlib.pyplot as plt

# Real-time plot the End-effector force and torque
def plotting_thread(plot):
    if not isinstance(plot, EeFtRealTimePlot):
        raise TypeError("Expecting to plot type is CartesianRealTimePlot, not ""{}".format(plot))
    while True:
        plot.update()

# Manipulate the whole process
# The sphere is used to visualize the reference trajectory, So I creat the sphere trajectory as the reference
def manipulator_thread(world, robot, sphere, FT_sensor):
    # First step is to arrive the initial position
    for t in count():
        # move sphere
        sphere.position = np.array([0.36, 0, 0.8])

        # get current end-effector position and velocity in the task/operational space
        x = robot.get_link_world_positions(link_id)
        dx = robot.get_link_world_linear_velocities(link_id)
        o = robot.get_link_world_orientations(link_id)
        do = robot.get_link_world_angular_velocities(link_id)

        # Get joint positions
        q = robot.get_joint_positions()

        # Get linear jacobian
        if robot.has_floating_base():
            J = robot.get_jacobian(link_id, q=q)[:, qIdx + 6]
        else:
            J = robot.get_jacobian(link_id, q=q)[:, qIdx]

        # Pseudo-inverse: \hat{J} = J^T (JJ^T + k^2 I)^{-1}
        Jp = robot.get_damped_least_squares_inverse(J, damping)

        dv = kp * (sphere.position - x) - kd * dx
        dw = kp * quaternion_error(sphere.orientation, o) - kd * do
        # evaluate damped-least-squares IK
        dq = Jp.dot(np.hstack((dv, dw)))

        # set joint positions
        q = q[qIdx] + dq * dt
        robot.set_joint_positions(q, joint_ids=joint_ids)
        if t > 300:
            break
        # step in simulation
        world.step(sleep_dt=dt)

    for t in count():
        Fz_desired = 10
        # move sphere
        sphere.position = np.array([0.36, 0, 0.8-0.0005*t])

        # get current end-effector position and velocity in the task/operational space
        x = robot.get_link_world_positions(link_id)
        dx = robot.get_link_world_linear_velocities(link_id)
        o = robot.get_link_world_orientations(link_id)
        do = robot.get_link_world_angular_velocities(link_id)

        # Get joint positions
        q = robot.get_joint_positions()

        # Get linear jacobian
        if robot.has_floating_base():
            J = robot.get_jacobian(link_id, q=q)[:, qIdx + 6]
        else:
            J = robot.get_jacobian(link_id, q=q)[:, qIdx]

        # Pseudo-inverse: \hat{J} = J^T (JJ^T + k^2 I)^{-1}
        Jp = robot.get_damped_least_squares_inverse(J, damping)

        dv = kp * (sphere.position - x) - kd * dx
        dw = kp * quaternion_error(sphere.orientation, o) - kd * do
        # evaluate damped-least-squares IK
        dq = Jp.dot(np.hstack((dv, dw)))

        # set joint positions
        # robot.set_joint_velocities(dq, joint_ids=joint_ids)
        q = q[qIdx] + dq * dt
        robot.set_joint_positions(q, joint_ids=joint_ids)

        if FT_sensor.sense() is not None:
            if FT_sensor.sense()[2] > Fz_desired:
                break
        # step in simulation
        world.step(sleep_dt=dt)
    Fz_error_old = 0
    sp_z = []
    num = []
    if flag == 1:
        detx = np.array([0.0, 0.0, 0.0])
    for t in count():
        Fz_desired = 100
        # move sphere
        if t == 0:
            z = robot.get_link_world_positions(link_id)[2]
            sphere.position = np.array([0.46 - r * np.sin(w * t + np.pi / 2), r * np.cos(w * t + np.pi / 2), z])
            # zz = z - 0.002  # Try to make the end-effector touch the surface of the table

        # get current end-effector position and velocity in the task/operational space
        x = robot.get_link_world_positions(link_id)
        dx = robot.get_link_world_linear_velocities(link_id)
        o = robot.get_link_world_orientations(link_id)
        do = robot.get_link_world_angular_velocities(link_id)

        # Get joint positions
        q = robot.get_joint_positions()

        # Get linear jacobian
        if robot.has_floating_base():
            J = robot.get_jacobian(link_id, q=q)[:, qIdx + 6]
        else:
            J = robot.get_jacobian(link_id, q=q)[:, qIdx]

        # Pseudo-inverse: \hat{J} = J^T (JJ^T + k^2 I)^{-1}
        Jp = robot.get_damped_least_squares_inverse(J, damping)
        # Apply the admittance control
        Fz_current = FT_sensor.sense()[2]  # record the current Fz

        Fz_error = Fz_current - Fz_desired  # record the current error
        # dv[2] = dv[2] + 0.00016 * Fz_error + 0.0000008 * (Fz_error - Fz_error_old) / dt # 结果较好的dt=2400
        # dv[2] = 0.0013 * Fz_error + 0.0000020 * (Fz_error - Fz_error_old) / dt  # 结果较好的dt=2400
        # dv[2] = 0.00093 * Fz_error + 0.000060 * (Fz_error - Fz_error_old) / dt
        # dv[2] = 0.0052 * Fz_error
        # sphere.position[2] = sphere.position[2] + 0.00095 * Fz_error + 0.000060 * (Fz_error - Fz_error_old) / dt
        if flag == 0:
            Fz_error_integral = Fz_error + Fz_error_old
            zzz = sphere.position[2] + 0.000001 * Fz_error + 0.000002 * Fz_error_integral
            Fz_error_old = Fz_error  # record the current error as the old error
        elif flag == 1:
            # xyz 3 direction impedance control
            # M = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
            # D = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
            # K = np.array([[20, 0, 0], [0, 20, 0], [0, 0, 20]])
            # numerator = np.array([[Fz_error[0], 0, 0], [0, Fz_error[1], 0], [0, 0, Fz_error[2]]]) * np.square(dt) \
            #             + D * dt * dx[:, 1] + M * (2 * dx[:, 1] - dx[:, 2])
            # denominator = M + D*dt + K*np.square(dt)
            # dx_ = numerator * np.linalg.inv(denominator)
            # dx[:, 2] = dx[:, 1]
            # dx[:, 1] = dx[:, 0]
            # dx[:, 0] = np.array([dx_[0, 0], dx_[1, 1], dx_[2, 2]])

            M = 1
            D = 9500
            K = 500000
            numerator = Fz_error * np.square(dt) + D * dt * detx[1] + M * (2 * detx[1] - detx[2])
            denominator = M + D*dt + K*np.square(dt)
            detx_ = numerator / denominator
            print (detx_)
            detx[2] = detx[1]
            detx[1] = detx[0]
            detx[0] = detx_

            zzz = sphere.position[2] + detx[0]

        sphere.position = np.array([0.46 - r * np.sin(w * t + np.pi / 2), r * np.cos(w * t + np.pi / 2), zzz])
        dv = kp * (sphere.position - x) - kd * dx  # compute the other direction tracking error term


        sp_z.append(sphere.position[2])
        num.append(t)

        dw = kp * quaternion_error(sphere.orientation, o) - kd * do
        # evaluate damped-least-squares IK
        dq = Jp.dot(np.hstack((dv, dw)))

        # set joint positions
        q = q[qIdx] + dq * dt
        robot.set_joint_positions(q, joint_ids=joint_ids)

        # print(Fz_error, dv[2])
        if t == 800:
            break
        # step in simulation
        world.step(sleep_dt=dt)
    plt.plot(num, sp_z)
    plt.show()



if __name__=='__main__':
    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # flag : 0 # PI control
    flag = 1
    # create robot
    robot = KukaIIWA(sim)
    robot.print_info()
    world.load_robot(robot)
    world.load_table(position=np.array([1, 0., 0.]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    # define useful variables for IK
    dt = 1. / 240
    link_id = robot.get_end_effector_ids(end_effector=0)
    joint_ids = robot.joints  # actuated joint
    damping = 0.01  # for damped-least-squares IK
    wrt_link_id = -1  # robot.get_link_ids('iiwa_link_1')
    qIdx = robot.get_q_indices(joint_ids)

    # define gains
    kp = 500  # 5 if velocity control, 50 if position control
    kd = 5  # 2*np.sqrt(kp)

    # create sphere to follow
    sphere = world.load_visual_sphere(position=np.array([0.5, 0., 0.5]), radius=0.05, color=(1, 0, 0, 0.5))
    sphere = Body(sim, body_id=sphere)

    # set initial joint p
    # ositions (based on the position of the sphere at [0.5, 0, 1])
    robot.reset_joint_states(q=[8.84305270e-05, 7.11378917e-02, -1.68059886e-04, -9.71690439e-01, 1.68308810e-05,
                                3.71467111e-01, 5.62890805e-05])

    # define amplitude and angular velocity when moving the sphere
    w = 0.01
    r = 0.1

    # I set the reference orientation to a constant
    sphere.orientation = np.array([1, 0, 0, 0])

    FT_sensor = sensors.JointForceTorqueSensor(sim, body_id=robot.id, joint_ids=6)
    # The plotting handle
    plot = EeFtRealTimePlot(robot, sensor=FT_sensor, forcex=True, forcey=True, forcez=True,
                            torquex=True, torquey=True, torquez=True, num_point=1000, ticks=24)
    # FT_ = np.zeros(6)

    plot_t = Thread(target=plotting_thread, args=[plot], name='plotting task')
    manipulator_t = Thread(target=manipulator_thread, args=(world, robot, sphere, FT_sensor), name='manipulator task')

    thread_pools = [plot_t, manipulator_t]
    for thread in thread_pools:
        thread.start()

    for thread in thread_pools:
        thread.join()

