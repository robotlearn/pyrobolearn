#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The task is to track the force along z axis (vertical to the table) by employing admittance control, meanwhile tracking
a circle trajectory on the xy plane. And the end-effector's target position is visualized by a sphere.
Reference:
[1] SONG, Peng; YU, Yueqing; ZHANG, Xuping. A tutorial survey and comparison of impedance control on robotic manipulation
. Robotica, 2019, 37.5: 801-836.
"""

import numpy as np
from itertools import count

from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA, Body, sensors
from pyrobolearn.utils.transformation import *

from plotting_ee_FT import EeFtRealTimePlot

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
    """
    First step: is to arrive the initial position
    """
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
    """
        Second step: From the initial pose, Move vertically downward 
        until end-effector touches the desktop with a force of 10N 
    """
    for t in count():
        Fz_desired = 10  # the threhold of the contact force with table
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
    sp_z = []
    num = []
    detx = np.array([0.0, 0.0, 0.0])
    """
    Third step to keep the target force along z axis(vertical to the table), 
    and complete circular motion trajectory on plane xy
    """
    circle_center = np.array([0.46, 0]) # the center of the trajectory
    for t in count():
        Fz_desired = 100  # desired force
        # move sphere
        if t == 0:
            z = robot.get_link_world_positions(link_id)[2]
            sphere.position = np.array([circle_center[0] - r * np.sin(w * t + np.pi / 2), circle_center[1] + r * np.cos(w * t + np.pi / 2), z])

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

        # set the M\D\K parameters by heuristic method, these parameters may have a good result
        M = 1
        D = 9500
        K = 500000
        # Refer the formula (33) in this article [1]
        # the formula is theta_x(k) = Fc(k)*Ts^2+Bd*Ts*theta_x(k-1)+Md*(2*theta_x(k-1)-theta_x(k-2))/(Md+Bd*Ts+Kd*Ts^2)
        numerator = Fz_error * np.square(dt) + D * dt * detx[1] + M * (2 * detx[1] - detx[2])
        denominator = M + D*dt + K*np.square(dt)
        detx_ = numerator / denominator
        print (detx_)
        detx[2] = detx[1]
        detx[1] = detx[0]
        detx[0] = detx_

        zzz = sphere.position[2] + detx[0]

        # circle_center the the centre of the circle trajectory on the table
        sphere.position = np.array([circle_center[0] - r * np.sin(w * t + np.pi / 2), circle_center[1] + r * np.cos(w * t + np.pi / 2), zzz])
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
    plt.plot(num, sp_z)  # plot the position on the z axis
    plt.xlabel("timesteps")
    plt.ylabel("vertical position")
    plt.title("The z axis position during the task")
    plt.show()



if __name__=='__main__':
    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # flag : 0 # PI control
    flag = 0
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

