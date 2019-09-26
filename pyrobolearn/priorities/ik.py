#!/usr/bin/env python
"""Inverse kinematics with the Kuka robot where the goal is to follow a moving sphere.

The inverse kinematics is performed using priority tasks and constraints, which are optimized using Quadratic
Programming (QP).
"""

import numpy as np
import time
import pyrobolearn as prl


# Create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# create robot
robot = world.load_robot('kuka_iiwa')

# define useful variables for IK
link_id = robot.get_end_effector_ids(end_effector=0)
joint_ids = robot.joints  # actuated joint
wrt_link_id = None  # robot.get_link_ids('iiwa_link_1')
q_idx = robot.get_q_indices(joint_ids)

# create sphere to follow
x_des = np.array([0.5, 0., 1.])
quat_des = np.array([0., 0., 0., 1.])
sphere = world.load_visual_sphere(position=x_des, radius=0.05, color=(1, 0, 0, 0.5), return_body=True)

# create task
model = prl.priorities.models.RobotModelInterface(robot)
cartesian_task = prl.priorities.tasks.velocity.CartesianTask(model, distal_link=link_id, base_link=wrt_link_id,
                                                             desired_position=x_des, kp_position=50.)
                                                             # desired_orientation=quat_des, kp_orientation=50.)
q_desired = [1.448e-03, 2.790e-01, -2.199e-03, -1.013, 5.948e-04, -1.293, 3.882e-04]
postural_task = prl.priorities.tasks.velocity.PosturalTask(model, q_desired=q_desired, kp=50.)
# task = cartesian_task
# task = postural_task
# task = 1 * cartesian_task + 1 * postural_task
task = cartesian_task / postural_task
print("\nTask: \n{}\n".format(task))
solver = prl.priorities.solvers.QPTaskSolver(task=task)


# define amplitude and angular velocity when moving the sphere
w = 0.01
r = 0.2

# run simulation
times = []
for t in prl.count():
    # move sphere
    sphere.position = np.array([0.5, r * np.cos(w*t + np.pi/2), (1.-r) + r * np.sin(w*t + np.pi/2)])
    # cartesian_task.set_desired_references(desired_position=sphere.position)
    cartesian_task.desired_position = sphere.position
    task.update(update_model=True)

    start = time.time()
    dq = solver.solve()
    end = time.time()
    times.append(end - start)

    if (t+1) % 1000 == 0:
        print("solving time: avg={}, std={}".format(np.mean(times), np.std(times)))
        times = []

    # set joint positions
    q = robot.get_joint_positions()
    q = q[q_idx] + dq * sim.dt
    robot.set_joint_positions(q, joint_ids=joint_ids)

    # step in simulation
    world.step(sleep_dt=sim.dt)
