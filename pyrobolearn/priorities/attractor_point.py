#!/usr/bin/env python
"""Attractor point using impedance control with RRBot.

Try to move the end-effector using the mouse, and see what happens. This example use priority tasks and constraints,
which are optimized using Quadratic Programming (QP).
"""

import numpy as np
import time
import pyrobolearn as prl


# Create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# create robot
robot = world.load_robot(prl.robots.RRBot)
robot.disable_motor()           # disable motors; comment the `robot.set_joint_torques(torques)` to see what happens
robot.print_info()
robot.change_transparency()

# define useful variables for impedance control
link_id = robot.end_effectors[0]  # the link we are interested to
x_des = robot.get_link_world_positions(link_id)       # desired cartesian position
wrt_link_id = None

# gains
K = 100 * np.identity(3)
D = 6 * np.sqrt(K)

# draw a sphere at the desired location
world.load_visual_sphere(position=x_des, radius=0.1, color=(0, 1, 0, 0.5))

# create task
model = prl.priorities.models.RobotModelInterface(robot)
cartesian_task = prl.priorities.tasks.torque.CartesianImpedanceControlTask(model, distal_link=link_id,
                                                                           base_link=wrt_link_id,
                                                                           desired_position=x_des, kp_position=100,
                                                                           kd_linear=60)
postural_task = prl.priorities.tasks.torque.JointImpedanceControlTask(model, q_desired=[0., 0.],
                                                                      kp=10)
# task = cartesian_task
# task = cartesian_task / postural_task
task = cartesian_task + 0.05 * postural_task
print("\nTask: \n{}\n".format(task))
solver = prl.priorities.solvers.QPTaskSolver(task=task)

# run simulation
times = []
for t in prl.count():

    # update task
    task.update(update_model=True)

    # solve task
    start = time.time()
    torques = solver.solve()
    end = time.time()
    times.append(end - start)

    if (t+1) % 1000 == 0:
        print("solving time: avg={}, std={}".format(np.mean(times), np.std(times)))
        times = []

    # set joint torques
    robot.set_joint_torques(torques)

    # step in simulation
    world.step(sleep_dt=sim.dt)
