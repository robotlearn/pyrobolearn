#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test the publisher using Gazebo with ROS control and Pybullet

Before running this file, read the following instructions!

First, clone the following repository in your catkin workspace (catkin_ws/src/):
$ git clone https://github.com/ros-simulation/gazebo_ros_demos

Then, in your workspace compile it using `catkin_make` and source it:
$ catkin_make
$ source devel/setup.bash

Launch the roslaunch gazebo file and control file:
$ roslaunch rrbot_gazebo rrbot_world.launch   # this will launch Gazebo and start the controller manager
$ roslaunch rrbot_control rrbot_control.launch   # this will load the controllers

Now, you should have your Gazebo running with the `rrbot` inside. Then, just run this file:
$ python bullet_ros_control_gazebo.py

And move the `rrbot` robot using your mouse by left-clicking on a part of the robot and moving it.

Here is a video of what it should give: https://www.youtube.com/watch?v=OPh-NCfKKK8

If you want to use 'kuka_iiwa' and 'franka', you will have to follow the same steps as above but this time by cloning:
- https://github.com/IFL-CAMP/iiwa_stack
- https://github.com/erdalpekel/franka_ros  and  https://github.com/erdalpekel/panda_simulation

Then run the corresponding roslaunch files that are provided in these repositories (`simulation.launch` for the panda
robot, and `iiwa_gazebo.launch` for the kuka robot).
"""

import pyrobolearn as prl

ros = prl.middlewares.ROS(publish=True, teleoperate=True)
sim = prl.simulators.Bullet(middleware=ros)

# load world
world = prl.worlds.BasicWorld(sim)

# load robot
robot = world.load_robot('rrbot')  # 'kuka_iiwa', 'franka', 'rrbot'

# run simulation
for t in prl.count():
    # get the joint positions from the Bullet simulator (because :attr:`teleoperate` has been set to True,
    # it will publish these read positions on the corresponding topic)
    q = robot.get_joint_positions()

    # perform a step in the simulator (and sleep for `sim.dt`)
    world.step(sim.dt)
