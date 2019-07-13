#!/usr/bin/env python
"""Attach a gripper/hand to the Kuka manipulator.

In this file, you can attach different grippers / hands to the kuka robot. You can move the robot with the mouse.
"""

import argparse

import pyrobolearn as prl


# create parser to select the gripper/hand
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gripper', help='the gripper/hand to attach to the kuka robot', type=str,
                    choices=['softhand', 'allegrohand', 'wam_gripper', 'youbot_gripper', 'pr2_gripper', 'jaco_gripper',
                             'fetch_gripper', 'franka_gripper', 'baxter_gripper', 'schunk_hand', 'shadowhand'],
                    default='softhand')
args = parser.parse_args()


# create simulator
sim = prl.simulators.Bullet()

# create basic world with floor and gravity
world = prl.worlds.BasicWorld(sim)

# load kuka robot
robot = world.load_robot('kuka_iiwa')

# load hand/gripper
hand = world.load_robot(args.gripper, position=(0., 0., 1.5), fixed_base=False)

# compute parent frame position (this will be removed later and integrated in PRL)
parent_frame_position = [0., 0., 0.]
if args.gripper == 'shadowhand':
    parent_frame_position = [0., 0., 0.1]
elif args.gripper == 'allegrohand':
    parent_frame_position = [0., 0., 0.06]
elif args.gripper == 'wam_gripper':
    parent_frame_position = [0., 0., 0.01]
elif args.gripper == 'youbot_gripper':
    parent_frame_position = [0., 0., 0.03]
elif args.gripper == 'pr2_gripper':
    parent_frame_position = [0., 0., 0.002]
elif args.gripper == 'jaco_gripper':
    parent_frame_position = [0., 0., 0.06]
elif args.gripper == 'fetch_gripper':
    parent_frame_position = [0., 0., 0.07]
elif args.gripper == 'franka_gripper':
    parent_frame_position = [0., 0., 0.02]
elif args.gripper == 'baxter_gripper':
    parent_frame_position = [0., 0., -0.03]
elif args.gripper == 'schunk_hand':
    parent_frame_position = [0., 0., 0.002]

# attach hand/gripper to robot
world.attach(body1=robot, body2=hand, link1=robot.end_effectors[0], link2=-1, joint_axis=[0., 0., 0.],
             parent_frame_position=parent_frame_position, child_frame_position=[0., 0., 0.])

# set the hand joint positions
hand.set_joint_positions([0.] * hand.num_actuated_joints)

# run simulation
for t in prl.count():
    sim.step(sim.dt)
