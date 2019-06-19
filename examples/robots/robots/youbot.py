#!/usr/bin/env python
"""Load the various Youbot robotic platforms.

These include: YoubotBase, KukaYoubotArm, Youbot, YoubotDualArm
"""

from itertools import count
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import YoubotBase, KukaYoubotArm, Youbot, YoubotDualArm

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# create robot
youbot_base = YoubotBase(sim, position=(0, -0.75))
kuka_arm = KukaYoubotArm(sim, position=(0, -0.25))
youbot = Youbot(sim, position=(0, 0.25))
youbot_dual_arm = YoubotDualArm(sim, position=(0, 0.75))

robots = [youbot_base, kuka_arm, youbot, youbot_dual_arm]

# print information about the robot
for robot in robots:
    robot.print_info()

# Position control using sliders
# robot.add_joint_slider()

# run simulator
for _ in count():
    # robots[0].update_joint_slider()
    # for robot in robots:
    #     robot.drive(5)
    world.step(sleep_dt=1./240)
