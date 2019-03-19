#!/usr/bin/env python
"""Load the various Youbot robotic platforms.

These include: YoubotBase, KukaYoubotArm, Youbot, YoubotDualArm
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import YoubotBase, KukaYoubotArm, Youbot, YoubotDualArm

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
youbot_base = YoubotBase(sim, init_pos=(0, -0.75))
kuka_arm = KukaYoubotArm(sim, init_pos=(0, -0.25))
youbot = Youbot(sim, init_pos=(0, 0.25))
youbot_dual_arm = YoubotDualArm(sim, init_pos=(0, 0.75))

robots = [youbot_base, kuka_arm, youbot, youbot_dual_arm]

# print information about the robot
for robot in robots:
    robot.printRobotInfo()

# Position control using sliders
# robot.addJointSlider()

# run simulator
for _ in count():
    # robots[0].updateJointSlider()
    # for robot in robots:
    #     robot.drive(5)
    world.step(sleep_dt=1./240)
