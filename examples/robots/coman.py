#!/usr/bin/env python
"""Load the Coman robot.
"""

from itertools import count
from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import Coman

# Create simulator
sim = BulletSim()

# create world
world = BasicWorld(sim)

# create robot
robot = Coman(sim, useFixedBase=True)

# print information about the robot
robot.printRobotInfo()
print(robot.link_names)

# # Position control using sliders
# robot.addJointSlider()

robot.changeTransparency()
# robot.drawLinkCoMs()
robot.drawLinkFrames()
# robot.drawBoundingBoxes(robot.right_leg[4])

# run simulator
for _ in count():
    # robot.updateJointSlider()
    # robot.computeAndDrawCoMPosition()
    # robot.computeAndDrawProjectedCoMPosition()
    world.step(sleep_dt=1./240)
