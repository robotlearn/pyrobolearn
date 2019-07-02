#!/usr/bin/env python
"""Demonstrate how the distance cost decreases as you move the manipulator (with your mouse) closer to the target
object in the world.
"""

from itertools import count
import pyrobolearn as prl


# Create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# create robot
robot = world.load_robot('kuka_iiwa')
end_effector_id = robot.end_effectors[0]
robot.print_info()

# desired position
sphere = world.load_visual_sphere([0.5, 0., 0.], radius=0.05, color=(1, 0, 0, 0.5), return_body=True)

# create state
state = prl.states.LinkWorldPositionState(robot, link_ids=end_effector_id)

# create reward
# note that the given 'sphere' to the cost is not a state, and thus a PositionState will automatically be created
# for that 'sphere', and called at each time the reward is computed.
reward = prl.rewards.DistanceCost(state, sphere)

# run simulation
for t in count():
    # update state
    state()

    # compute reward
    print("Reward value = {}".format(reward()))

    # perform a step in the simulator
    world.step(sim.dt)
