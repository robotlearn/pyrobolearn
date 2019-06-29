#!/usr/bin/env python
"""Demonstrate how the reward function that measures how much a robot has moved forward increases / decreases based on
the robot velocity. Use the arrow keys on your keyboard to move the robot, and observe how the computed reward value
changes.
"""

from itertools import count
import pyrobolearn as prl


# Create simulator
sim = prl.simulators.Bullet()

# create world
world = prl.worlds.BasicWorld(sim)

# create wheeled robot
robot = world.load_robot('epuck')

# create interface and bridge to control the robot with the keyboard
interface = prl.tools.interfaces.MouseKeyboardInterface()
bridge = prl.tools.bridges.BridgeMouseKeyboardDifferentialWheeledRobot(robot=robot, interface=interface)

# create state
state = prl.states.BasePositionState(robot)

# create reward
reward = 1000 * prl.rewards.ForwardProgressReward(state=state, direction=(1, 0, 0))


# run simulation
for t in count():
    # perform a step with the bridge and interface
    bridge.step(update_interface=True)

    # update state: in this case it will get the base position state and will save it in the state instance
    state()

    # compute reward: this will look in the previously given state instance its current state data
    print("Reward value = {}".format(reward()))

    # perform a step in the simulator
    world.step(sim.dt)
