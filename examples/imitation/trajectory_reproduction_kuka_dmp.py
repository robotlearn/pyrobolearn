# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Example on how to use an imitation learning task in PyRoboLearn using Dynamic Movement Primitives as a policy,
and a mouse keyboard interface.
"""

# General imports
import numpy as np
import matplotlib.pyplot as plt

# Import robots and world
from pyrobolearn.simulators import Bullet
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import KukaIIWA

# Import related to the IL task (states/actions, policy, env, interface)
from pyrobolearn.states import ExponentialPhaseState, JointPositionState
from pyrobolearn.actions import JointPositionAction
from pyrobolearn.envs import Env
from pyrobolearn.policies import BioDiscreteDMPPolicy
from pyrobolearn.tools.interfaces import MouseKeyboardInterface
from pyrobolearn.tools.bridges import BridgeMouseKeyboardImitationTask
from pyrobolearn.recorders import StateRecorder, ActionRecorder
from pyrobolearn.tasks import ILTask


# variables
joint_ids = None  # None for all the actuated joints, or you can select which joint you want to move; e.g. [0, 1, 2]
num_basis = 20
rate = 30

# Create simulator
sim = Bullet()

# create world
world = BasicWorld(sim)

# load robot in the world
robot = world.load_robot(KukaIIWA)
print("Robot's actuated joint ids: {}".format(robot.joints))

# create state/action
state = ExponentialPhaseState(ticks=rate)
action = JointPositionAction(robot, joint_ids=joint_ids)
print("State: {}".format(state))
print("Action: {}".format(action))

# create environment
env = Env(world, state)

# create DMP policy
policy = BioDiscreteDMPPolicy(action, state, num_basis=num_basis, rate=rate)

# create interface/bridge
interface = MouseKeyboardInterface(sim)
bridge = BridgeMouseKeyboardImitationTask(world, interface=interface, verbose=True)

# create recorder
recorder = StateRecorder(JointPositionState(robot, joint_ids=joint_ids), rate=rate)

# create imitation learning task
task = ILTask(env, policy, interface=bridge, recorders=recorder)

# record, train, and test policy using the policy
# task.run()

# record demonstrations
print("\nRecording phase: press `ctrl+r` to start/stop the recording. Once finished, press `shift+r`.")
task.record(signal_from_interface=True)
print("Recording phase: finished the recording!")

# train policy
print("Training phase: training the policy...")
task.train(signal_from_interface=False)
print("Training phase: policy trained!")

# plot what the DMP policy has learned by performing a rollout
y, dy, ddy = policy.rollout()
plt.figure()
plt.suptitle('DMP position trajectories in joint space')
for i in range(y.shape[0]):
    plt.subplot(3, 3, i+1)
    plt.title('q'+str(i))
    plt.plot(y[i])
plt.tight_layout()
plt.show()

# test policy
print("Reproduction phase: test policy...")
task.test(num_steps=rate*100, signal_from_interface=False)
print("Reproduction phase: Policy tested!")
