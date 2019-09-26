#!/usr/bin/env python
"""In this file, we create from scratch the inverted pendulum swing-up environment defined in OpenAI Gym.

This is based on the control problem proposed in OpenAI Gym [1]:
"The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem,
the pendulum starts in a random position, and the goal is to swing it up so it stays upright." [1]

References:
    - [1] Pendulum environment in OpenAI Gym: https://gym.openai.com/envs/Pendulum-v0/
"""

import numpy as np

import pyrobolearn as prl


# create simulator
sim = prl.simulators.Bullet()


# create basic world with the pendulum
world = prl.worlds.BasicWorld(sim)
robot = world.load_robot('pendulum')
robot.disable_motor()  # such that it swings freely
robot.print_info()


# create state: [cos(q_1), sin(q_1), \dot{q}_1]
trig_position_state = prl.states.JointTrigonometricPositionState(robot=robot)
velocity_state = prl.states.JointVelocityState(robot=robot)
state = trig_position_state + velocity_state
print("\nObservation: {}".format(state))


# create action: \tau_1
action = prl.actions.JointTorqueAction(robot, bounds=(-2., 2.))
print("\nAction: {}".format(action))


# create reward/cost: ||d(q,q_{target})||^2 + 0.1 * ||\dot{q}||^2 + 0.001 * ||\tau||^2
position_cost = prl.rewards.JointPositionCost(prl.states.JointPositionState(robot),
                                              target_state=np.zeros(len(robot.joints)),
                                              update_state=True)
velocity_cost = prl.rewards.JointVelocityCost(velocity_state)
torque_cost = prl.rewards.JointTorqueCost(prl.states.JointForceTorqueState(robot=robot), update_state=True)
reward = position_cost + 0.1 * velocity_cost + 0.001 * torque_cost
print("Reward: {}".format(reward))


# create initial state generator: generate the state each time we reset the environment
def reset_robot(robot):  # function to disable the motors every time we reset the joint state
    def reset():
        robot.disable_motor()
    return reset


init_state = prl.states.JointPositionState(robot)
low, high = np.array([-np.pi] * len(robot.joints)), np.array([np.pi] * len(robot.joints))
initial_state_generator = prl.states.generators.UniformStateGenerator(state=init_state, low=low, high=high,
                                                                      fct=reset_robot(robot))


# create physics randomizer: randomize the mass each time we reset the environment
masses = robot.get_link_masses(link_ids=robot.joints)
masses = (masses - masses/10., masses + masses/10.)
physics_randomizer = prl.physics.LinkPhysicsRandomizer(robot, link_ids=robot.joints, masses=masses)


# create the environment using composition
env = prl.envs.Env(world=world, states=state, rewards=reward, actions=action,
                   initial_state_generators=initial_state_generator, physics_randomizers=physics_randomizer,
                   terminal_conditions=None)


# run simulation
env.reset()
for t in prl.count():

    if (t % 800) == 0:  # reset to see what initial_state_generator and physics randomizer do
        env.reset()
        print("New link mass: {}".format(robot.get_link_masses(link_ids=robot.joints)))

    states, rewards, done, info = env.step(sleep_dt=1./240)
    print("Reward: {}".format(rewards))
