#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example on how to use the 'Cartpole' OpenAI Gym environments in PyRoboLearn using a random policy
"""

from pyrobolearn.envs import gym
from pyrobolearn.policies import RandomPolicy
from pyrobolearn.tasks import RLTask


# create env, state, and action from gym
env = gym.make('CartPole-v1')
state, action = env.state, env.action
print("State and action space: {} and {}".format(state.space, action.space))

# create policy
policy = RandomPolicy(state, action)

# create task and run it
task = RLTask(env, policy)
task.run(num_steps=1000, dt=0.02, use_terminating_condition=False, render=True)
