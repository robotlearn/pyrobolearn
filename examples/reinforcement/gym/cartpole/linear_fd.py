#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example on how to use the 'Cartpole' OpenAI Gym environments in PyRoboLearn using a linear policy trained with
the finite difference algorithm.
"""

import matplotlib.pyplot as plt

from pyrobolearn.envs import gym
from pyrobolearn.policies import LinearPolicy
from pyrobolearn.tasks import RLTask
from pyrobolearn.algos import FD


# create env, state, and action from gym
env = gym.make('CartPole-v1')
state, action = env.state, env.action
print("State and action space: {} and {}".format(state.space, action.space))

# create policy
policy = LinearPolicy(state, action)

# create task and run it
task = RLTask(env, policy)
task.run(num_steps=1000, use_terminating_condition=True, render=True)

# create RL algo
# Note: the hyperparameters can be a little bit tricky to optimize...
algo = FD(task, policy, std_dev=0.01, learning_rate=0.01, difference_type='central', normalize_grad=True)
rewards = algo.train(num_steps=1000, num_rollouts=5, num_episodes=50, verbose=True)

# plot
plt.figure()
plt.plot(rewards)
plt.show()

# test optimized policy
reward = algo.test(num_steps=1000, use_terminating_condition=True, render=True)
print("Final reward obtained on the test: {}".format(reward))
