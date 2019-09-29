# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Example on how to use the 'Cartpole' OpenAI Gym environments in PyRoboLearn using a linear policy trained with
the Cross-Entropy Method algorithm.
"""

import matplotlib.pyplot as plt

from pyrobolearn.envs import gym
from pyrobolearn.policies import LinearPolicy
from pyrobolearn.tasks import RLTask
from pyrobolearn.algos import CEM


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
algo = CEM(task, policy, population_size=20, elite_fraction=0.2)
avg_rewards, max_rewards = algo.train(num_steps=1000, num_episodes=30, verbose=True)

# plot
plt.figure()
plt.plot(avg_rewards, label='avg')
plt.plot(max_rewards, label='max')
plt.legend()
plt.show()

# test optimized policy
reward = algo.test(num_steps=1000, use_terminating_condition=True, render=True)
print("Final reward obtained on the test: {}".format(reward))
