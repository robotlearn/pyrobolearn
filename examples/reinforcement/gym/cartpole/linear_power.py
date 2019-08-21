#!/usr/bin/env python
"""Example on how to use the 'Cartpole' OpenAI Gym environments in PyRoboLearn using a linear policy trained with
the PoWER RL algorithm.
"""

import matplotlib.pyplot as plt

from pyrobolearn.envs import gym
from pyrobolearn.policies import LinearPolicy
from pyrobolearn.tasks import RLTask
from pyrobolearn.algos import PoWER


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
# Note: depends a lot on the initialization of the policy
algo = PoWER(task, policy)
rewards = algo.train(num_steps=1000, num_rollouts=10, num_episodes=300, verbose=True)

# plot
plt.figure()
plt.plot(rewards)
plt.show()

# test optimized policy
reward = algo.test(num_steps=1000, use_terminating_condition=True, render=True)
print("Final reward obtained on the test: {}".format(reward))
