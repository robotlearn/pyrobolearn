# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Example on how to use the 'Acrobot' OpenAI Gym environments in PRL using the `stable_baselines` library.
"""

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from pyrobolearn.envs import gym  # this is a thin wrapper around the gym library

# create env, state, and action from gym
env = gym.make('Acrobot-v1')
state, action = env.state, env.action
print("State and action space: {} and {}".format(state.space, action.space))

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
