#!/usr/bin/env python
"""Example on how to use the PRL 'Acrobot' environment using the `stable_baselines` library.
"""

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import gym

import pyrobolearn as prl
from pyrobolearn.envs.control.pendulum import InvertedPendulumSwingUpEnv

# create env, state, and action from gym
sim = prl.simulators.Bullet(render=True)
env = InvertedPendulumSwingUpEnv(sim)
print("State and action space: {} and {}".format(env.state.space, env.action.space))
print("State and action merged space: {} and {}".format(env.state.merged_space, env.action.merged_space))

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
# env.render()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
