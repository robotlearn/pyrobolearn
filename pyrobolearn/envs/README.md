## Environments

This folder provides the environment as it described in an imitation / reinforcement learning setting. That is, given an action performed by an agent (i.e. policy), the environment computes and returns the next state and possibly a reward. A policy can then interact with this environment, and be trained.

The environment defines the world, the rewards, and the states that are returned by it. To allow our framework to be flexible, the world, rewards, states, and terminal conditions are decoupled from the environment, and defined outside of this one and then provided as inputs to the `Env` class. See the `worlds`, `states`, and `rewards` folders. We thus favor [composition over inheritance](https://en.wikipedia.org/wiki/Composition_over_inheritance). This is a different approach compared to what is usually done with the [OpenAI gym](https://github.com/openai/gym) framework, where a world, state, rewards are defined inside the class that inherits from `gym.Env`.


```python
from itertools import count
import time
import pyrobolearn as prl


sim = prl.simulators.BulletSim()
world = prl.worlds.BasicWorld(sim)
robot = prl.robots.loadRobot('wam')  # try another robot such as 'coman' or 'littledog'

state = prl.states.JointPositionState(robot) + prl.states.JointVelocityState(robot)  # you can add other states
reward = prl.rewards.<Reward>(<state, action, etc>)  # define the reward/cost
env = prl.envs.Env(world, state, reward, terminal_condition=None)  # the state, world, reward, and terminal condition are defined outside the environment

for t in count():
    next_obs, rew, done, info = env.step()  # this will ask the state to read or compute the next value, and perform a step in the simulator 
    time.sleep(1./240)
    if (t %  240) == 0:
        print(reward)  # print the reward
```

#### What to check/look next?

Check first the `policies` and `tasks` folders.
