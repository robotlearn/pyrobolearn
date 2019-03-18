## States

The `State` is returned by the environment and given to the policy. The state might include information about the state of one or several objects in the world, including robots.

It is the main bridge between the robots/objects in the environment and the policy. Specifically, it is given as an input to the policy which knows how to feed the state to the learning model. Usually, the user only has to instantiate a child of this class, and give it to the policy and environment, and that's it. In addition to the policy, the state can be given to a controller, dynamic model, value estimator, reward function, and so on.

To allow the framework to be modular and flexible, we favor [composition over inheritance][https://en.wikipedia.org/wiki/Composition_over_inheritance] leading the state to be decoupled from notions such as the environment, policy, rewards, etc. This class also describes the `state_space` which has initially been defined in [`gym.Env`](https://github.com/openai/gym).


```python
from itertools import count
import time
import pyrobolearn as prl


sim = prl.simulators.BulletSim()
world = prl.worlds.BasicWorld(sim)
robot = prl.robots.loadRobot('wam')  # try another robot such as 'coman' or 'littledog'

state = prl.states.JointPositionState(robot) + prl.states.JointVelocityState(robot)  # you can add other states
env = prl.envs.Env(world, state)  # the state, world, and possible rewards are defined outside the environment

for t in count():
    if (t %  240) == 0:
        print(state)  # print the joint positions and velocities of the specified robot
    env.step()  # this will ask the state to read or compute the next value, and perform a step in the simulator 
    time.sleep(1./240)
```

#### What to check/look next?

Check first the `actions` folder, then the `approximators`, `policies`, `rewards`, and `envs` folders.
