## Reward Processors

This folder contains code to process the rewards that are returned by the environment. These wraps the original reward, and when called returned a processed reward value. Reward processors are also considered as rewards (as they inherit from the `Reward` class). Wrapping the rewards allows to use different processors for different rewards, and inheriting from `Reward` allows to use the various operations defined in that class. You can, for instance, add two reward processors.

Here is a pseudo-code to illustrate the above points:
```python
r1 = Reward1(args)
r2 = Reward2(args)

reward = 0.5 * r1 + r2
proc_reward_1 = RewardProcessor(reward, args)
proc_reward_2 = 0.5 * RewardProcessor1(r1, args1) + RewardProcessor2(r2, args2)

# to compute the reward, just call the reward which would normally provide different values
reward_value = reward()
proc_reward_1_value = proc_reward_1()
proc_reward_2_value = proc_reward_2()
```

One of the important reward processors used in the RL field (as described in other libraries such as the `baselines.common.vec_env.vec_normalize.py`) is the `GammaStandardizeRewardProcessor` which computes the return :math:`R = r + \gamma * R ` at each time step, then computes a running standard deviation on that return, and finally divides the reward by that standard deviation before returning it.

