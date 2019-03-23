## Terminal conditions

This folder contains terminal conditions that are used to end an environment. This for instance can be used to notify that the task was successfully carried out or it resulted in a failure.

More concretely, assume you have a locomotion task where a robot is supposed to move from a point A to a point B. Two terminal conditions can be defined in this case, one which is triggered when the robot has fallen (failure), and one where the robot arrived at point B (success).

With respect to `gym.envs`: currently, gym environments only return a boolean to notify if the environment is over or not, which is quite restrictive. Did the agent succeeded to perform the task or did it failed? What particular condition causes the environment to end? Also, some environments share the same terminal conditions but they are copied-pasted in the code under the `step` method, resulting in code duplication. Instead, as we did for the `world`, `rewards`, `states`, and other modules, we define the terminal conditions outside the environment, and give them as arguments to the environment constructor. We thus favor [composition over inheritance](https://en.wikipedia.org/wiki/Composition_over_inheritance) resulting in better reusability and flexibility.

## What to look/check next?

Check the `envs` folder.

