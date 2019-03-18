## Actions

The `Action` is produced by the policy in response to a certain state/observation. From a programming point of view, compared to the `State` class, the action is a setter object. Thus, they have a very close relationship and share many functionalities. Some actions are mutually exclusive and cannot be executed at the same time.

An action is defined as something that affects the environment; that forces the environment to go to the next state. For instance, an action could be the desired joint positions, but also an abstract action such as 'open a door' which would then open a door in the simulator and load the next part of the world. This would depend on how the user implemented his/her action class.

In the framework, the `Action` class is decoupled from the policy and environment rendering it more modular and [flexible](https://en.wikipedia.org/wiki/Composition_over_inheritance). Nevertheless, the `Action` class still acts as a bridge between the policy and environment. In addition to be the output of a policy/controller, it can also be the input to some value estimators, dynamic models, reward functions, and so on.

You can for instance call `JointPositionAction(robot)`, and this will use position control to set the given joint positions. By giving the robot as input to the `JointPositionAction` class, it will automatically get the number of joints that the given robot possesses. This can then later be useful for instance when building a certain learning model for a policy. For example, assume we want to use a multilayer perceptron as the policy. The number of units on the last layer depends on the number of joints of the considered robot. Using the `MLPPolicy(outputs=actions)` it will automatically sets the correct number of output units depending on the considered robot.


#### What to check/look next?

Check first the `states` folder if not already done, then the `approximators`, `policies`, `rewards`, and `envs` folders.
