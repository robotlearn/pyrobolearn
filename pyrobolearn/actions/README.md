## Actions

The `Action` is produced by the policy in response to a certain state/observation. From a programming point of view, compared to the `State` class, the action is a setter object. Thus, they have a very close relationship and share many functionalities. Some actions are mutually exclusive and cannot be executed at the same time.

An action is defined as something that affects the environment; that forces the environment to go to the next state. For instance, an action could be the desired joint positions, but also an abstract action such as 'open a door' which would then open a door in the simulator and load the next part of the world.

In the framework, the `Action` class is decoupled from the policy and environment rendering it more modular [1]. Nevertheless, the `Action` class still acts as a bridge between the policy and environment. In addition to be the output of a policy/controller, it can also be the input to some value estimators, dynamic models, reward functions, and so on.

References:
	[1] "Wikipedia: Composition over Inheritance", https://en.wikipedia.org/wiki/Composition_over_inheritance