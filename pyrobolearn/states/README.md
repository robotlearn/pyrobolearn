## States

 The `State` is returned by the environment and given to the policy. The state might include information about the state of one or several objects in the world, including robots.

It is the main bridge between the robots/objects in the environment and the policy. Specifically, it is given as an input to the policy which knows how to feed the state to the learning model. Usually, the user only has to instantiate a child of this class, and give it to the policy and environment, and that's it. In addition to the policy, the state can be given to a controller, dynamic model, value estimator, reward function, and so on.

To allow the framework to be modular, we favor composition over inheritance [1] leading the state to be decoupled from notions such as the environment, policy, rewards, etc. This class also describes the `state_space` which has initially been defined in `gym.Env` [2].

References:
    [1] "Wikipedia: Composition over Inheritance", https://en.wikipedia.org/wiki/Composition_over_inheritance
    [2] "OpenAI gym": https://gym.openai.com/   and    https://github.com/openai/gym