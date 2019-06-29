Rewards
=======

In this folder, we provide examples on how to use reward/cost functions which are provided to reinforcement learning environments.
We show the available operations you can use on these.

The reward function might be defined as [1]_:

- :math:`r: \mathcal{S} \rightarrow \mathbb{R}`: given the state :math:`s \in \mathcal{S}`, it returns the reward value :math:`r(s)`.
- :math:`r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}`: given the state :math:`s \in \mathcal{S}` and action :math:`a \in \mathcal{A}`, it returns the reward value :math:`r(s,a)`.
- :math:`r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}`: given the state :math:`s \in \mathcal{S}`, action :math:`a \in \mathcal{A}`, and next state :math:`s' \in \mathcal{S}`, it returns the reward value :math:`r(s,a,s')`.

Note that the cost function is just minus the reward function, i.e. it is given by :math:`c(s,a,s') = -r(s,a,s')`.

In PRL, all reward functions inherit from the abstract ``Reward`` class defined in `pyrobolearn/rewards/reward.py <https://github.com/robotlearn/pyrobolearn/tree/master/pyrobolearn/rewards>`_, and several methods and operations are provided.
You can for instance:

* provide the ``State`` and/or ``Action`` instances to some reward functions that will compute the reward value based on their value.
* access to the range of the reward function.
* add, multiply, divide, subtract, and apply basic functions such as :math:`\exp`, :math:`\cos`, :math:`\sin`, and others on reward functions. The resulting range is automatically scaled based on the operations.
* define your own rewards/costs and reuse them in your code.

Here is a short snippet showing the basic usage of reward functions:

.. code-block:: python
    :linenos:

    import pyrobolearn as prl
    from pyrobolearn.rewards import FixedReward, YourReward

    # define the simulator and world (and load what you want in it)
    sim = ...
    world = ...
    ...

    # define your state / action for your reward function
    state = ...
    action = ...

    # define the reward function
    reward = 2 * FixedReward(3) + 0.5 * YourReward(state, action)

    # print the range of the reward function
    print(reward.range)

    # compute the reward value
    value = reward()
    print(value)

    # update the state for instance
    state()     # this will modify the internal state data

    # recompute the reward value
    value = reward()
    print(value)    # you will normally get a different value

    # you can give the reward function to your RL environment
    # which will use it when calling `env.step()`.
    env = prl.envs.Env(world, state, reward)


Examples
~~~~~~~~

Here are few examples that you can find in this folder that better demonstrate how to use the reward functions:

1. ``basics.py``: demonstrate the various features (operations) you can use with the ``Reward`` class.
2. ``manipulator.py``: show how the distance cost decreases as you move the manipulator (with your mouse) closer to the target object in the world.
3. ``forward_progress.py``: show how the reward function that measures how much a robot has moved forward increases / decreases based on the robot velocity. Use the arrow keys on your keyboard to move the robot, and observe how the computed reward value changes.

References:

.. [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 1998