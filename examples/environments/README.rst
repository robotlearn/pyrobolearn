Environments
============

In this folder, we provide examples on how to define and use environments which are notably useful for imitation and reinforcement learning.

An environment is defined as the following figures (inspired by [1]_):

.. image:: ../../docs/figures/environment.png
    :alt: environment
    :align: center

In PRL, the environment is an abstraction layer class that regroups:

- the world; an instance of ``World`` which will be used to perform a step in the world (simulator). This is called at each step performed by the environment.
- the states: an instance of ``State`` (or a list of them). The states are updated at each time step by the environment.
- the rewards (optional): an instance of ``Reward`` (or a list of them). It is optional because some environments like in imitation learning does not require a reward function. The reward functions are computed at each time step.
- the terminal conditions (optional): an instance of ``TerminalCondition`` (or a list of them) that checks at each time step if the goal of the environment has been achieved. A ``TerminalCondition`` also details if the environment ended with a success or failure.
- the initial state generators (optional): an instance of ``StateGenerator`` (or a list of them) which are called to generate the initial states each time the environment is reset.
- the physics randomizers (optional): an instance of ``PhysicsRandomizer`` (or a list of them) to randomize the physical properties of bodies in the simulator, or the simulator itself, each time the environment is reset.
- the actions (optional): an instance of ``Action`` (or a list of them). The actions are not used nor updated by the environment. This is left to the ``Policy`` or ``Controller``.


By favoring `composition over inheritance <https://en.wikipedia.org/wiki/Composition_over_inheritance>`_ for the environment class, we improve the flexibility of the framework and the reuse of different modules.
This leads ultimately to less code duplication, and ease the process of creating environments.

Here is a short snippet showing the basic usage of an environment:

.. code-block:: python
    :linenos:

    import pyrobolearn as prl

    # define the simulator and world (and load what you want in it)
    sim = ...
    world = ...
    robot = ...

    # define state, action, and reward (and possibly action)
    state = ...
    action = ...
    reward = ...

    # you can give the reward function to your RL environment
    # which will use it when calling `env.step()`.
    env = prl.envs.Env(world, state, reward)

    # like in OpenAI gym environments, you can reset and step in the environment
    obs = env.reset()
    for t in count():
        obs, rew, done, info = env.step()


Few notes regarding the code above:

- the ``action`` can also be given to the environment but it won't be called by the environment. This is carried out by the policy(ies)/agent(s). The main reason why you can give an action to an environment is when later you will create your own environment class (that inherits from ``prl.envs.Env``), you will be able to get the states and actions for your policies in the following way:

.. code-block:: python
    :linenos:

    import pyrobolearn as prl

    # define your environment
    class MyEnv(prl.envs.Env):
        ...

    # create the environment and provide possible arguments
    env = MyEnv(args)

    # get states and actions from your environment
    states, actions = env.states, env.actions

    # create policy
    policy = Policy(states, actions)

- the observation ``obs`` is a list of arrays that are returned by the environment. This is a bit different from what it is usually returned by gym environments (which is an array). The reason is that the states returned by the environment might have different dimensions (e.g. joint positions = 1D array, camera = 2D/3D array, etc) so you can not return one array.
- You can easily update the state, reward function, world, and other modules that are given to environment. This results in less code duplication and greater flexibility.


For more info, please check the documentation.


Examples
~~~~~~~~

Here are few examples that you can find in this folder that better demonstrate how to use the environment:

1. ``basics.py``: show the flexibility of how to build an environment and use it.
2. ``manipulator.py``: show how to define an environment where the goal is to reach a target object using a manipulator.

References:

.. [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 1998
