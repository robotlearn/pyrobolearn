Environments
============

- World
- States
- Actions
- Rewards

Available environments include ...


How to use an environment?
--------------------------


Design
------


How to create my own environment?
---------------------------------


What are the differences with the OpenAI gym's environments?
------------------------------------------------------------

To better depict the differences, let's consider an environment which contains a quadruped robot and the goal is that it learns to walk. Usually, as it can be seen on multiple repositories, people would inherit from the gym ``Env`` class and call it something similar to ``QuadrupedFlatTerrainWalkEnv(Env)``. Inside of ``step`` function, they would compute the next states and rewards. Now suppose, you would like to change ...

In our framework, the ``world``, ``states``, and ``rewards`` are given to the PRL ``Env`` class. This means that if you would like to change the world, reward function, or states you can do it outside the function.

- Actions

Having said that, we tried to make PRL compatible with OpenAI gym at the exception that the returned state is not a array but a list of arrays.
