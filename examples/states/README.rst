States
======

In this folder, you will find some examples on how to use the ``pyrobolearn.states.*`` states.

States are classes defined outside the environment and are basically containers. They can be updated by calling them ``state()`` (same as ``state.read()``), or by setting their ``data`` variable.

They can be combined using the addition operator. For instance, ``s_a = s1 + s2``, ``s_b = s2 + s3``, and ``s=s_a + s_b``. Calling ``s_a()`` will update the states ``s1`` and ``s2``, and thus the data contained in ``s_b`` as well (as it contains a pointer to ``s2`` which has been updated). You can just call ``s()`` to update in one loop ``s1, s2, s3`` altogether (``s_a`` and ``s_b`` will reflect that change because they contain a pointer to these states ``s1, s2, s3``).

States are given to the policy and the environment. The environment is responsible to update them while policies read their ``data`` and feed it to the underlying learning model. In the case we use a physics simulator like PyBullet, the environment performs one step in the simulation and calls the ``states()`` which updates the ``data`` they contained. Instead, if you have a dynamical model function, the environment can call this one to update the ``data`` of the various ``states`` without having to call the ``states()`` itself to update their values.

States can also be given to dynamical models (which predicts the next state given the current state and last action), value function approximators (which predicts a scalar value given a state and possibly an action), reward functions, etc.


Here are few examples that you can find in this folder:

1. ``basics.py``: demonstrate the various features of the ``State`` class.
2. ``world.py``: get the pose state of an object loaded in the world.
3. ``robot.py``: get the joint states of a specific robot and print them.


Simple Example
--------------

.. code-block:: python
    :linenos:

	import pyrobolearn.states as states

	s1 = states.CumulativeTimeState()
	s2 = states.AbsoluteTimeState()
	s = s1 + s2
	print(s)

	# update s1
	s1.read()  # or s1()
	print(s1)
	print(s)  # just s1 changed

	# update s1 and s2 by calling s
	s()
	print(s)
	print(s1)
	print(s2)

	# get the data
	print(s.data)  # this will return a list of 2 arrays; each one of shape (1,). The size of the list is equal to the number of states that it contains
	print(s1.data)  # this will return a list with one array of shape (1,)
	print(s.merged_data)  # this will return a merged state; it will merge the states that have the same dimensions together and return a list of arrays which has a size equal to the number of different dimensions. The arrays inside that list are ordered by their dimensionality in an ascending way.


For the programmer
------------------

Why states are defined outside and not inside the environments like usually done in ``gym.envs``. States are defined outside for a better modularity, reusability, flexibility, and lower coupling. 

- Why better modularity? Because you define a module for each possible state which you can combine at your taste later on.
- Why better reusability? Because it avoids you to define how to read similar state in different environments which often lead to code duplication. 
- Why lower coupling? Coupling between two modules measures how much they are dependent on each other. There is a lower coupling, because instead of having a composition relationship between the modules we have an aggregation relationship (see `UML Association vs Aggregation vs Composition <https://www.visual-paradigm.com/guide/uml-unified-modeling-language/uml-aggregation-vs-composition/>`_). That is, because states are defined outside of the environment and given to the environment, even if we destroyed the environment, the states still exist.
- Why better flexibility? Because we favor `composition over inheritance <https://en.wikipedia.org/wiki/Composition_over_inheritance>`_. you can combine different states as you wish, give different states to different policies, and provide them at the end to the environment (which will update them). 
