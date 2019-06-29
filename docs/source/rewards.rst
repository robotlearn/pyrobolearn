Rewards
=======

In PRL, every concept is modelized as a class. This is also true for rewards which are returned by the environment as shown in the figure below (inspired by [1]_):

.. figure:: ../figures/environment.png
    :alt: environment
    :align: center

    The agent-environment interaction


The reward function might be defined as [1]_:

- :math:`r: \mathcal{S} \rightarrow \mathbb{R}`: given the state :math:`s \in \mathcal{S}`, it returns the reward value :math:`r(s)`.
- :math:`r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}`: given the state :math:`s \in \mathcal{S}` and action :math:`a \in \mathcal{A}`, it returns the reward value :math:`r(s,a)`.
- :math:`r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}`: given the state :math:`s \in \mathcal{S}`, action :math:`a \in \mathcal{A}`, and next state :math:`s' \in \mathcal{S}`, it returns the reward value :math:`r(s,a,s')`.

Note that the cost function is just minus the reward function, i.e. it is given by :math:`c(s,a,s') = -r(s,a,s')`.


Design
------

In PRL, all reward functions inherit from the abstract ``Reward`` class defined in `pyrobolearn/rewards/reward.py <https://github.com/robotlearn/pyrobolearn/tree/master/pyrobolearn/rewards>`_, and several methods and operations are provided.

You can for instance:

* provide the ``State`` and/or ``Action`` instances to some reward functions that will compute the reward value based on their value.
* access to the range of the reward function.
* add, multiply, divide, subtract, and apply basic functions such as :math:`\exp`, :math:`\cos`, :math:`\sin`, and others on reward functions. The resulting range is automatically scaled based on the operations.
* define your own rewards/costs and reuse them in your code.


How to use a particular reward?
-------------------------------

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


More examples on how to use the rewards can be found in `pyrobolearn/examples/rewards <https://github.com/robotlearn/pyrobolearn/tree/master/examples/rewards/>`_.


How to create your own reward?
------------------------------

In order to create your own reward, you have to inherit from ``Reward`` defined in `pyrobolearn/rewards/reward.py <https://github.com/robotlearn/pyrobolearn/tree/master/pyrobolearn/rewards>`_.

.. code-block:: python
    :linenos:

    import pyrobolearn as prl

    class MyReward(prl.rewards.Reward):
    	"""Description"""

    	def __init__(self, args):
    		# initialize your reward function based on the args
    		...

    		# gives initial value to your reward
    		# this attribute will be used to cache the computed value
    		self.value = 0

    	def _compute(self):
    		# compute the reward function
    		...
    		# save the computed value and return it
    		self.value = ...
    		return self.value


Once done, you will be able to use your reward function and perform operations on it (such as addition, substraction, etc).


FAQs
----

* If you have any questions, please submit an issue on the `Github page <https://github.com/robotlearn/pyrobolearn>`_.


References:
-----------

.. [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 1998
