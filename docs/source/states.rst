States
======

In PRL, every concept is modelized as a class. This is also true for states which are returned by the environment.

.. figure:: ../figures/environment.png
    :alt: environment
    :align: center

    The agent-environment interaction


States are given to the policy and the environment. The environment is responsible to update them while policies read their ``data`` and feed it to the underlying learning model. In the case we use a physics simulator like PyBullet, the environment performs one step in the simulation and calls the ``states()`` which updates the ``data`` they contained. Instead, if you have a dynamical model function, the environment can call this one to update the ``data`` of the various ``states`` without having to call the ``states()`` itself to update their values.

States can also be given to dynamical models (which predicts the next state given the current state and last action), value function approximators (which predicts a scalar value given a state and possibly an action), reward functions, etc.


Design
------

UML


How to use a particular state?
------------------------------

Let's assume you have a quadruped robot, and you would like to get its joint positions, velocities, and base position.

.. code-block:: python
    :linenos:

    from itertools import count
    import pyrobolearn as prl
   	from pyrobolearn.states import BasePositionState, JointPositionState, JointVelocityState


    # create simulator
    sim = prl.simulators.Bullet()

    # load robot
    robot = prl.robots.HyQ2Max(sim)

    # create the states
    base_pos_state = BasePositionState(robot)
    joint_pos_state = JointPositionState(robot, joint_ids=robot.legs)  # you can specify which joints you would like to get the position
    joint_vel_state = JointVelocityState(robot, joint_ids=robot.legs)  # you can specify which joints you would like to get the velocity

    state = base_pos_state + joint_pos_state + joint_vel_state

    # run simulation
    for t in count():
    	# call and print the state
    	print(state())

    	# perform a step in the world
    	world.step(sim.dt)


All the states accept also as inputs:

- ``window_size``: size (by default, it is set to one)
- ``ticks``: the number of simulation ticks to sleep before getting the next state. 

In the example above, for joint states, you could also specify the joints that you would like to get the states from by setting ``joint_ids``. Note that in order to be able to generalize to other robots, avoid to give manually the joint ids but instead gives an attribute of the robot, like, ``robot.legs`` (wich is a list containing the joint ids associated with each leg). 

Now, let's assume that you forgot to include the robot's base orientation and its linear/angular velocity in the state. In other frameworks, it is very likely that you would have to change manually the state and everything that depends on it (the policy / value function approximator which accepts as input the state, the step function in the environment which compute the next state, possibly the reward function which is often based on the current state, etc). In PRL, everything is automatized, and thus setting:

.. code-block:: python

	state = state + BaseOrientationState(robot) + BaseLinearVelocity(robot)

will automatically results the other components to update their input size or because this new state is provided to them.


How to create your own state?
-----------------------------

Let's assume that you want to create a state that accepts as inputs the game controller .

.. code-block:: python
    :linenos:

    import pyrobolearn as prl


    class MyState(prl.states.State):  # inherit from the abstract State class
    	"""Description"""

    	def __init__(self, ...):

    		super(MyState, self).__init__(...)


FAQs
----


Other functionalities
---------------------

- `State generator <https://github.com/robotlearn/pyrobolearn/tree/master/pyrobolearn/states/generators>`_: generate a state (used as initial state generator)
- `State processor <https://github.com/robotlearn/pyrobolearn/tree/master/pyrobolearn/states/processors>`_: process the given state (before giving it to another model such as a policy)


Future works
------------

* add a ``rate`` attribute to the states which is used when we set the real-time on the simulator. Or better, using the ``ticks`` and ``sim.dt`` infer the rate.
