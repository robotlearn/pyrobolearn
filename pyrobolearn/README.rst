PyRoboLearn (PRL)
=================

In each folder, you will find a ``README`` file describing the purpose of the corresponding submodule.

Here we provide a brief overview of each submodule and its intended use:

- ``simulators``: this contains the abstract ``Simulator`` interface from which all the simulators should inherit from.
  This interface allows to decouple the rest of the code in PRL with the simulator being used. Some simulators might
  have some features that other simulators don't have, in that case, an error is raised or an approximation is made.
  For instance, ``PyBullet`` don't provide joint accelerations, but a simulator like ``MuJoCo`` does. As such, it is
  checked in the ``Robot`` class if the simulator provide these accelerations, if not, it is approximated using finite
  difference. Currently, only ``PyBullet`` is fully-supported.

  - ``middlewares``: this will contain the middleware classes that inherit from the ``Middleware`` class, such as
    ``ROS`` and others. You will be able to provide it to the simulator and the simulator will use it to publish or
    receive packages.

- ``robots``: this contains the various robots (manipulators, grippers, legged robots, wheeled robots, flying robots,
  etc) that can be used in PRL. They all inherit from ``Robot`` which itself inherit from ``Body`` (which is the most
  abstract class). The ``Body`` has direct access to the simulator interface. Robots have also access to:

  - ``sensors``: this contains the various sensors used by robots. They all inherit from the ``Sensor`` class. They
    use the simulator to get their values.
  - ``actuators``: this contains the various actuators used by robots. They all inherit from the ``Actuator`` class.
    They might use the simulator to perform action through it.

- ``worlds``: this contains the main ``World`` class which is inherited by all the other worlds. The ``World`` class
  has direct access to the simulator interface (like ``Body``), and users should interact with it to load the various
  bodies and robots in the world, or change the physical properties (friction, restitution, etc) of the world. Through
  that class, you can also attach two bodies together and generate terrains.
- ``utils``: this contains the various util methods like ``transformations`` (from one orientation representation to
  another one, and functions that can be applied on quaternions), converters (that convert from one data type to
  another), ``interpolators``, ``feedback laws``, and others that are used by other parts of the framework.

  - ``data_structures``: this contains data structures such as ordered sets and the different type of queues.
  - ``plotting``: this contains real-time plotting tools than can plot the joint positions, velocities, accelerations,
    torques, or link frames in real-time. This is used in combination with the ``Simulator``.
  - ``parsers``: this contains mainly parsers for some datasets.

- ``tools``: this contains the *interfaces* and *bridges*. *Interfaces* allows to receive or send the data from/to
  various I/O interfaces (such as mouse, keyboard, 3D space mouse, game controllers, webcam, depth cameras, sensors
  like LeapMotion, and others). They all inherit from the abstract ``Interface`` class which has thread supports. If
  threads are not used, the user has to call the ``step`` method such that it reads the next value (i.e. these are not
  event-driven, i.e. you control when you want to get/set the data). Interfaces are independent from the other
  components in the PRL framework (with maybe at the exception of some ``utils`` methods), and as such can be used in
  other software. *Bridges* makes the connection between an interface and another component in PRL (like a robot or
  body in the world, or the world camera). Fundamentally, they accept as input an interface and the component, and the
  user details what should be done in that class. This allows to decouple the interface from the application part; e.g.
  the same game controller interface could be used to move a wheeled robot or quadcopter robot by providing two bridges
  (one for wheeled robots, and one for quadcopter robots). All the bridges inherit from the abstract ``Bridge`` class,
  and as with interface a ``step`` method can be called.
- ``states``: this contains the various states which all inherit from the ``State`` abstract class. States can easily
  be composed together such that you could specify which states you would like to have. For instance, if you want
  the joint positions, velocities, and the base position and orientation states, you can add them to form one common
  state. Calling the state will compute their values, and they will save these in the ``data`` attribute. They
  basically act as useful containers. States are notably provided as inputs to controllers, policies, and rewards among
  others, and are outputted by the environments.
- ``actions``: this contains the various actions which all inherit from the ``Action`` abstract class. They are given
  notably to the policy during the initialization, which sets the action data. Calling an action will perform an action
  in the simulator (e.g. move the robot joints using position control) or through an interface (e.g. say something
  through the computer speakers).
- ``rewards``: this contains the various *rewards* and *costs*. They all inherit from the ``Reward`` abstract class,
  and various arithmetic operations can be performed on them. They accept as possible arguments the ``State`` and
  ``Action``. Each time you call them, they check the data contained in the given states and actions and compute
  the corresponding reward value. This allows the user to reuse different reward functions and easily combine them
  without worrying how to get or compute the reward value. Note that as for states and actions, rewards can have a
  particular range which specifies their domain. This is useful if we would like to know if a reward function is
  strictly positive or not (e.g. the PoWER RL algorithm only accepts strictly positive rewards which it can check by
  looking at the reward's range).
- ``envs``: this contains the various environments. They all inherit from the ``Env`` class which accepts as arguments
  at least the world, the state, and possibly a reward (if we are in the reinforcement learning case). These arguments
  can be provided at runtime making it easy to (re)use other modules, and render the framework very flexible (see
  `Composition over inheritance <https://en.wikipedia.org/wiki/Composition_over_inheritance>`_). Few robotic
  environments are also provided in this class.

  - ``states/generators``: this contains ``state generators`` which generates ``states`` for the environment. You can
    for instance generate the position / orientation of a body, or its joints. They can be provided to the environment
    and are called each time you reset the environment.
  - ``physics``: this contains ``physics randomizers`` which can randomize the physical properties of the joints
    (e.g. joint damping), links (e.g. mass), and the world (e.g. friction). They can be provided to the environment,
    and are called each time the environment is reset.
  - ``terminal_conditions``: this contains terminal conditions which detect if an episode is over or not. They can
    in addition specify if the environment ended with a success or failure. You can provide them to the environment
    which check them at each time step.

- ``models``: this contains the various learning models, which have parameters or hyperparameters to optimize given
  some data. These models can be categorized into two different types: movement primitives and general function
  approximators. The models are independent from the rest of the framework (except maybe few ``utils`` functions).
  Some models were implemented from scratch while others were wrapped.
- ``approximators``: this contains the various approximators which is basically a wrapper around the above models (only
  the ones that are function approximators and not movement primitives), and accepts as inputs states, actions and
  general arrays/tensors. They all inherit from the ``Approximator`` class and represents an abstraction above the
  model classes. Because they can accept states and actions, this makes them dependent on these submodules in PRL.
  Approximators are notably used to model policies (which maps states to actions), value function approximators (which
  maps states to a scalar, or states and actions to a scalar, or states to a scalar for each discrete action), and
  dynamic transition functions (which maps states and actions to the next states). These are described next.
- ``policies``: this contains the various policies that can be used in PRL. They all inherit from the ``Policy`` class,
  and use internally approximators, or models (if movement primitives). They can operate at different rates, and are
  provided with a state and action instance at the initialization.
- ``values``: this contains the various value function approximators that can be used in PRL. They accepts as inputs
  the states and possibly the actions (if Q-value function approximator). They are mostly used by reinforcement
  learning algorithms).
- ``dynamics``: this contains the various dynamic function approximators. This is mostly used by model-based 
  reinforcement learning algorithms. This is currently not fully-implemented/operational.

- ``tasks``: this contains the various learning tasks. They all inherit from the ``Task`` class and accepts at least
  as inputs the policy(ies) and environment. They act as a container for these two's, and calling the ``step`` method
  will perform one full cycle in the agent-environment interaction loop. Subsequently, you can also call ``run`` to
  run several loop for the specified number of steps. Tasks can notably be provided to algorithms (especially RL
  algorithms).

- ``distribution``: this contains few distributions that are used by exloration strategies (see next bullet point).
- ``exploration``: this contains the various exploration strategies that can be used by the policy; parameter and
  action exploration. They all inherit from the ``Exploration`` class and accepts as inputs the policy that they wrap
  around.
- ``storages``: this contains the various data storages/containers (such as experience replay storage and batches)
  that are used during the learning process.
- ``losses``: this contains the various losses that are used by the various algorithms. As for the rewards, you can
  perform arithmetic operations on them and combine them in different ways.
- ``optimizers``: this contains the various optimizers that can be used. We provide a common interface and wrap popular
  optimizers. Currently, some optimizers are not fully-operational.
- ``returns``: this provides the various returns and estimators that are used in RL.
- ``algos``: this contains the various learning algorithms on how to acquire the data and train the various models
  (policies, values, dynamics, etc).

Other folders include:

- ``filters``: this contains various filters (KF, EKF, UKF, HF, etc).
