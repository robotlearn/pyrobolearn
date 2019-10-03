Comparisons with other Frameworks
=================================

Frameworks can mostly be divided into 2 groups:

1. the ones that focus on providing environments
2. the ones that focus on providing learning models and algorithms

In this document, I will review and compare the various frameworks.

PyRoboLearn:

* OS: Ubuntu 16.04 and 18.04. Also, Windows 10, and Mac OSX but does not support all interfaces.
* Python: 2.7, 3.5, 3.6
* PEP8 compliant (unified code style)
* Documented functions and classes
* Unified framework


Simulators
----------

In this section, we first review the various robotic simulators, and provide comparisons between them:

* Gazebo-ROS
    * free + open source
    * C++ (Python through ROS)
    * Python 3.* but Python 2.7 for some ROS packages
    * `Gazebo <http://gazebosim.org/>`__, `ROS <https://www.ros.org/>`__
* Bullet/PyBullet
    * Free (zlib license) + open source
    * C++
    * Python wrapper (Python 2.7 and 3.*)
    * `Bullet Github repo <https://github.com/bulletphysics/bullet3>`__, and
      `Pybullet Github repo <https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet>`__
* Mujoco/mujoco-py
    * Not free unless student but cannot be used for research (require a License) + not open-source
    * C
    * Python wrapper (Python 3.5)
    * `Main Webpage <http://www.mujoco.org/>`__, `mujoco-py <https://github.com/openai/mujoco-py>`__
* Dart
    * free + open-source
    * C++
    * Python wrapper (dartpy)
    * `Main Webpage <https://dartsim.github.io/>`__
* RaiSim/RaiSimPy
    * free + not open-source (the cpp files)
    * C++
    * Python wrapper (Python 2.7 and 3.*)
    * `RaiSimLib <https://github.com/leggedrobotics/raisimLib>`__,
      `RaiSim Ogre <https://github.com/leggedrobotics/raisimOgre>`__,
      `RaiSimPy <https://github.com/robotlearn/raisimpy>`__
* V-REP/PyRep
    * free + open-source
    * C++
    * Python wrapper (Python 3.*)
    * `V-REP <http://www.coppeliarobotics.com/>`__, `PyRep <https://github.com/stepjam/PyRep>`__
* Isaac Gym (PhysX and FleX)
    * not available for now
    * C++
* Webots
    * free + open-source
    * C++
    * No Python wrappers
    * `Main webpage <https://cyberbotics.com/>`__, `Github repo <https://github.com/cyberbotics/webots>`__
* Argos3
    * free + open-source
    * C++
    * Purpose: for multiple robots
    * No Python wrappers
    * `Main webpage <https://www.argos-sim.info/>`__, `Github repo <https://github.com/ilpincy/argos3>`__
* Chronos/PyChronos
    * free + open-source + cross-platform (BSD-3 license)
    * C++
    * `Chronos <https://projectchrono.org/>`__, `PyChronos <https://projectchrono.org/pychrono/>`__
* OpenSim
    * Purpose: "to develop models of musculoskeletal structures and create dynamic simulations of movement"
    * OS: Linux, Windows, Ubuntu
    * C++
    * `Main Webpage <https://opensim.stanford.edu/>`__,
      `Documentation <https://simtk-confluence.stanford.edu:8443/display/OpenSim/OpenSim+Documentation>`__,
      `Github repo <https://github.com/opensim-org/opensim-core>`__
* Airsim
    * Purpose: simulator for mainly drones and cars
    * Backend: Unreal Engine
    * C++
    * Python client
    * `Documentation <https://microsoft.github.io/AirSim/>`__, `Github repo <https://github.com/microsoft/AirSim>`__
* Carla
    * Purpose: simulator for autonomous driving systems
    * C++
    * Python client
    * `Main webpage <http://carla.org/>`__, `Documentation <https://carla.readthedocs.io/en/stable/>`__

Not really simulators but more like tools:

* Drake
    * modelling dynamical systems + solving mathematical programs + multibody kinematics and dynamics
    * C++
    * Python wrapper
    * `Main Webpage <https://drake.mit.edu/>`__
* Robotics toolbox
    * Matlab
    * No Python wrappers
    * `Main Webpage <http://petercorke.com/wordpress/toolboxes/robotics-toolbox>`__

Not really simulators but provide physics engine (i.e. dynamics/physics library):

* ODE
    * C++
    * `Bitbucket repo <https://bitbucket.org/odedevs/ode>`__
* SimBody
    * C++
    * `Main Webpage <https://simtk.org/projects/simbody>`__, `Github repo <https://github.com/simbody/simbody>`__


The following table summarizes the comparisons between the various simulators:

.. csv-table::
    :header: "Name", "Language", "Free", "Open-source", "Supported OS", "Python wrapper", "Main purpose", "License"

    "Bullet", "C/C++", "Yes", "Yes", "Linux, Mac OSX, Windows", "2.7 and 3.*", "Robotics, Game, Graphics", "Zlib"
    "MuJoCo", "C", "Yes", "No", "Linux, Mac OSX, Windows", "mujoco_py >=3.5", "Robotics, Game, Graphics", "Proprietary (MIT for mujoco_py)"
    "Dart", "C++", "Yes", "Yes", "Linux, Mac OSX, Windows", "2.7 and 3.*", "Robotics", "BSD 2"
    "Raisim", "C++", "Yes", "No", "Linux (Ubuntu)", "2.7 and 3.*", "Robotics", "EULA (MIT for raisimpy)"
    "V-Rep", "C++", "Yes", "Yes", "Linux (Ubuntu), Mac OSX, Windows", ">= 3.5", "Robotics", "Commercial or GNU GPL (MIT for PyRep)"
    "Gazebo + ROS", "C++", "Yes", "Yes", "NA", "ROS (2.7 and some packages >=3.5)", "Robotics", "Apache 2.0 (for Gazebo), BSD3 (for ROS)"
    "Isaac", "C++", "NA", "NA", "NA", "NA", "Robotics", "NA"
    "Chronos", "C++", "Yes", "Yes", "Linux, Mac OSX, Windows", "Yes", "Robotics", "BSD3"
    "OpenSim", "C++", "Yes", "Yes", "Mac OSX, Windows", "NA", "Musculoskeletal models", "Apache 2.0"
    "Airsim", "C++", "Yes", "Yes", "Linux and Windows", ">=3.5", "Cars and Drones", "MIT"
    "Carla", "C++", "Yes", "Yes", "Linux and Windows", "Yes", "Autonomous driving agents", "MIT"
    "Webots", "C++", "Yes", "Yes", "Linux, Mac OSX, Windows", "No", "Robotics", "Apache 2.0"
    "Argos3", "C++", "Yes", "Yes", "Linux, Mac OSX", "No", "Swarm Robotics", "MIT"


Have also a look at `SimBenchmark <https://leggedrobotics.github.io/SimBenchmark/>`__ for a comparison between various
physics engines.


Environments
------------

* OpenAI Gym
    * OS: Linux and OS X
    * Python 2.7 or 3.5
* gym-miniworld
    * Python 3.5+
* DeepMind Control Suite
    * OS: Ubuntu 14.04 and 16.04
    * Python: 2.7 and 3.5
    * Simulator: Mujoco
* Roboschool
    * OS: Ubuntu/Debian and Mac OS X
    * Python 3 (might be compatible with Python 2.7 but "may require non-trivial amount of work")
    * Simulator: Internal
* Pybullet Gym
    * OS: Linux, Windows and OS X
    * Python 2.7 or 3.5
    * Simulator: PyBullet
* GibsonEnv
    * Nvidia GPU with VRAM > 6GB
    * OS: Ubuntu >= 14.04
    * Python 3.5 is recommended
* AI-habitat
    * Python 3
* Airsim
    * Requirements:
        * OS: Linux and Windows
        * C++, Python, C# and Java
        * Unreal Engine + Unity
* Carla
    * OS: Linux and Windows
    * Python
* Nvidia Isaac Gym/Sim
    * Information unavailable for the moment
* Surreal Robotics Suite
    * OS: Mac OS X and Linux
    * Python 3.5 or 3.7
    * Simulator: Mujoco
    * Robots: Baxter
    * Devices: mouse and spacemouse
    * Paradigms: Imitation and reinforcement
    * Robot Manipulation
* PyRobot
    * OS: Ubuntu 16.04
    * Python 2.7
    * Simulator: Gazebo(+ROS)
    * PyRobot is a lightweight Python framework which is built on top of Gazebo-ROS, and focuses on manipulation and navigation.
    * Comparisons with PyRoboLearn: PyRoboLearn can be seen as the more heavyweight version of that framework.
* S-RL Toolbox (Reinforcement Learning (RL) and State Representation Learning (SRL) Toolbox for Robotics)
    * OS: Linux, Mac OSX, Windows
    * Python 3
    * Simulator: PyBullet
    * use stable-baselines
* RLBench
    * OS: Ubuntu 16.04 + Windows + Mac OSX
    * Python 3
    * Simulator: PyRep
    * Paradigms: 
    * https://github.com/stepjam/RLBench
* gym-chrono
    * OS: Linux, Windows, OSX
    * Python
    * Simulator: PyChrono
    * Paradigm: reinforcement
    * https://github.com/projectchrono/gym-chrono
* ROBEL
    * "ROBEL (RObotics BEnchmarks for Learning): a collection of affordable, reliable hardware designs for studying
      dexterous manipulation and locomotion on real-world hardware"
    * `Main webpage <https://sites.google.com/view/roboticsbenchmarks/>`__,
      `Github repo <https://github.com/google-research/robel>`__


.. csv-table:: Comparisons between different robot learning frameworks that provide environments. PL stands for perception learning, SRL for state representation learning, and AV for autonomous vehicles.
    :header: "Name", "Supported OS", "Python", "Simulator", "Paradigm", "Robot", "Domain", "Last active"

    "OpenAI Gym", "Linux, Mac OSX", "2.7, 3.5", "MuJoCo", "RL", "3D chars", "Manipulation, Locomotion", "few days ago"
    "DeepMind Control Suite", "Ubuntu 14.04/16.04", "2.7, 3.5", "MuJoCo", "RL", "3D chars", "Locomotion, Control"
    "Roboschool", "Ubuntu/Debian, Mac OSX", "3", "Bullet", "RL", "3D chars", "Locomotion, Control"
    "Pybullet Gym", "Linux, Mac OSX, Windows", "2.7, 3.5", "PyBullet", "RL", "3D chars, Atlas", "Manipulation, Locomotion, Control", "8 months ago"
    "GibsonEnv", "Ubuntu", "3.5", "Bullet", "PL/RL", "3D chars, 5 robots", "Perception, Navigation"
    "Airsim", "Linux, Windows", "3.5+", "Unreal Engine/Unity", "IL/RL", "AV", "Navigation"
    "Carla", "Ubuntu 16.04+, Windows", "2.7, 3.5", "Unreal Engine", "IL/RL", "AV", "Navigation"
    "Surreal Robotics Suite", "Linux, Mac OSX", "3.5, 3.7", "MuJoCo", "IL/RL", "Baxter/Sawyer", "Manipulation"
    "S-RL Toolbox", "Linux, Mac OSX, Windows", "3.5+", "PyBullet", "RL/SRL", "Kuka/OmniRobot", "Manipulation, Navigation"
    "RLBench", "Linux, Mac OSX, Windows", "3.5+", "PyRep", "IL/RL/ML/MTL", "Franka Emika Panda", "Manipulation", "few days ago"
    "gym-chrono", "Linux, Windows, OSX", "NS", "PyChrono", "RL", "Pendulum, Ant, Hexapod, Manipulator", "Manipulation, Locomotion, Control", "few days ago"
    "ROBEL", "NS", "3.5+", "MuJoCo", "RL", "D'Claw, D'Kitty", "Hardware, Manipulation, Locomotion", "few days ago"
    "PyRoboLearn", "Linux (Mac OSX*, Windows*)", "2.7, 3.5, 3.6", "Agnostic (PyBullet)", "IL/RL", "60+ robots", "Manipulation, Locomotion, Control", "few days ago"

Note that except PRL none of these frameworks is modular nor flexible. PRL also has the advantage of being heavily documented.

Full comparison with RLBench as it is the one that seems to be most similar to the proposed framework:

- RLBench is only available in Python 3.* (not back compatible with Python 2.7), while PRL is available in both versions
- RLBench uses the PyRep simulator while PRL is agnostic wrt the used simulator in principle
- You currently can not load URDFs with PyRep (and thus with RLBench)
- You have to specify the scene (environment) and the objects/robots in the scene beforehand in V-REP and generate a ttm
  or ttt file in a static way.
- RLBench focuses mostly on manipulation tasks but has the advantage of providing more than 100 tasks to users. However, note that currently these
  are defined for the Franka Emika Panda manipulator.
- PRL proposes more different robotic platforms, interfaces, and has in general more features (priority tasks, ROS
  support, and others. See following `link <https://github.com/robotlearn/pyrobolearn/blob/master/pyrobolearn/README.rst>`__).
- Regarding the 3D models that are available (in OBJ, and binary format but can be opened with VREP), I am currently not
  sure if their redistribution is allowed especially without providing the license and proper attribution (see
  following `link <https://github.com/robotlearn/pyrobolearn/blob/master/pyrobolearn/worlds/meshes/README.rst>`__).


Models & Algorithms
-------------------

Models and algorithms depends on the use of a backend library (numpy, tensorflow, keras, pytorch, ...).
Because PRL is using pytorch and numpy as backends, I will mostly focus on these.

* Keras-RL:
    * Model and algorithm coupled
* rllab/garage
    * Python 3.5+ (officially), old branch for Python 2 (for rllab)
    * Backend: TensorFlow
* rllib
    * OS: Ubuntu 14.04, 16.04, 18.04 + Mac OSX 10.11, 10.12, 10.13, 10.14
    * Python 2 and 3
    * Backend: TensorFlow, PyTorch
* pytorch
    * OS: Linux, Mac, Windows
    * Python 2.7, 3.5, 3.6, 3.7
* stable-baselines
    * OS: Ubuntu, Mac OSX, Windows 10
    * Python >= 3.5
    * Backend: TensorFlow
* Catalyst
    * Python 3.6+
    * Backend: PyTorch 0.4.1+
    * Modular and more flexible
* rlpyt
    * Backend: PyTorch
    * https://github.com/astooke/rlpyt
* Deep Reinforcement Learning Algorithms with PyTorch
    * Backend: PyTorch
    * https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch

.. csv-table:: Learning models and algorithms
    :header: "Name", "Supported OS", "Python", "Backend", "Flexible"

    "rllab / garage", "Linux, Mac OSX", "3.5+", "TensorFlow", "No"
    "rllib", "Ubuntu 1[4,6,8], Mac OSX 10.1[1-4]", "2, 3", "TensorFlow, PyTorch", "No"
    "stable-baselines", "Linux, Mac OSX, Windows", "3.5", "TensorFlow", "No"
    "Catalyst", "NS", "3.6+", "PyTorch", "Yes"
    "rlpyt", "NS", "NS", "PyTorch", "Yes"
    "DRL with PyTorch", "NS", "NS", "PyTorch", "No"
    "PyRoboLearn", "Ubuntu 16.04/18.04", "2.7, 3.5, 3.6", "PyTorch", "Yes"
