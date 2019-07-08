Examples
========

In this folder, you will find different examples on how to use the framework.

Warning: this folder is currently being updated; few files might still have some bugs or not 
implemented completely. Some other folders will be added in the upcoming days.

You can check the following folders on:

- ``simulators``: how to use a particular simulator. Currently, the Bullet simulator is the one fully operational.
- ``worlds``: how to create a world in the simulator, load various objects inside and interact with them, use the camera, and load or generate terrains.
- ``robots``: how to load a specific robot (biped, quadruped, wheeled, etc) into the world.
- ``interfaces``: the various interfaces (game controllers, webcam, etc) and bridges that you can use.
- ``kinematics``: how to use forward and inverse kinematics as well as position and velocity control.
- ``dynamics``: how to use forward and inverse dynamics as well as force control.
- ``manipulability``: how to use the velocity and dynamic manipulability ellipsoids.
- ``states``: how to query the states / observations.
- ``models``: the different learning models that you can use.
- ``imitation``: how to use imitation learning with the framework.
- ``gym/cartpole``: policies that are trained with different algorithms on the gym Cartpole environment.
