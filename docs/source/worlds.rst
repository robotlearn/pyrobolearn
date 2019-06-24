Worlds
======

The world is the second important item in PRL; it is, with the `Body` class, the only class that can access the simulator.


How to use the world in PRL?
----------------------------

You can get the world camera.

light

You can check for more examples in the [`examples/worlds`](https://github.com/robotlearn/pyrobolearn/tree/master/examples/worlds) folder.


FAQs and Troubleshootings
-------------------------

* Why do the `Body` class (and all the classes that inherit from it such as `Robot`) can access the simulator as well? This is because, creating a world when the `Simulator` is the real world doesn't make much sense. The Robot is completely independent of the World.


Where can I find 3d models?
---------------------------

* If it a combination of simple shapes linked together, you can build it in the simulator.
* Pybullet data
* gazebo database
* 
