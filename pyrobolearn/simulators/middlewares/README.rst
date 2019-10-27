Middlewares
===========

This folder provides interfaces to the middlewares that are used in robotics (such as ROS, YARP, etc). All these
classes inherit from the ``Middleware`` abstract class. Middlewares can be provided to simulators which can then use
them to send/receive messages. This allows to communicate with real platforms as well.

The Middleware has a list of RobotMiddleware, where each one specifies how to communicate with the robot middleware.

Note that this part is under construction, but we could already make it work with a real Franka Emika Panda robot 
arm. The code used for that are the 2 examples that are located in ``examples/middlewares/bullet_ros_control_gazebo.py``,
and ``examples/imitation/demo.py``. The last example required to move the real robot, which automatically then 
moved the corresponding robot in the simulator where trajectory data was collected. A DMP was then trained on 
that data, and then the real robot was teleoperated from the simulator using the trained DMP. Note that if 
you are interested to implement your own robot middleware, check the ``franka.py`` file in the ``robots`` 
subfolder.

