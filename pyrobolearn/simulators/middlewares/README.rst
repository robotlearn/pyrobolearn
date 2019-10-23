Middlewares
===========

This folder provides interfaces to the middlewares that are used in robotics (such as ROS, YARP, etc). All these
classes inherit from the ``Middleware`` abstract class. Middlewares can be provided to simulators which can then use
them to send/receive messages. This allows to communicate with real platforms as well.

The Middleware has a list of RobotMiddleware, where each one specifies how to communicate with the robot middleware.

