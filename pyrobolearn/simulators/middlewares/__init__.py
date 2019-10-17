# -*- coding: utf-8 -*-

# import middleware abstract class
from .middleware import MiddleWare

# import ROS
try:
    import rospy
    import roslaunch
    import rosparam
    import rosmsg
    import rosservice
    import rostopic
    import controller_manager.controller_manager_interface as cm_interface

    from .ros import ROS
except ImportError as e:
    print("Some ROS packages were not found... Skipping prl.simulators.middlewares.ROS...")

# import YARP

# # define decorator
# def middleware(function):
#     def wrapper(self, *args, **kwargs):
#         if self.middleware is None:
#             return function(*args, **kwargs)
#     return wrapper
