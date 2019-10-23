# -*- coding: utf-8 -*-

# import middleware abstract class
from .middleware import Middleware

# import ROS
try:
    import rospy
    import roslaunch
    import rosparam
    import rosmsg
    import rosservice
    import rostopic
    try:
        import controller_manager.controller_manager_interface as cm_interface
    except ImportError as e:
        print("ROS control is not installed for this Python version, please install it... For now, disabling the "
              "ROS control module... Calling methods that use the controller mananger will fail...")

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
