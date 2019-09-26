
# import middleware abstract class
from .middleware import MiddleWare

# import ROS
from .ros import ROS


# # define decorator
# def middleware(function):
#     def wrapper(self, *args, **kwargs):
#         if self.middleware is None:
#             return function(*args, **kwargs)
#     return wrapper
