#!/usr/bin/env python
"""Define the main basic Camera interface

This defines the main basic camera interface from which all other interfaces which uses a camera inherit from.
"""

from pyrobolearn.tools.interfaces.interface import InputInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CameraInterface(InputInterface):
    r"""Camera Interface.

    This is the abstract class Camera Interface which is inherited from all the interfaces that use cameras
    such as webcams, kinects, asus xtion, etc.
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        super(CameraInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
