#!/usr/bin/env python
"""Define the main basic Camera interface

This defines the main basic camera interface from which all other interfaces which uses a camera inherit from.
"""

from pyrobolearn.tools.interfaces.interface import InputInterface

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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
        """
        Initialize the camera input interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring or
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        super(CameraInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
        self.frame = None
