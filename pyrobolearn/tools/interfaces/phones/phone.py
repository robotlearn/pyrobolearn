#!/usr/bin/env python
"""Define the smartphone and tablet interface.

These interfaces allows you to get sensor values sent with your phone or tablet, and values returned by the
corresponding (Android / IPhone) application.
"""

from pyrobolearn.tools.interfaces import InputOutputInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: define common interface for phone and tablet?

class PhoneInterface(InputOutputInterface):
    r"""Smart-Phone Interface
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        """
        Initialize the smartphone interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring /
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        super(PhoneInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)


# class TabletInterface(InputOutputInterface):
#     r"""Tablet Interface
#     """
#
#     def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
#         """
#         Initialize the tablet interface.
#
#         Args:
#             use_thread (bool): If True, it will run the interface in a separate thread than the main one.
#                 The interface will update its data automatically.
#             sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring /
#                 setting the next sample.
#             verbose (bool): If True, it will print information about the state of the interface. This is let to the
#                 programmer what he / she wishes to print.
#         """
#         super(TabletInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
