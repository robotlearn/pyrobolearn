#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the android interface.
"""

# TODO: finish this interface
# TODO: implement android application (Kotlin or Java)

from pyrobolearn.tools.interfaces.phones import PhoneInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class AndroidInterface(PhoneInterface):
    r"""Android phone/tablet interface
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        """
        Initialize the android interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring /
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        super(AndroidInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
