#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Facial Expression Recognition Interface
"""
# TODO

from pyrobolearn.tools.interfaces.camera import CameraInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FERInterface(CameraInterface):
    r"""Facial Expression Recognition (FER) Interface

    References:
        [1] EmoPy - A deep neural net toolkit for emotion analysis via Facial Expression Recognition:
            https://github.com/thoughtworksarts/EmoPy
        [2] http://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/
        [3] https://github.com/a514514772/Real-Time-Facial-Expression-Recognition-with-DeepLearning
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        """
        Initialize the FER input interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring the
                next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        super(FERInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
