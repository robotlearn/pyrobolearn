#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the abstract game controller interface class.

All the game controller interfaces inherit from the `GameControllerInterface` class defined here.

Dependencies:
- `pyrobolearn.tools.interfaces.InputInterface`
"""

from pyrobolearn.tools.interfaces.interface import InputOutputInterface

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GameControllerInterface(InputOutputInterface):
    r"""Game Controller Interface
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        super(GameControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
