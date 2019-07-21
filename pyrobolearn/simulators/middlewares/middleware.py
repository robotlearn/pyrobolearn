#!/usr/bin/env python
"""Define the abstract middleware API.

Dependencies in PRL:
* NONE
"""

# TODO
import os
import subprocess
import psutil
import signal
import importlib
import inspect


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MiddleWare(object):
    r"""Middleware (abstract) class

    Middlewares can be provided to simulators which can then use them to send/receive messages.
    """
    pass
