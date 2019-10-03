# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the YARP middleware API.

Dependencies in PRL:
* `pyrobolearn.simulators.middlewares.middleware.MiddleWare`
"""

# TODO
import os
import subprocess
import psutil
import signal
import importlib
import inspect

from pyrobolearn.simulators.middlewares.middleware import MiddleWare


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["YARP (IIT)", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class YARP(MiddleWare):
    r"""YARP Interface middleware

    This middleware can be given to the simulator which can then interact with robots.
    """

    def __init__(self, subscribe=False, publish=False, teleoperate=False, **kwargs):
        """
        Initialize the YARP middleware.

        Args:
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robots, and will read
              the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robots.
            teleoperate (bool): if True, it will move the robot based on the received or sent values based on the 2
              previous attributes :attr:`subscribe` and :attr:`publish`.
        """
        super(YARP, self).__init__(subscribe=subscribe, publish=publish, teleoperate=teleoperate)
