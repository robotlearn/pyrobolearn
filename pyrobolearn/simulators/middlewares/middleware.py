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

    def __init__(self, subscribe=False, publish=False, teleoperate=False):
        """
        Initialize the middleware to communicate.

        Args:
            subscribe (bool): if True, it will subscribe to the topics associated to the loaded robots, and will read
              the values published on these topics.
            publish (bool): if True, it will publish the given values to the topics associated to the loaded robots.
            teleoperate (bool): if True, it will move the robot based on the received or sent values based on the 2
              previous attributes :attr:`subscribe` and :attr:`publish`.
        """
        # set variables
        self.subscribe = subscribe
        self.publish = publish
        self.teleoperate = teleoperate

    @property
    def subscribe(self):
        return self._subscribe

    @subscribe.setter
    def subscribe(self, subscribe):
        self._subscribe = bool(subscribe)

    @property
    def publish(self):
        return self._publish

    @publish.setter
    def publish(self, publish):
        self._publish = bool(publish)

    @property
    def teleoperate(self):
        return self._teleoperate

    @teleoperate.setter
    def teleoperate(self, teleoperate):
        self._teleoperate = bool(teleoperate)
