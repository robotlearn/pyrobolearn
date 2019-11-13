#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bridges between speech interface and quadcopter
"""

from pyrobolearn.robots import Quadcopter
from pyrobolearn.tools.interfaces.audio.speech import SpeechRecognizerInterface
from pyrobolearn.tools.bridges.bridge import Bridge


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BridgeSpeechRecognizerQuadcopter(Bridge):
    r"""Bridge Speech Wheeled Robot

    Bridge between the speech recognizer interface and a wheeled robot. You can give oral orders to the robot.
    """

    def __init__(self, robot, interface=None, speed=10., priority=None, verbose=False, *args, **kwargs):
        """
        Initialize the bridge between the speech recognizer interface and the quadcopter robot.

        Args:
            robot (Quadcopter): quadcopter robot instance.
            interface (SpeechRecognizerInterface, None): speech recognizer interface. If None, it will instantiate it
                here, and launch it in a thread.
            speed (float): initial speed of the quadcopter robot.
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        # check the robot
        if not isinstance(quadcopter, Quadcopter):
            raise TypeError("Expecting the given 'quadcopter' to be an instance of `Quadcopter`, but got instead: "
                            "{}".format(type(robot)))
        self.robot = robot
        self.speed = speed

        if interface is None:
            interface = SpeechRecognizerInterface(use_thread=True, verbose=verbose)
        if not isinstance(interface, SpeechRecognizerInterface):
            raise TypeError("Expecting the given 'interface' to be an instance of `SpeechRecognizerInterface`, but "
                            "got instead: {}".format(type(interface)))

        super(BridgeSpeechRecognizerQuadcopter, self).__init__(interface, priority=priority, verbose=verbose)

    def step(self, update_interface=False):
        """Perform a step with the bridge."""
        # update interface
        if update_interface:
            self.interface()

        # get the data
        data = self.interface.data

        if self.verbose and data is not None:
            print("Bridge: the data = {}".format(data))

        # split the data
        data = data.split()

        if data[-1] == 'higher':
            self.robot.ascend(speed=self.speed)
        elif data[-1] == 'lower':
            self.robot.descend(speed=self.speed)
        elif data[-1] == 'forward':
            self.robot.move_forward(speed=self.speed)
        elif data[-1] == 'backward':
            self.robot.move_backward(speed=self.speed)
        elif data == 'turn right':
            self.robot.turn_right(speed=self.speed)
        elif data == 'turn left':
            self.robot.turn_left(speed=self.speed)
        elif data[-1] == 'right':
            self.robot.move_right(speed=self.speed)
        elif data[-1] == 'left':
            self.robot.move_left(speed=self.speed)
        elif data[-1] == 'faster':
            self.speed *= 2.
        elif data[-1] == 'slower':
            self.speed /= 2.
        else:
            self.robot.hover()
