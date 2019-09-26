# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Bridges between speech interface and wheeled robots
"""

import numpy as np

from pyrobolearn.robots import WheeledRobot, DifferentialWheeledRobot, AckermannWheeledRobot
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


class BridgeSpeechRecognizerWheeledRobot(Bridge):
    r"""Bridge Speech Wheeled Robot

    Bridge between the speech recognizer interface and a wheeled robot. You can give oral orders to the robot.
    """

    def __init__(self, robot, interface=None, speed=1., priority=None, verbose=False):
        """
        Initialize the bridge between the speech recognizer interface and the wheeled robot.

        Args:
            robot (WheeledRobot): wheeled robot instance.
            interface (SpeechRecognizerInterface, None): speech recognizer interface. If None, it will instantiate it
                here, and launch it in a thread.
            speed (float): initial speed of the wheeled robot.
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        # check the robot
        if not isinstance(robot, WheeledRobot):
            raise TypeError("Expecting a wheeled robot, instead got: {}".format(robot))
        self.robot = robot
        self.speed = speed

        # check the interface
        if interface is None:
            interface = SpeechRecognizerInterface(use_thread=True, verbose=verbose)
        if not isinstance(interface, SpeechRecognizerInterface):
            raise TypeError("Expecting a speech recognizer interface, instead got: {}".format(type(interface)))

        super(BridgeSpeechRecognizerWheeledRobot, self).__init__(interface, priority=priority, verbose=verbose)

    def step(self, update_interface=False):
        """Perform a step with the bridge."""
        # update interface
        if update_interface:
            self.interface()


class BridgeSpeechRecognizerDifferentialWheeledRobot(BridgeSpeechRecognizerWheeledRobot):
    r"""Bridge between Speech Recognizer and Differential Wheeled Robot

    Bridge between the speech recognizer interface and a differential wheeled robot. You can give oral
    orders to the robot.
    """

    def __init__(self, robot, interface=None, speed=1., priority=None, verbose=False):
        """
       Initialize the bridge between the speech recognizer interface and the wheeled robot.

       Args:
           robot (DifferentialWheeledRobot): wheeled robot instance.
           interface (SpeechRecognizerInterface, None): speech recognizer interface. If None, it will instantiate it
                here, and launch it in a thread.
           speed (float): initial speed of the wheeled robot.
           verbose (bool): If True, print information on the standard output.
       """
        if not isinstance(robot, DifferentialWheeledRobot):
            raise TypeError("Expecting a wheeled robot of type Ackermann steering, instead got: {}".format(robot))

        super(BridgeSpeechRecognizerDifferentialWheeledRobot, self).__init__(robot=robot, interface=interface,
                                                                             speed=speed, priority=priority,
                                                                             verbose=verbose)

    def step(self, update_interface=False):
        """Perform a step with the bridge."""
        super(BridgeSpeechRecognizerDifferentialWheeledRobot, self).step(update_interface=update_interface)

        # get the data from the interface
        data = self.interface.data

        if self.verbose and data is not None:
            print("Bridge: the data = {}".format(data))

        if data == 'move forward':
            self.robot.drive_forward(self.speed)
        elif data == 'move backward':
            self.robot.drive_backward(self.speed)
        elif data == 'turn right':
            self.robot.turn_right(1)
        elif data == 'turn left':
            self.robot.turn_left(1)
        elif data == 'faster':
            self.speed *= 2.
        elif data == 'slower':
            self.speed /= 2.
        else:
            self.robot.stop()


class BridgeSpeechRecognizerAckermannWheeledRobot(BridgeSpeechRecognizerWheeledRobot):
    r"""Bridge Speech Ackermann Wheeled Robot

    Bridge between the speech recognizer interface and a wheeled robot (with ackermann steering). You can give oral
    orders to the robot.
    """

    def __init__(self, robot, interface=None, speed=1., priority=None, verbose=False):
        """
       Initialize the bridge between the speech recognizer interface and the wheeled robot.

       Args:
           robot (AckermannWheeledRobot): wheeled robot instance.
           interface (SpeechRecognizerInterface, None): speech recognizer interface. If None, it will instantiate it
               here, and launch it in a thread.
           speed (float): initial speed of the wheeled robot.
           verbose (bool): If True, print information on the standard output.
       """
        if not isinstance(robot, AckermannWheeledRobot):
            raise TypeError("Expecting a wheeled robot of type Ackermann steering, instead got: {}".format(robot))

        super(BridgeSpeechRecognizerAckermannWheeledRobot, self).__init__(robot=robot, interface=interface,
                                                                          speed=speed, priority=priority,
                                                                          verbose=verbose)
        self.steering_angle = 0.

    def step(self, update_interface=False):
        """Perform a step with the bridge."""
        super(BridgeSpeechRecognizerAckermannWheeledRobot, self).step(update_interface=update_interface)

        # get the data from the interface
        data = self.interface.data

        if self.verbose and data is not None:
            print("Bridge: the data = {}".format(data))

        if data == 'move forward':
            self.robot.drive_forward(self.speed)
        elif data == 'move backward':
            self.robot.drive_backward(self.speed)
        elif data == 'turn right':
            self.robot.steer(np.deg2rad(-20))
        elif data == 'turn left':
            self.robot.steer(np.deg2rad(20))
        elif data == 'faster':
            self.speed *= 2.
        elif data == 'slower':
            self.speed /= 2.
        else:
            self.robot.stop()
