#!/usr/bin/env python
"""Bridges between audio interface and rotatory wing robots
"""

from pyrobolearn.robots import RotaryWingUAV
from pyrobolearn.tools.interfaces.audio.audio import SpeechRecognizerInterface
from pyrobolearn.tools.bridges.bridge import Bridge


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BridgeSpeechRecognizerRotatoryUAV(Bridge):
    r"""Bridge Speech Wheeled Robot

    Bridge between the speech recognizer interface and a wheeled robot. You can give oral orders to the robot.
    """

    def __init__(self, interface, uav_robot, init_speed=1.):
        if not isinstance(interface, SpeechRecognizerInterface):
            raise TypeError("Expecting a speech recognizer interface")
        if not isinstance(uav_robot, RotaryWingUAV):
            raise TypeError("Expecting a wheeled robot")
        super(BridgeSpeechRecognizerRotatoryUAV, self).__init__(interface)
        self.robot = uav_robot
        self.speed = init_speed

    def step(self):
        data = self.interface.data
        data = data.split()
        # print('data: {}'.format(data))
        if data[0] == 'stop' or data[0] == 'stay':
            self.robot.stop()
        elif data[-1] == 'higher':
            pass
        elif data[-1] == 'lower':
            pass
        elif data[-1] == 'forward':
            pass
        elif data[-1] == 'backward':
            pass
        elif data == 'turn right':
            pass
        elif data == 'turn left':
            pass
        elif data[-1] == 'right':
            pass
        elif data[-1] == 'left':
            pass
        elif data[-1] == 'faster':
            self.speed *= 2.
        elif data[-1] == 'slower':
            self.speed /= 2.
        elif data:
            pass
            # print('I do not know the meaning of {}'.format(data))
