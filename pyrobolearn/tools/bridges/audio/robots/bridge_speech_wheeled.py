# Bridges between audio interface and wheeled robots

import numpy as np

from pyrobolearn.robots import WheeledRobot, AckermannWheeledRobot
from pyrobolearn.tools.interfaces.audio import SpeechRecognizerInterface
from pyrobolearn.tools.bridges.bridge import Bridge


class BridgeSpeechRecognizerWheeledRobot(Bridge):
    r"""Bridge Speech Wheeled Robot

    Bridge between the speech recognizer interface and a wheeled robot. You can give oral orders to the robot.
    """

    def __init__(self, interface, wheeled_robot, init_speed=1.):
        if not isinstance(interface, SpeechRecognizerInterface):
            raise TypeError("Expecting a speech recognizer interface")
        if not isinstance(wheeled_robot, WheeledRobot):
            raise TypeError("Expecting a wheeled robot")
        super(BridgeSpeechRecognizerWheeledRobot, self).__init__(interface)
        self.robot = wheeled_robot
        self.speed = init_speed

    def step(self):
        data = self.interface.data
        #print('data: {}'.format(data))
        if data == 'stop':
            self.robot.stop()
        elif data == 'move forward':
            self.robot.drive_forward(self.speed)
        elif data == 'move backward':
            self.robot.drive_backward(self.speed)
        elif data == 'turn right':
            pass
        elif data == 'turn left':
            pass
        elif data == 'faster':
            self.speed *= 2.
        elif data == 'slower':
            self.speed /= 2.
        elif data:
            pass
            #print('I do not know the meaning of {}'.format(data))


class BridgeSpeechRecognizerAckermannWheeledRobot(Bridge):
    r"""Bridge Speech Ackermann Wheeled Robot

    Bridge between the speech recognizer interface and a wheeled robot (with ackermann steering). You can give oral
    orders to the robot.
    """

    def __init__(self, interface, wheeled_robot, init_speed=1.):
        if not isinstance(interface, SpeechRecognizerInterface):
            raise TypeError("Expecting a speech recognizer interface")
        if not isinstance(wheeled_robot, AckermannWheeledRobot):
            raise TypeError("Expecting a wheeled robot of type Ackermann steering")
        super(BridgeSpeechRecognizerAckermannWheeledRobot, self).__init__(interface)
        self.robot = wheeled_robot
        self.speed = init_speed
        self.steering_angle = 0.

    def step(self):
        data = self.interface.data
        #print('data: {}'.format(data))
        if data == 'stop':
            self.robot.stop()
        elif data == 'move forward':
            self.robot.drive_forward(self.speed)
        elif data == 'move backward':
            self.robot.drive_backward(self.speed)
        elif data == 'turn right':
            self.robot.set_steering(np.deg2rad(-20))
        elif data == 'turn left':
            self.robot.set_steering(np.deg2rad(20))
        elif data == 'faster':
            self.speed *= 2.
        elif data == 'slower':
            self.speed /= 2.
        elif data:
            pass
            #print('I do not know the meaning of {}'.format(data))