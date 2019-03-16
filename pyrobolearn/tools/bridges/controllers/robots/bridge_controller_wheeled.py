
# Bridges between controller interface and wheeled robots

from pyrobolearn.robots import WheeledRobot, AckermannWheeledRobot
from pyrobolearn.tools.interfaces.controllers import XboxControllerInterface, XboxOneControllerInterface
from pyrobolearn.tools.bridges.bridge import Bridge


class BridgeXboxWheeledRobot(Bridge):
    r"""Bridge Xbox Wheeled Robot

    Bridge between the Xbox controller interface and a wheeled robot. You can move the robot using the joystick.
    """

    def __init__(self, interface, wheeled_robot):
        # quick checks
        if not isinstance(interface, XboxControllerInterface):
            raise TypeError("Expecting a speech recognizer interface")
        if not isinstance(wheeled_robot, WheeledRobot):
            raise TypeError("Expecting a wheeled robot")

        # call super class
        super(BridgeXboxWheeledRobot, self).__init__(interface)
        self.robot = wheeled_robot
        self.speed = 1.

    def step(self):
        x,y = self.interface.LJ


class BridgeXboxOneWheeledRobot(Bridge):
    r"""Bridge Xbox Wheeled Robot

    Bridge between the Xbox One controller interface and a wheeled robot. You can move the robot using the joystick.
    """

    def __init__(self, interface, wheeled_robot):
        # quick check
        if not isinstance(interface, XboxOneControllerInterface):
            raise TypeError("Expecting a speech recognizer interface")
        if not isinstance(wheeled_robot, WheeledRobot):
            raise TypeError("Expecting a wheeled robot")

        super(BridgeXboxOneWheeledRobot, self).__init__(interface)
        self.robot = wheeled_robot
        self.speed = 1.

    def step(self):
        x,y = self.interface.LJ


class BridgeXboxOneAckermannWheeledRobot(Bridge):
    r"""Bridge Xbox One Ackermann Wheeled Robot

    Bridge between the Xbox One controller interface and a wheeled robot. You can move the robot using the joystick.
    """

    def __init__(self, interface, wheeled_robot):
        if not isinstance(interface, XboxOneControllerInterface):
            raise TypeError("Expecting a speech recognizer interface")
        if not isinstance(wheeled_robot, AckermannWheeledRobot):
            raise TypeError("Expecting a wheeled robot")
        super(BridgeXboxOneAckermannWheeledRobot, self).__init__(interface)
        self.robot = wheeled_robot
        self.speed = 1.

    def step(self):
        x,y = self.interface.LJ
        self.robot.setSteering(-x / 2.)
        self.robot.driveForward(y * self.speed)

        if self.interface.A:
            print('increasing speed +1')
            self.speed += 1.
        if self.interface.B:
            print('decreasing speed -1')
            self.speed -= 1.
            if self.speed < 1.:
                self.speed = 1.
