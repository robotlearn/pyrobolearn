#!/usr/bin/env python
"""Provide the Wheeled robot abstract classes.
"""

import numpy as np

from pyrobolearn.robots.robot import Robot


class WheeledRobot(Robot):

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(WheeledRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)

        self.wheels = []
        self.wheel_directions = []

    ##############
    # Properties #
    ##############

    @property
    def num_wheels(self):
        """Return the number of wheels"""
        return len(self.wheels)

    ###########
    # Methods #
    ###########

    def getNumberOfWheels(self):
        """
        Return the number of wheels.

        Returns:
            int: the number of wheels.
        """
        return self.num_wheels

    def getWheelIds(self, wheels=None):
        """
        Return the wheel id associated with the given wheel number(s)/name(s).

        Args:
            wheels (int, str): wheel number(s) (has to be between 0 and the total number of wheels), or wheel name(s)

        Returns:
            list[int]: wheel id(s)
        """
        if wheels:
            if isinstance(wheels, int):
                return self.wheels[wheels]
            elif isinstance(wheels, str):
                return self.wheels[self.getLinkIds(wheels)]
            elif isinstance(wheels, (list, tuple)):
                wheelIds = []
                for wheel in wheels:
                    if isinstance(wheels, int):
                        wheelIds.append(self.wheels[wheels])
                    elif isinstance(wheels, str):
                        wheelIds.append(self.wheels[self.getLinkIds(wheels)])
                    else:
                        raise TypeError("Expecting a str or int for items in wheels")
                return wheelIds
        return self.wheels

    def getWheelNames(self, wheelId=None):
        """Return the wheel names associated to the given ids"""
        return self.getLinkNames(wheelId)

    def drive(self, speed):
        if isinstance(speed, (int, float)):
            speed = speed * np.ones(self.num_wheels)
            speed = speed * self.wheel_directions
        self.setJointVelocities(speed, self.wheels)

    def stop(self):
        self.setJointVelocities(np.zeros(self.num_wheels), self.wheels)

    def driveForward(self, speed):
        self.drive(speed)

    def driveBackward(self, speed):
        self.drive(-speed)

    def turnRight(self):
        pass

    def turnLeft(self):
        pass


class DifferentialWheeledRobot(WheeledRobot):
    r"""Differential Wheeled Robot

    "A differential wheeled robot is a mobile robot whose movement is based on two separately driven wheels placed
    on either side of the robot body. It can thus change its direction by varying the relative rate of rotation of
    its wheels and hence does not require an additional steering motion. To balance the robot, additional wheels or
    casters may be added." [1]

    Check also [2] for the different types of drive.

    References:
        [1] Wikipedia: https://en.wikipedia.org/wiki/Differential_wheeled_robot
        [2] "Pros and cons for different types of drive selection":
            https://robohub.org/pros-and-cons-for-different-types-of-drive-selection/
        [3] Wheel Control Theory:
            http://www.robotplatform.com/knowledge/Classification_of_Robots/wheel_control_theory.html
    """

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(DifferentialWheeledRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase,
                                                       scaling)


class AckermannWheeledRobot(WheeledRobot):
    r"""Ackermann steering Wheeled Robot

    "Ackermann steering geometry is a geometric arrangement of linkages in the steering of a car or other vehicle
    designed to solve the problem of wheels on the inside and outside of a turn needing to trace out circles of
    different radii." [1] Wheels in this particular configuration do not need to slip in order to turn.

    Check also [2] for the different types of drive.

    References:
        [1] Wikipedia: https://en.wikipedia.org/wiki/Ackermann_steering_geometry
        [2] "Pros and cons for different types of drive selection":
            https://robohub.org/pros-and-cons-for-different-types-of-drive-selection/
        [3] Wheel Control Theory:
            http://www.robotplatform.com/knowledge/Classification_of_Robots/wheel_control_theory.html
    """
    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(AckermannWheeledRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase,
                                                    scaling)

        self.steering = 0   # id of steering joint

    def setSteering(self, angle):
        """Set steering angle"""
        pass
