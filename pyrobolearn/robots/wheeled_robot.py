#!/usr/bin/env python
"""Provide the Wheeled robot abstract classes.
"""

import numpy as np

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WheeledRobot(Robot):
    r"""Wheeled robot

    This type of robots has wheels.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        super(WheeledRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

        self.wheels = []
        self.wheel_directions = []

    ##############
    # Properties #
    ##############

    @property
    def num_wheels(self):
        """Return the number of wheels."""
        return len(self.wheels)

    ###########
    # Methods #
    ###########

    def get_wheel_ids(self, wheels=None):
        """
        Return the wheel id associated with the given wheel number(s)/name(s).

        Args:
            wheels (int, str): wheel number(s) (has to be between 0 and the total number of wheels), or wheel name(s)

        Returns:
            list[int]: wheel id(s)
        """
        if wheels is not None:
            if isinstance(wheels, int):
                return self.wheels[wheels]
            elif isinstance(wheels, str):
                return self.wheels[self.get_link_ids(wheels)]
            elif isinstance(wheels, (list, tuple)):
                wheel_ids = []
                for wheel in wheels:
                    if isinstance(wheel, int):
                        wheel_ids.append(self.wheels[wheel])
                    elif isinstance(wheel, str):
                        wheel_ids.append(self.wheels[self.get_link_ids(wheel)])
                    else:
                        raise TypeError("Expecting a str or int for items in wheels")
                return wheel_ids
        return self.wheels

    def drive(self, speed):
        if isinstance(speed, (int, float)):
            speed = speed * np.ones(self.num_wheels)
            speed = speed * self.wheel_directions
        self.set_joint_velocities(speed, self.wheels)

    def stop(self):
        self.set_joint_velocities(np.zeros(self.num_wheels), self.wheels)

    def drive_forward(self, speed):
        self.drive(speed)

    def drive_backward(self, speed):
        self.drive(-speed)

    def turn_right(self):
        pass

    def turn_left(self):
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

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        super(DifferentialWheeledRobot, self).__init__(simulator, urdf, position, orientation, fixed_base,
                                                       scale)


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

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        super(AckermannWheeledRobot, self).__init__(simulator, urdf, position, orientation, fixed_base,
                                                    scale)

        self.steering = 0   # id of steering joint

    def set_steering(self, angle):
        """Set steering angle"""
        pass
