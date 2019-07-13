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
        """
        Initialize the wheeled robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[3]): Cartesian world position.
            orientation (np.array[4]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
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

    def move(self, velocity):
        """Move the robot at the specified 2D velocity vector.

        Args:
            velocity (np.array[2]): 2D velocity vector defined in the xy plane. The magnitude represents the speed.
        """
        pass

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


class DifferentialWheeledRobot(WheeledRobot):
    r"""Differential Wheeled Robot

    "A differential wheeled robot is a mobile robot whose movement is based on two separately driven wheels placed
    on either side of the robot body. It can thus change its direction by varying the relative rate of rotation of
    its wheels and hence does not require an additional steering motion. To balance the robot, additional wheels or
    casters may be added." [1]

    Check also [2] for the different types of drive.

    The kinematics of these kind of platforms (with two wheels) can be described mathematically by [4]:

    .. math::

        v &= \frac{r (\omega_R +  \omega_L)}{2} \\
        \omega &= \frac{r (\omega_R - \omega_L)}{d}

    where :math:`\omega_R` (resp. :math:`\omega_L`) is the angular velocity of the right (resp. left) wheel,
    :math:`v` is the driving velocity of the platform, :math:`\omega` is its steering velocity, :math:`r` is the
    radius of the wheels and :math:`d` is the distance between their centers.

    This formulation is equivalent to:

    .. math::

        \omega_R &= v + \frac{d}{2r} \omega \\
        \omega_L &= v - \frac{d}{2r} \omega

    References:
        - [1] Wikipedia: https://en.wikipedia.org/wiki/Differential_wheeled_robot
        - [2] "Pros and cons for different types of drive selection":
            https://robohub.org/pros-and-cons-for-different-types-of-drive-selection/
        - [3] Wheel Control Theory:
            http://www.robotplatform.com/knowledge/Classification_of_Robots/wheel_control_theory.html
        - [4] "Robotics: Modelling, Planning and Control" (section 11.2), Siciliano et al., 2010
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the differential wheeled robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[3]): Cartesian world position.
            orientation (np.array[4]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(DifferentialWheeledRobot, self).__init__(simulator, urdf, position, orientation, fixed_base,
                                                       scale)

    def turn(self, speed):
        """Turn the robot. If the speed is positive, turn to the left, otherwise turn to the right (using the
        right-hand rule).

        Args:
            speed (float): speed to turn to the left (if speed is positive) or to the right (if speed is negative).
        """
        pass

    def turn_right(self, speed):
        """Turn the quadcopter to the right.

        Args:
            speed (float): positive speed to turn to the right.
        """
        self.turn(speed)

    def turn_left(self, speed):
        """Turn the robot to the left.

        Args:
            speed (float): positive speed to turn to the left.
        """
        self.turn(-speed)


class AckermannWheeledRobot(WheeledRobot):
    r"""Ackermann steering Wheeled Robot

    "Ackermann steering geometry is a geometric arrangement of linkages in the steering of a car or other vehicle
    designed to solve the problem of wheels on the inside and outside of a turn needing to trace out circles of
    different radii." [1] Wheels in this particular configuration do not need to slip in order to turn.

    Check also [2] for the different types of drive.

    References:
        - [1] Wikipedia: https://en.wikipedia.org/wiki/Ackermann_steering_geometry
        - [2] "Pros and cons for different types of drive selection":
            https://robohub.org/pros-and-cons-for-different-types-of-drive-selection/
        - [3] Wheel Control Theory:
            http://www.robotplatform.com/knowledge/Classification_of_Robots/wheel_control_theory.html
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the Ackermann wheeled robot.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[3]): Cartesian world position.
            orientation (np.array[4]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(AckermannWheeledRobot, self).__init__(simulator, urdf, position, orientation, fixed_base,
                                                    scale)

        self.steering = 0   # id of steering joint

    def move(self, velocity):
        """Move the robot at the specified 2D velocity vector.

        Args:
            velocity (np.array[2]): 2D velocity vector defined in the xy plane. The magnitude represents the speed.
        """
        if velocity[0] > 0:  # forward
            angle = np.arctan2(velocity[1], velocity[0])
            magnitude = np.linalg.norm(velocity)
            self.steer(angle)
            self.drive_forward(magnitude)
        else:  # backward
            angle = np.arctan2(velocity[1], -velocity[0])
            magnitude = np.linalg.norm(velocity)
            self.steer(angle)
            self.drive_backward(magnitude)

    def steer(self, angle):
        """Set steering angle. If the angle is positive, turn to the left, otherwise turn to the right (using the
        right-hand rule).

        Args:
            angle (float): steering angle. If the angle is positive, steer to the left, otherwise, steer to the right.
        """
        pass

    def steer_left(self, angle):
        """
        Steer to the left at the specified angle.

        Args:
            angle (float): positive steering angle.
        """
        self.steer(angle)

    def steer_right(self, angle):
        """
        Steer to the right at the specified angle.

        Args:
            angle (float): positive steering angle.
        """
        self.steer(-angle)
