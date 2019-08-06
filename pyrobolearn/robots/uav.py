#!/usr/bin/env python
"""Provide the Unmanned Aerial Vehicle (UAV) robot abstract classes.
"""

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class UAVRobot(Robot):
    r"""Unmanned Aerial Vehicle Robot

    Vehicles/Robots that operate in the air. These are also called drones.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the UAV.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(UAVRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

        self.propellers = []  # list of propellers id
        self.wings = []  # list of wing id

    @property
    def num_propellers(self):
        """Return the number of propellers"""
        return len(self.propellers)

    @property
    def num_wings(self):
        """Return the number of wings."""
        return len(self.wings)


class FixedWingUAV(UAVRobot):
    r"""Fixed Wing Robot

    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the Fixed wing UAV.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(FixedWingUAV, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)


class RotaryWingUAV(UAVRobot):
    r"""Rotary Wing UAV

    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the rotary UAV.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(RotaryWingUAV, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)

    def hover(self):
        pass

    def move(self, velocity):
        pass

    def ascend(self, speed=0):
        pass

    def descend(self, speed=0):
        pass

    def turn(self, speed=0):
        pass

    def turn_left(self, speed=0):
        pass

    def turn_right(self, speed=0):
        pass

    def move_forward(self, speed=0):
        pass

    def move_backward(self, speed=0):
        pass

    def move_left(self, speed=0):
        pass

    def move_right(self, speed=0):
        pass


class FlappingWingUAV(UAVRobot):
    r"""Flapping Wing Robot

    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        """
        Initialize the flapping UAV.

        Args:
            simulator (Simulator): simulator instance.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the robot base will be fixed in the world.
            scale (float): scaling factor that is used to scale the robot.
        """
        super(FlappingWingUAV, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
