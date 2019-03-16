#!/usr/bin/env python
"""Define the various actuators used in robotics.

This is decoupled from the robots such that actuators can be defined outside the robot class and can be selected at
run-time. This is useful for instance when a version of the robot has specific joint motors while another version has
other joint actuators. Additionally, this is important as more realistic motors can result in a better transfer from
simulation to reality.
"""

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Actuator(object):
    r"""Actuator class

    All actuator classes inherit from this class. Actuators such as motors are often attached to the robot joints.
    Other actuators such as speakers, leds, and others are attached to links.
    """
    def __init__(self):
        pass

    #     self.sim = simulator
    #
    # @property
    # def simulator(self):
    #     return self.sim
    #
    # @simulator.setter
    # def simulator(self, simulator):
    #     self.sim = simulator
