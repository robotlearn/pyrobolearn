#!/usr/bin/env python
"""Define the Bullet-ROS Simulator API.

This is the main interface that communicates with the PyBullet simulator [1] and use ROS [3] to query the state of the
robot and send instructions to it. By defining this interface, it allows to decouple the PyRoboLearn framework from
the simulator. It also converts some data types to the ones required by PyBullet. For instance, some methods in
PyBullet do not accepts numpy arrays but only lists. The interface provided here makes the necessary conversions.
Using ROS to query the state of the robot, it changes the state of the robot in the simulator, and moving the robot
in the simulator results in the real robot to move. Virtual sensors and actuators can also be defined.

The signature of each method defined here are inspired by [1] but in accordance with the PEP8 style guide [2].

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`
* `pyrobolearn.simulators.bullet.Bullet`
* `pyrobolearn.simulators.ros.ROS`

References:
    [1] PyBullet: https://pybullet.org
    [2] PyBullet Quickstart Guide: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
    [3] ROS: http://www.ros.org/
    [4] PEP8: https://www.python.org/dev/peps/pep-0008/
"""

# TODO

import rospy

from pyrobolearn.simulators.simulator import Simulator
# from pyrobolearn.simulators.bullet import Bullet
# from pyrobolearn.simulators.ros import ROS


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BulletROS(Simulator):  # Bullet, ROS):
    r"""Bullet ROS

    Update the Bullet simulator based on the real robot(s): it updates the robot kinematic and dynamic state based on
    the values returned from the real robot(s).

    This can be useful for debug (check the differences between the real world and the simulated world), for virtual
    sensors, actuators, and forces, to map the real world to the simulated one, etc.
    """

    def __init__(self):
        super(BulletROS, self).__init__()
        raise NotImplementedError
