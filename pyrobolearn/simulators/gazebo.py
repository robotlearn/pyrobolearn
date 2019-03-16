#!/usr/bin/env python
"""Define the Gazebo Simulator API.

This is the main interface that communicates with the Gazebo simulator [1]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
Gazebo. Note that this simulator does not use any ROS packages.

Warnings: The use of this simulator necessitates Python wrappers for the Gazebo simulator [1]. Currently, none are
provided, and thus the interface defined here is currently unusable.

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    [1] Gazebo: http://gazebosim.org/
"""

from simulator import Simulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Gazebo(Simulator):
    r"""Gazebo Simulator interface.

    References:
        [1] Gazebo: http://gazebosim.org/
    """

    def __init__(self, render=True):
        super(Gazebo, self).__init__(render=render)
        raise NotImplementedError
