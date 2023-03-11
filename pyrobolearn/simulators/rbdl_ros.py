#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS-RBDL simulator

This 'simulator' is not per se a simulator, it communicates with the real robots in the real world using ROS [1], and
computes any necessary kinematic and dynamics information using the RBDL library [2].

Specifically, this 'simulator' starts the `roscore` (if not already running), then loads robot urdf models and creates
the necessary topics/services, and uses the rigid body dynamics library to compute kinematic and dynamic information
about the model.

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    [1] ROS: http://www.ros.org/
    [2] RBDL: https://rbdl.bitbucket.io/
"""

# TODO

import rospy
import rbdl

from pyrobolearn.simulators.simulator import Simulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RBDL_ROS(Simulator):
    r"""RBDL-ROS Interface.

    References:
        [1] ROS: http://www.ros.org/
        [2] RBDL: https://rbdl.bitbucket.io/
        [3] RBDL in Python: https://rbdl.bitbucket.io/dd/dee/_python_example.html
    """

    def __init__(self, **kwargs):
        super(RBDL_ROS, self).__init__(render=False)
        raise NotImplementedError

    def step(self, sleep_time=0):
        """Perform a step in the simulator, and sleep the specified amount of time.

        Args:
            sleep_time (float): amount of time to sleep after performing one step in the simulation.
        """
        pass

    def load_urdf(self, filename, position, orientation):
        # load the model in rbdl
        model = rbdl.loadModel(filename)
