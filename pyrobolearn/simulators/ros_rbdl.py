#!/usr/bin/env python
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
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ROS_RBDL(Simulator):
    r"""ROS-RBDL Interface.

    References:
        [1] ROS: http://www.ros.org/
        [2] RBDL: https://rbdl.bitbucket.io/
        [3] RBDL in Python: https://rbdl.bitbucket.io/dd/dee/_python_example.html
    """

    def __init__(self):
        super(ROS_RBDL, self).__init__()

    def step(self):
        """Perform a step in the simulator."""
        pass

    def load_urdf(self, filename, position, orientation):
        # load the model in rbdl
        model = rbdl.loadModel(filename)
