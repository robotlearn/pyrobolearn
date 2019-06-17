#!/usr/bin/env python
"""Define the Bullet Simulator API.

This is the main interface that communicates with the PyBullet simulator [1]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
PyBullet. For instance, some methods in PyBullet do not accepts numpy arrays but only lists. The interface provided
here makes the necessary conversions.

The signature of each method defined here are inspired by [1,2] but in accordance with the PEP8 style guide [3].

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    [1] PyBullet: https://pybullet.org
    [2] PyBullet Quickstart Guide: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
    [3] PEP8: https://www.python.org/dev/peps/pep-0008/
"""

# TODO

import rospy

from pyrobolearn.simulators.simulator import Simulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ROSModel(object):
    r"""ROS Model

    """

    def __init__(self, filename):
        self.urdf = filename
        # get ros services and ros topics from URDF

        # create
        pass


class ROS(Simulator):
    r"""ROS Interface
    """

    def __init__(self):
        super(ROS, self).__init__()
        self.models = []

    # def load_urdf(self, filename, position=None, orientation=None):
    #     # load URDF: get ros services and ros topics
    #     model = ROSModel(filename)
    #
    #     # create id and add model to the list of models
    #     idx = len(self.models)
    #     self.models.append(model)
    #
    #     # return id
    #     return idx
