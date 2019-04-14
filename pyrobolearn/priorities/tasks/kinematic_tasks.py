#!/usr/bin/env python
"""Provide the various kinematic tasks (i.e. objective functions) used in QP.

References:
    [1] "Quadratic Programming in Python" (https://scaron.info/blog/quadratic-programming-in-python.html), Caron, 2017
    [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    [3] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import rbdl
import numpy as np

from pyrobolearn.robots.robot import Robot
from pyrobolearn.priorities.tasks.task import Task


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi)", "Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class KinematicTask(Task):
    r"""Kinematic Task

    """
    pass


class BasePosition(KinematicTask):
    r"""Base Position Task

    """
    pass


class BaseOrientation(KinematicTask):
    r"""Base Orientation Task

    """
    pass


class BasePose(KinematicTask):
    r"""Base Pose Task

    """
    pass


class Postural(KinematicTask):
    r"""Postural kinematic task

    While optimizing, the robot can get to weird configurations. This postural kinematic task tries to keep the
    robot's kinematic configuration close to the given default joint positions. This task is usually put at the end of
    the stack of tasks, once all the other tasks have been fulfilled.
    """
    pass


class Cartesian(KinematicTask):
    r"""Cartesian kinematic task

    """
    pass


class CoM(KinematicTask):
    r"""Center of Mass task

    """
    pass


class VelocityManipulability(KinematicTask):
    r"""Velocity manipulability task

    """
    pass


class MinVelocity(KinematicTask):
    r"""Min velocity task

    """
    pass
