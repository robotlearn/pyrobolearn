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

    The A matrix is the identity matrix, the variables being optimized are the joint velocities :math:`\dot{q}`, and
    the :math:`b` vector is :math:`\dot{q}_d + K_q (q_d - q)`.
    """
    pass


class CartesianPosition(KinematicTask):
    r"""Cartesian position kinematic task.

    The A matrix is the translational Jacobian matrix between the base (or world) and the specified link, the variables
    being optimized are the joint velocities :math:`\dot{q}`, and the :math:`b` vector is
    :math:`\dot{p}_d + K_p (p_d - p)`.
    """
    pass


class CartesianOrientation(KinematicTask):
    r"""Cartesian orientation kinematic task.

    The A matrix is the rotational Jacobian matrix, the variables being optimized are the joint velocities
    :math:`\dot{q}`, and the :math:`b` vector is :math:`\omega_d + K_o e_o` where
    :math:`e_o = -(\eta_d \espilon - \eta \epsilon_d + [\epsilon_d \times] \epsilon)`. The :math:`\eta` and
    :math:`\epsilon` are respectively the scalar and vector part of the quaternion.
    The quaternion error :math:`\Delta Q` is given by :math:`Q_d * Q_e^{-1}`.

    Once the optimal joint velocities :math:`\dot{q}^*` have been computed, they can be given to a velocity controller,
    or a position controller using :math:`q = q + \dot{q}^* dt`, with :math:`dt` being the control loop period.
    """
    pass


class Cartesian(KinematicTask):
    r"""Cartesian kinematic task

    The Cartesian task is the aggregate of a Cartesian position task and a Cartesian orientation task.
    """
    pass


class CoM(KinematicTask):
    r"""Center of Mass task

    The Center of Mass is a specific Cartesian position task.
    """
    pass


class VelocityManipulability(KinematicTask):
    r"""Velocity manipulability task

    The A matrix is the identity matrix, the variables being optimized are the joint velocities :math:`\dot{q}`, and
    the :math:`b` vector is 0.

    This task is often used to avoid joint singularities and their neighborhood.
    """
    pass


class MinVelocity(KinematicTask):
    r"""Min velocity task

    """
    pass
