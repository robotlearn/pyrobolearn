#!/usr/bin/env python
r"""Provide the Cartesian acceleration task.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

# TODO: finish to implement this

import numpy as np

from pyrobolearn.priorities.tasks import JointAccelerationTask


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Arturo Laurenzi (C++)", "Songyan Xin (insight)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CartesianAccelerationTask(JointAccelerationTask):
    r"""Cartesian Acceleration Task

    The Cartesian acceleration task tries to impose a desired pose, velocity and acceleration profiles for a distal
    link with respect to a base link, or world frame.

    Before presenting the optimization problem, a small reminder. The acceleration is the time derivative of the
    velocity, i.e. :math:`a = \frac{dv}{dt}` where the cartesian velocities are related to joint velocities by
    :math:`v = J(q) \dot{q}` where :math:`J(q)` is the Jacobian, thus deriving that expression wrt time gives us:

    .. math:: a = \frac{d}{dt} v = \frac{d}{dt} J(q) \dot{q} = J(q) \ddot{q} + \dot{J}(q) \dot{q}.

    Now, we can formulate our minimization problem as:

    .. math:: || J(q) \ddot{q} - \dot{J} \dot{q} - (a_d + K_d (v_d - v) + K_p e) ||^2,

    where :math:`\ddot{q}` are the joint accelerations being optimized, :math:`a_d` is the desired cartesian
    acceleration, :math:`v_d = [\omega_d^\top, v` is the desired cartesian velocity, ...


    The above formulation is equivalent to the QP objective function :math:`||Ax - b||^2`, by setting
    :math:`A = J(q)`, :math:`x = \ddot{q}`, and :math:`b = - \dot{J} \dot{q} - (a_d + K_d (v_d - v) + K_p e)`.

    This task can, for instance, be used for foot pose tracking when this one is not in contact with the ground. If
    the foot is in contact, we switch to a foot damping task which can be achieved by setting
    :math:`a_d = v_d = e = 0` and thus we are trying to solve :math:`||J(q) \ddot{q} - \dot{J} \dot{q} - K_d v_d||^2`.


    Inverse dynamics
    ----------------

    Once the optimal joint accelerations :math:`\ddot{q}^*` have been computed, we can use inverse dynamics to
    compute the corresponding torques to apply on the joints. This is given by:

    .. math:: \tau = H(q) \ddot{q} + N(q,\dot{q)}

    where :math:`H(q)` is the inertia joint matrix, and N(q, \dot{q}) is a vector force that accounts for all the
    other forces acting on the system (Coriolis, centrifugal, gravity, external forces, friction, etc.).


    .. seealso:: `tasks/velocity/cartesian.py` and `tasks/torque/cartesian_impedance_control.py`
    """

    def __init__(self, model, distal_link, base_link=None, local_position=(0, 0, 0), x_desired=None,
                 dx_desired=None, kp=1., kd=1., weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            model (ModelInterface): model interface.
            distal_link (int, str): distal link id or name.
            base_link (int, str, None): base link id or name. If None, it will be the world.
            local_position (np.array[3]): local position on the distal link.
            x_desired (np.array[7], None): desired cartesian pose of distal link wrt the base.
            dx_desired (np.array[6], None): desired cartesian velocity of distal link wrt the base.
            kp (float, np.array[6,6]): stiffness gain.
            weight (float, np.array[6,6]): weight scalar or matrix associated to the task.
            constraints (list of Constraint): list of constraints associated with the task.
        """
        super(CartesianAccelerationTask, self).__init__(model=model, weight=weight, constraints=constraints)

        # set variables
        self.distal_link = distal_link
        self.base_link = base_link

        # pose = (position, quaternion)
        self.desired_pose = np.array([0]*6 + 1)
        self.current_pose = np.array([0]*6 + 1)

        # velocity = (angular, linear)
        self.desired_velocity = np.zeros(6)
        self.current_velocity = np.zeros(6)

        # acceleration = (angular, linear)
        self.desired_acceleration = np.zeros(6)
