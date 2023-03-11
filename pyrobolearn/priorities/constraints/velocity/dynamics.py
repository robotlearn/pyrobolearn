#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the dynamics constraint.


The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
"""

# TODO: finish to implement this class

import numpy as np

from pyrobolearn.priorities.constraints.constraint import Constraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["OpenSoT (Alessio Rocchi and Enrico Mingo Hoffman, C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DynamicsConstraint(Constraint):
    r"""Dynamics constraint.

    From the documentation of the framework of [1]: "the DynamicsConstraint class implements constraints on joint
    velocities due to dynamics feasibility.

    The constraint is written as:

    .. math:: u_{min} \leq (M/dT) dq \leq u_{max}

    with:

    .. math::

        u_{min} = \tau_{min} dT - N(q, \dot{q}) dT + M \dot{q} - dT J_c^\top F_c \\
        u_{max} = \tau_{max} dT - N(q, \dot{q}) dT + M \dot{q} - dT J_c^\top F_c \\\\
        N(q, \dot{q}) = C(q, \dot{q}) \dot{q} + g(q) \\
        J_c = [J_{c,1} \cdot J_{c,N}]^\top \\
        F_c = [F_{c,1} \cdot J_{c,N}]^\top

    where :math:`\dot{q}` is the velocity in the previous step, :math:`J_c` is the Jacobian of all the contacts (here,
    we consider these Jacobians from the base link to the force/torque sensor frames), :math:`F_c` are the contact
    forces (at the force/torque sensor frames transformed in the base link)."


    The implementation of this class is inspired by [1] (which is licensed under the LGPLv2).

    References:
        - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    """

    def __init__(self, model):
        super(DynamicsConstraint, self).__init__(model)
